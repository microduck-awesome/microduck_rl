#!/usr/bin/env python3
"""Local browser controls for the existing CPU MuJoCo + SC0090 BAM rehearsal.

Only the main thread touches MuJoCo, ONNX, BAM or the renderer. HTTP handlers
exchange immutable snapshots and validated input through a bounded mailbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import mimetypes
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote

REPO = Path(__file__).resolve().parents[1]
RL_REPO = Path(os.environ.get("MICRODUCK_RL_DIR", str(REPO.parent))).resolve()
sys.path.insert(0, str(RL_REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))

import mujoco
import numpy as np
import onnxruntime as ort

import infer_policy as rehearsal
from checkpoint_library import CheckpointLibrary

CONTROL_DT = 0.02  # Trained 50 Hz policy, four 5 ms physics steps.
PHYSICS_DT = 0.005
COMMAND_TIMEOUT = 0.4  # Browser sends at 10 Hz; lost focus/link stops commands.
STABLE_HOLD = 0.5  # Same continuous nominal-standing hold as recovery evaluation.
POSES = ("standing", "sitting", "prone", "supine", "left_side", "right_side")
MODES = ("auto", "walking", "recovery")


def default_models():
    manifest = json.loads((REPO / 'models/manifest.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported model manifest')
    motor_hash = hashlib.sha256(rehearsal.SC0090_MODEL_PATH.read_bytes()).hexdigest()
    if manifest['motor_model_sha256'] != motor_hash:
        raise ValueError('Demo policies were exported for a different SC0090 model')
    paths = []
    for name in ('walking', 'recovery'):
        entry = manifest['policies'][name]
        path = (REPO / 'models' / entry['file']).resolve()
        if not path.is_relative_to(REPO / 'models'):
            raise ValueError('Policy path is outside models/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{name} policy checksum mismatch')
        paths.append(path)
    return paths


def finite_vector(value, size, limits):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {size} finite numbers")
    return np.clip(array, -np.asarray(limits), limits).tolist()


class Mailbox:
    """One active browser lease; sequence numbers reject reordered requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None
        self.seq = -1
        self.received = -math.inf
        self.latest = {}
        self.events = []
        self.frame = b'{}'

    def submit(self, message, now):
        client = message.get("client")
        seq = message.get("seq")
        if not isinstance(client, str) or not 1 <= len(client) <= 80:
            raise ValueError("Invalid browser ID")
        if type(seq) is not int or seq < 0:
            raise ValueError("Invalid sequence")
        mode = message.get("mode", "auto")
        if mode not in MODES:
            raise ValueError("Invalid policy mode")
        state = {
            "twist": finite_vector(message.get("twist", [0, 0, 0]), 3, [.3, .15, 1.0]),
            "mode": mode,
        }
        event = message.get("event")
        if event is not None and event not in (*POSES, "push", "pause"):
            raise ValueError("Invalid event")
        load = message.get('load')
        if load is not None:
            if (not isinstance(load, dict) or load.get('slot') not in ('walking','recovery')
                    or not isinstance(load.get('id'), str) or not 1 <= len(load['id']) <= 80
                    or event is not None):
                raise ValueError('Invalid checkpoint selection')
            event = {'load': {'slot':load['slot'], 'id':load['id']}}
        with self.lock:
            if self.owner != client:
                if now - self.received < COMMAND_TIMEOUT:
                    return False
                self.owner, self.seq = client, -1
                self.events.clear()
            if seq <= self.seq:
                return False
            if event is not None and len(self.events) >= 16:
                raise ValueError("Input queue full")
            self.seq, self.received, self.latest = seq, now, state
            if event is not None:
                self.events.append(event)
        return True

    def consume(self, now):
        with self.lock:
            state = dict(self.latest)
            events, self.events = self.events, []
            if now - self.received >= COMMAND_TIMEOUT:
                state["twist"] = [0., 0., 0.]
                events = []  # Never execute stale reset/push after a stalled loop.
        return state, events

    def publish(self, frame):
        blob = json.dumps(frame, allow_nan=False).encode()
        with self.lock:
            self.frame = blob

    def snapshot(self):
        with self.lock:
            return self.frame


class Demo:
    def __init__(self, walking, recovery):
        bam = rehearsal.load_bam_model(rehearsal.BAM_KP_FW, 12., None)
        self.model, self.data, self.motor, _ = rehearsal.load_mujoco_with_bam(
            str(RL_REPO / rehearsal.MICRODUCK_XML), bam, PHYSICS_DT, .1,
            rehearsal.BAM_VIN_MIN,
        )
        # Match the training solver and FULL_COLLISION overrides; scene.xml
        # alone has MuJoCo's default Euler/100 iterations and contact params.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20
        self.foot_geoms = {self.model.geom(f"{side}_foot_collision").id for side in ("left", "right")}
        self.floor_geom = self.model.geom("floor").id
        for geom in range(self.model.ngeom):
            if (self.model.geom(geom).name or "").endswith("_collision"):
                self.model.geom_condim[geom] = 3 if geom in self.foot_geoms else 1
            if geom in self.foot_geoms:
                self.model.geom_priority[geom] = 1
                self.model.geom_friction[geom, 0] = 1.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.policy = rehearsal.PolicyInference(
            self.model, self.data, str(walking), standing_onnx_path=str(recovery),
            bam_ctrl=self.motor, use_projected_gravity=True, new_cmd_obs=True,
            session_options=options, providers=["CPUExecutionProvider"],
        )
        self.sessions = {"walking": self.policy.walking_session,
                         "recovery": self.policy.standing_session}
        names = [self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]]
        self.joint_names = names
        for session in self.sessions.values():
            self.validate_session(session)
        self.labels = {"walking": walking.name, "recovery": recovery.name}
        self.mode = "auto"
        self.paused = False
        self.error = None
        self.reset("standing")

    def validate_session(self, session):
        if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
            raise ValueError('需要归一化的 61 维输入、14 维输出模型')
        meta = session.get_modelmeta().custom_metadata_map
        if meta.get('joint_names', '').split(',') != self.joint_names:
            raise ValueError('模型关节顺序与仿真不一致')
        if float(meta.get('action_scale', 'nan')) != 1.:
            raise ValueError('模型动作缩放与训练配置不一致')
        action = session.run(None, {session.get_inputs()[0].name:np.zeros((1,61), dtype=np.float32)})[0]
        if action.shape != (1,14) or not np.isfinite(action).all():
            raise ValueError('模型预检产生了无效数值')

    def load_policy(self, slot, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        candidate = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.validate_session(candidate)
        # Commit only after the candidate has passed all checks. A failed load
        # leaves both the old session and the physical trajectory untouched.
        self.sessions[slot] = candidate
        if slot == 'walking':
            self.policy.walking_session = candidate
        else:
            self.policy.standing_session = candidate
        self.labels[slot] = path.name
        self.reset('standing')

    def reset(self, pose):
        if pose not in POSES:
            raise ValueError("Unknown spawn")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.policy.joint_qpos_indices] = self.policy.default_pose
        qa = self.policy._trunk_qpos_adr
        self.data.qpos[qa:qa+3] = [0, 0, .12 if pose == "standing" else .07]
        s = math.sqrt(.5)
        quats = {"standing": [1, 0, 0, 0], "sitting": [1, 0, 0, 0],
                 "prone": [s, 0, s, 0], "supine": [s, 0, -s, 0],
                 "left_side": [.5, -.5, -.5, .5],
                 "right_side": [.5, .5, -.5, -.5]}
        self.data.qpos[qa+3:qa+7] = quats[pose]
        if pose == "sitting":
            # microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES (trained SIT).
            angles = {1: 0., 2: -.4079, 3: 1.35, 4: 0.,
                      10: 0., 11: .4079, 12: -1.35, 13: 0.}
            for index, angle in angles.items():
                self.data.qpos[self.policy.joint_qpos_indices[index]] = angle
        mujoco.mj_forward(self.model, self.data)
        rehearsal.reset_bam_controller(self.motor)
        self.policy.last_action.fill(0)
        self.policy.command.fill(0)
        self.policy.vel_cmd.fill(0)
        self.spawn = pose
        self.hold = 0.
        self.rise_time = None
        self.recovering = pose != "standing"
        self.recovery_started_at = 0. if self.recovering else None
        self.active = "recovery" if self.recovering else "walking"
        self.paused, self.error = False, None

    def posture(self):
        gravity = self.policy.get_projected_gravity()
        tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
        height = float(self.data.qpos[self.policy._trunk_qpos_adr + 2])
        return height, tilt

    def step(self, twist):
        height, tilt = self.posture()
        # Hysteresis prevents rapid policy swaps while finishing a recovery.
        if height < .08 or tilt > math.radians(45):
            if not self.recovering:
                self.recovery_started_at = float(self.data.time)
                self.rise_time = None
            self.recovering = True
        contacting = set()
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.floor_geom in pair:
                contacting.update(pair & self.foot_geoms)
        va = int(self.model.joint("trunk_base_freejoint").dofadr[0])
        stable = (height >= .105 and tilt <= math.radians(20)
                  and np.linalg.norm(self.data.qvel[va:va+3]) < .15
                  and np.linalg.norm(self.policy.get_base_ang_vel()) < 1.5
                  and contacting == self.foot_geoms)
        self.hold = self.hold + CONTROL_DT if stable else 0.
        if self.hold + 1e-9 >= STABLE_HOLD:
            self.recovering = False
            if self.rise_time is None and self.recovery_started_at is not None:
                self.rise_time = float(self.data.time) - self.recovery_started_at
        self.active = ("recovery" if self.recovering else "walking") if self.mode == "auto" else self.mode
        session = self.sessions[self.active]
        self.policy.current_policy = "standing" if self.active == "recovery" else "walking"
        self.policy.ort_session = session
        self.policy.input_name = session.get_inputs()[0].name
        self.policy.output_name = session.get_outputs()[0].name
        self.policy.vel_cmd[:] = twist
        self.policy._update_command()
        if not np.isfinite(self.policy.get_observations()).all():
            raise FloatingPointError("Non-finite observation; simulation paused")
        action = self.policy.infer()
        if action.shape != (14,) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid policy action; simulation paused")
        self.policy.apply_action(action)  # No filtering or action clipping.
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            warnings = self.data.warning.number.copy()
            self.motor.update()
            mujoco.mj_step(self.model, self.data)
            if (self.data.warning.number > warnings).any():
                raise FloatingPointError("MuJoCo reported a physics warning; reset to retry")
            if not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
                raise FloatingPointError("Non-finite physics state; simulation paused")

    def push(self):
        # A single horizontal yaw-frame impulse, including when the body is
        # tilted: no vertical boost and no continuous hidden assistance.
        w, x, y, z = self.data.qpos[self.policy._trunk_qpos_adr+3:self.policy._trunk_qpos_adr+7]
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        joint = self.model.joint("trunk_base_freejoint")
        va = int(joint.dofadr[0])
        self.data.qvel[va:va+3] += np.array([-math.sin(yaw), math.cos(yaw), 0.]) * .2
        mujoco.mj_forward(self.model, self.data)

    def status(self):
        height, tilt = self.posture()
        # Free-joint translation is world-frame; display velocity in the
        # yaw-only robot frame, so roll/pitch do not turn vertical motion into vx.
        q = self.data.qpos[self.policy._trunk_qpos_adr+3:self.policy._trunk_qpos_adr+7]
        w, x, y, z = q
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        va = int(self.model.joint("trunk_base_freejoint").dofadr[0])
        vx, vy = self.data.qvel[va:va+2]
        velocity = [math.cos(yaw)*vx+math.sin(yaw)*vy, -math.sin(yaw)*vx+math.cos(yaw)*vy]
        return {"time": float(self.data.time), "active": self.active, "mode": self.mode,
                "paused": self.paused, "height": height, "tilt": math.degrees(tilt),
                "velocity": velocity, "command": self.policy.command[:3].tolist(),
                "rise_time": self.rise_time, "hold": self.hold, "models": self.labels,
                "spawn": self.spawn, "error": self.error,
                "position": self.data.qpos[self.policy._trunk_qpos_adr:self.policy._trunk_qpos_adr+3].tolist(),
                "quaternion": q.tolist(),
                "joints": dict(zip(self.joint_names, self.data.qpos[self.policy.joint_qpos_indices].tolist()))}


def handler_for(mailbox, library=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *args):
            pass

        def reply(self, status, body, kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/api/frame":
                self.reply(200, mailbox.snapshot())
            elif route == '/api/checkpoints' and library is not None:
                self.reply(200, json.dumps(library.refresh()).encode())
            else:
                base = (REPO / "app/dist").resolve()
                path = (base / ("index.html" if route == "/" else route.lstrip("/"))).resolve()
                if not path.is_relative_to(base) or not path.is_file():
                    self.reply(404, b'{"error":"Build the web app first: cd app && npm ci && npm run build"}')
                    return
                self.reply(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_POST(self):
            if self.path != "/api/control":
                self.reply(404, b'{}')
                return
            # Local page only. Custom JSON requests cannot be sent by a foreign
            # page without a preflight; this server does not enable CORS.
            origin = self.headers.get("Origin")
            if (origin and origin != "http://" + self.headers.get("Host", "")) or self.headers.get("Content-Type") != "application/json":
                self.reply(403, b'{}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                message = json.loads(self.rfile.read(length))
                if not isinstance(message, dict):
                    raise ValueError("Expected object")
                accepted = mailbox.submit(message, time.monotonic())
                self.reply(200 if accepted else 409, b'{}')
            except (ValueError, TypeError, OverflowError) as exc:
                self.reply(400, json.dumps({"error": str(exc)}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking", type=Path, help="Override the manifest's walking policy")
    parser.add_argument("--recovery", type=Path, help="Override the manifest's recovery policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seconds", type=float, default=0, help="Optional bounded smoke duration")
    args = parser.parse_args()
    if args.walking is None or args.recovery is None:
        walking, recovery = default_models()
        args.walking = args.walking or walking
        args.recovery = args.recovery or recovery
    for path in (args.walking, args.recovery):
        if not path.is_file():
            parser.error(f"Missing {path}; export a checkpoint with scripts/export.py, then pass --walking / --recovery")
    demo = Demo(args.walking, args.recovery)
    library = CheckpointLibrary(REPO)
    for slot, path in zip(('walking','recovery'), (args.walking,args.recovery)):
        if path.resolve() != (REPO / 'models' / library.manifest['policies'][slot]['file']).resolve():
            library.current[slot] = ''
    mailbox = Mailbox()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(mailbox, library))
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    print(f"SC0090 keyboard demo: http://{args.host}:{server.server_port}/", flush=True)
    start = time.monotonic()
    next_step, next_frame = start, start
    rendered, last_status = 0, demo.status()
    def apply_loaded(result):
        if result is not None:
            entry, path = result
            demo.load_policy(entry['slot'], path)
            library.loaded(entry)
    try:
        while not args.seconds or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            controls, events = mailbox.consume(now)
            for event in events:
                if isinstance(event, dict):
                    try:
                        apply_loaded(library.begin(event['load']['slot'], event['load']['id']))
                    except Exception as exc:
                        library.failed(exc)
                elif event in POSES:
                    demo.reset(event)
                    controls["twist"] = [0., 0., 0.]
                elif event == "pause":
                    demo.paused = not demo.paused if demo.error is None else True
                elif event == "push" and demo.error is None:
                    demo.push()
            try:
                apply_loaded(library.poll())
            except Exception as exc:
                library.failed(exc)
            demo.mode = controls.get("mode", demo.mode)
            if now >= next_step:
                if not demo.paused:
                    try:
                        demo.step(controls.get("twist", [0., 0., 0.]))
                        last_status = demo.status()
                    except FloatingPointError as exc:
                        demo.paused, demo.error = True, str(exc)
                        print(f"ERROR: {exc}", flush=True)
                # Slower hosts slow wall-clock playback, never enlarge dt or
                # skip physics/policy steps to catch up to rendering.
                next_step = max(next_step + CONTROL_DT, now)
            if now >= next_frame:
                if demo.error is None:
                    last_status = demo.status()
                last_status.update(paused=demo.paused, error=demo.error)
                rendered += 1
                mailbox.publish({"frame": rendered, **last_status,
                                 'selected_models': dict(library.current), 'model_load': library.status})
                next_frame = now + 1/25
            time.sleep(max(0., min(next_step, next_frame) - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            library.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
