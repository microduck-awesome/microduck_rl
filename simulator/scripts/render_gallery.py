#!/usr/bin/env python3
"""Record pinned local-demo policies for the static Pages video gallery.

Run from the repository root with MUJOCO_GL=egl. Physics and inference reuse
the local simulator; recording runs separately and never controls its HTTP API.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sc0090_server import Demo, CONTROL_DT, POSES, REPO
from exploration import Exploration
from mjlab_microduck.actuator.sc0090 import SC0090_DYNAMICS_REVISION, SC0090_MODEL_PATH

CASES = [
    ('walk_idle', '站立', 'walking', [0,0,0]),
    ('walk_forward_003', '低速前进 0.03 m/s', 'walking', [.03,0,0]),
    ('walk_forward_008', '低速前进 0.08 m/s', 'walking', [.08,0,0]),
    ('walk_forward_010', '前进 0.10 m/s', 'walking', [.1,0,0]),
    ('walk_forward_020', '前进 0.20 m/s', 'walking', [.2,0,0]),
    ('walk_forward_030', '前进 0.30 m/s', 'walking', [.3,0,0]),
    ('walk_backward', '后退 0.10 m/s', 'walking', [-.1,0,0]),
    ('walk_left', '左侧移 0.10 m/s', 'walking', [0,.1,0]),
    ('walk_right', '右侧移 0.10 m/s', 'walking', [0,-.1,0]),
    ('walk_turn_left', '左转 0.60 rad/s', 'walking', [0,0,.6]),
    ('walk_turn_right', '右转 0.60 rad/s', 'walking', [0,0,-.6]),
    ('walk_curve', '行走并转弯', 'walking', [.1,0,.3]),
    ('walk_start', '启动：静止 → 0.08 m/s', 'walking', [.08,0,0]),
    ('walk_stop', '停止：0.10 m/s → 静止', 'walking', [0,0,0]),
    *[(f'recovery_{pose}', title+'起身', 'recovery', pose) for pose,title in
      [('sitting','坐姿'),('prone','俯卧'),('supine','仰卧'),('left_side','左侧卧'),('right_side','右侧卧')]],
    ('exploration', '自由探索 · 行走、倒地与起身', 'exploration', None),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=[case[0] for case in CASES])
    parser.add_argument('--output-dir',type=Path,default=REPO.parent/'docs/videos/latest')
    parser.add_argument('--models-dir',type=Path,default=REPO/'models')
    args=parser.parse_args()
    output=args.output_dir; output.mkdir(exist_ok=True,parents=True)
    models_dir=args.models_dir.resolve(strict=True)
    manifest=json.loads((models_dir/'manifest.json').read_text())
    if manifest['schema'] != 1 or manifest['motor_model_sha256'] != hashlib.sha256(SC0090_MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError('Unsupported manifest or mismatched SC0090 motor model')
    paths=[]; labels={}
    for kind in ('walking','recovery'):
        entry=manifest['policies'][kind]
        path=(models_dir/entry['file']).resolve(strict=True)
        if not path.is_relative_to(models_dir) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Invalid {kind} policy path or checksum')
        paths.append(path)
        labels[kind]=Path(entry['checkpoint']).stem.removeprefix('model_')
    demo=Demo(*paths)
    demo.model.vis.global_.offwidth=640; demo.model.vis.global_.offheight=360
    demo.model.stat.extent=1.5
    demo.model.light_castshadow[:]=False
    renderer=mujoco.Renderer(demo.model,height=360,width=640)
    render_data=mujoco.MjData(demo.model)
    camera=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(demo.model,camera)
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=.65; camera.azimuth=135; camera.elevation=-18
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,21); small=ImageFont.truetype(font_path,15)
    results=[]
    try:
        for key,title,kind,command in CASES:
            if args.only and key!=args.only: continue
            duration=60 if kind=='exploration' else 4 if kind=='recovery' else 8
            demo.reset(command if kind=='recovery' else 'standing')
            demo.mode='auto'
            planner=Exploration()
            if kind=='exploration':
                planner.sync(dict(id='gallery',seed=1,duration=4,recovery_timeout=12))
            path=output/f'{key}.mp4'
            writer=imageio_ffmpeg.write_frames(str(path),(640,480),fps=25,codec='libx264',
                pix_fmt_out='yuv420p',quality=None,output_params=['-crf','20','-preset','fast','-movflags','+faststart','-threads','2'])
            writer.send(None)
            frames=0; samples=[]; events=[]; actions=[]; failure=None
            try:
                for step in range(round(duration/CONTROL_DT)):
                    twist=command if kind=='walking' else [0,0,0]
                    if key=='walk_start' and step*CONTROL_DT<2:
                        twist=[0,0,0]
                    elif key=='walk_stop' and step*CONTROL_DT<2:
                        twist=[.1,0,0]
                    if kind=='exploration' and planner.active:
                        twist,event=planner.step(demo.status(),CONTROL_DT,.1,.6)
                        if planner.count>len(actions):
                            actions.append(dict(number=planner.count,action=planner.action,label=planner.label,time=step*CONTROL_DT))
                        if event in POSES:
                            demo.reset(event)
                            events.append(dict(time=step*CONTROL_DT,pose=event,reason='explicit exploration pose reset'))
                        elif event=='push':
                            demo.push()
                            events.append(dict(time=step*CONTROL_DT,event='horizontal velocity impulse 0.2 m/s'))
                        if planner.failed:
                            failure=planner.label; demo.paused=True
                    state=demo.status()
                    if step%2==0:
                        camera.lookat[:]=[state['position'][0],state['position'][1],.12]
                        # Rendering gets a separate data object and recomputes
                        # transforms there; it cannot change live physics caches.
                        mujoco.mj_copyData(render_data,demo.model,demo.data)
                        mujoco.mj_forward(demo.model,render_data)
                        renderer.update_scene(render_data,camera=camera)
                        frame=Image.new('RGB',(640,480),(18,24,33))
                        frame.paste(Image.fromarray(renderer.render()),(0,64))
                        draw=ImageDraw.Draw(frame)
                        draw.text((14,4),title,font=font,fill='#f5f8fc')
                        checkpoint=labels[kind] if kind in labels else f'{labels["walking"]} / {labels["recovery"]}'
                        draw.text((14,36),f'12 V / 80 rpm · checkpoint {checkpoint} · 物理修订 {SC0090_DYNAMICS_REVISION} · 1×',font=small,fill='#aac1d5')
                        if kind=='exploration':
                            detail=f'#{planner.count} {planner.label}'
                        elif kind=='recovery':
                            detail=f'已站稳：{state["rise_time"]:.2f} s（含 0.5 s 确认）' if state['rise_time'] is not None else '模型正在尝试起身 · 尚未达到站稳标准'
                        else:
                            recent=samples[-50:]
                            velocity=np.mean([s['velocity'] for s in recent],axis=0) if recent else state['velocity']
                            detail=f'实测前后 {velocity[0]:+.3f} / 侧向 {velocity[1]:+.3f} m/s · {"起身中" if state["recovering"] else "行走策略"}'
                            if key in ('walk_start','walk_stop'):
                                detail=f'指令 {twist[0]:.2f} m/s · 实测前后 {velocity[0]:+.3f} / 侧向 {velocity[1]:+.3f} m/s'
                        draw.text((14,429),failure or detail,font=small,fill='#f4c076' if failure else '#c3deee')
                        draw.text((14,454),f't={step*CONTROL_DT:5.2f} s · 躯干高 {state["height"]*100:.1f} cm · 倾角 {state["tilt"]:.1f}° · 固定标称参数',font=small,fill='#a6b7ca')
                        writer.send(np.asarray(frame)); frames+=1
                        if step==0: frame.save(output/f'{key}.jpg',quality=90)
                    if not demo.paused:
                        demo.step(twist)
                    state=demo.status()
                    state['yaw_rate']=float(demo.policy.get_base_ang_vel()[2])
                    samples.append(state)
            finally:
                writer.close()
            row=dict(id=key,title=title,kind=kind,file=path.name,poster=f'{key}.jpg',
                     duration_s=frames/25,frames=frames,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     samples=1,pose_resets=events,failed=failure,final_height_m=state['height'],
                     final_tilt_deg=state['tilt'])
            if kind=='walking':
                window=[3,4] if key in ('walk_start','walk_stop') else [2,8]
                measured=samples[round(window[0]/CONTROL_DT):round(window[1]/CONTROL_DT)]
                row.update(command=command,mean_velocity_m_s=np.mean([s['velocity'] for s in measured],axis=0).tolist(),
                           mean_yaw_rate_rad_s=float(np.mean([s['yaw_rate'] for s in measured])),
                           measurement_window_s=window,fell=any(s['recovering'] for s in samples))
                if key in ('walk_start','walk_stop'):
                    initial=[0,0,0] if key=='walk_start' else [.1,0,0]
                    row['command_schedule']=[dict(time_s=0,command=initial),dict(time_s=2,command=command)]
            elif kind=='recovery':
                row.update(pose=command,rise_time_s=state['rise_time'],stable_at_end=state['stable'])
            else:
                row.update(sequence_seed=1,action_duration_s=4,recovery_timeout_s=12,actions=actions)
            results.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        renderer.close()
    result=dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dynamics_revision=SC0090_DYNAMICS_REVISION,joint_velocity_obs_lag_steps=demo.policy.joint_velocity_obs_lag,
                models=manifest,physics='CPU MuJoCo + SC0090 BAM, same as local simulator; nominal parameters; no domain randomization',
                playback_fps=25,policy_hz=50,walk_measurement_window_s=[2,8],videos=results)
    (output/('preview.json' if args.only else 'gallery.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
