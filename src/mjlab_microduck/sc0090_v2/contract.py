"""Validate the installed training contract and carry it with exported policies."""

import hashlib
import json
from pathlib import Path

from mjlab_microduck.actuator.sc0090_v5 import MODEL_SHA256
from .artifacts import FILES


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    expected = {
        "schema": "microduck-v2-sc0090-training-v1",
        "timing": {
            "physics_dt_s": 0.005,
            "feedback_dt_s": 0.01,
            "policy_dt_s": 0.02,
            "query_delay_s": 0.02,
            "warmup_s": 0.28,
        },
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"Unsupported V2 {field}: changing frozen timing requires requalification"
            )
    observation = contract["observations"]
    if observation != dict(
        measurement="S0",
        observer="O0",
        neural_observer=False,
        actor_dimension=61,
        action_dimension=14,
        mouth_control="independent",
    ):
        raise ValueError("Unsupported V2 observation contract")
    randomization = contract["randomization"]
    if (
        randomization["nominal_voltages_v"] != [11.1]
        or randomization["measured_voltages_v"] != [10.2, 11.1, 12.4]
        or not randomization["shared_supply"]
        or randomization["motor_parameter_randomization"]
        or randomization["bench_pendulum_loads"]
    ):
        raise ValueError("Unsupported V2 R1 condition contract")
    dynamics = contract["dynamics"]
    if (
        dynamics["sha256"] != MODEL_SHA256
        or dynamics["firmware_p"] != 15
        or dynamics["firmware_d"] != 30
        or dynamics["firmware_i"] != 0
        or not dynamics["goal_profile_zeroed"]
        or dynamics["nominal_voltage_v"] != 11.1
    ):
        raise ValueError("V2 requires the P04 frozen M6 operating conditions")
    commands = contract["commands"]
    if (
        commands["native_count_limits"] != [20, 1001]
        or commands["virtual_zero_count"] != 512
        or commands["action_ema"]
        or commands["target_mapping"] != "inverse_O0_affine_then_round_native_count"
        or commands["radians_per_count"] != 0.005624596222247351
    ):
        raise ValueError("Unsupported V2 native command mapping")
    rate = commands["command_rate_rad_s"]
    if rate is not None and (type(rate) not in (float, int) or not 0 < rate <= 20):
        raise ValueError("Command slew must be null or finite within (0, 20] rad/s")
    if contract["artifacts"]["files"] != FILES:
        raise ValueError(
            "V2 artifact identities differ from the qualified S0/O0/R1 archive"
        )
    return contract


def export_contract(env, contract):
    """Return JSON metadata using the *running* configuration, including overrides.

    61 inputs and 14 outputs alone do not establish deployment compatibility.
    The stock XL330/alpha-home runtime must not silently accept this policy.
    """
    value = json.loads(json.dumps(contract))
    action = env.action_manager.get_term("joint_pos")
    value["commands"]["command_rate_rad_s"] = action.cfg.command_rate_rad_s
    value["commands"]["action_scale"] = action.cfg.scale
    value["randomization"]["stage"] = action.cfg.randomization_stage
    value["hardware_runtime"] = "microduck-v2-sc0090-s0-o0-v1"
    if not action.cfg.spike_rejection:
        value['hardware_runtime']='microduck-v2-bench-no-spike-diagnostic-v1'
        value['observations']['spike_rejection']=False
    if action.cfg.observer_frontend=='firmware-ls3':
        value['hardware_runtime']='microduck-v2-sc0090-firmware-ls3-v1'
        value['observations']['observer']='firmware-native-ls3'
        value['observations']['frontend']='native timestamps, 8-point physical history, 3-point LS'
        value['observations']['firmware_source']='ee77131564eef52a64575628b0c117d69f583039'
        value['observations']['calibration']='virtual affine expressed as per-joint zero/scale; hardware unverified'
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    result = {
        "microduck_training_contract": encoded,
        "microduck_training_contract_sha256": hashlib.sha256(
            encoded.encode()
        ).hexdigest(),
        "microduck_robot": "microduck-v2-sc0090",
        "microduck_runtime": value["hardware_runtime"],
    }
    if action.cfg.observer_frontend=='firmware-ls3':
        result['microduck_deployment']='simulation-runtime-alignment-only'
    if not action.cfg.spike_rejection:
        result['microduck_deployment']='simulation-measurement-ablation-only'
    if action.cfg.o0_local_history:
        value['observations']['o0_history']='five valid sampled points; no O1 220ms refractory period'
        value['observations']['spike_rejection']=action.cfg.spike_rejection
        value['hardware_runtime']='microduck-v2-p04-o0-valid-raw-native-counts-v2'
        value['observations']['native_feedback_count_limits']=[0,1023]
        value['observations']['command_count_limits']=[20,1001]
        encoded=json.dumps(value,sort_keys=True,separators=(',',':'))
        result.update(microduck_training_contract=encoded,
            microduck_training_contract_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            microduck_runtime=value['hardware_runtime'],
            microduck_deployment='simulation-feedback-repair-validation')
    # Training execution provenance is separate from the hardware interface.
    # In particular, a new donor RNG protocol must not inherit the old frozen
    # source identity merely because the coefficient files are unchanged.
    feedback = getattr(env.unwrapped, '_v2_joint_feedback', None)
    if feedback is not None:
        from .measurement.gpu_donors import SAMPLER_VERSION
        root = Path(__file__).resolve().parent
        files = sorted([root/'feedback.py',root/'graph.py',*root.glob('measurement/*.py')])
        if action.cfg.observer_frontend=='firmware-ls3':
            files.append(root.parent/'custom_v2/firmware_ls3.py')
        result['microduck_measurement_execution'] = json.dumps(dict(
            schema=1, donor_sampler=SAMPLER_VERSION, candidate_count=1024,
            pipeline=type(feedback.pipeline).__name__,
            frontend=type(feedback.pipeline.stream).__name__,
            spike_rejection=action.cfg.spike_rejection,
            o0_local_history=action.cfg.o0_local_history,
            native_feedback_count_limits=list(feedback.pipeline.stream.raw_count_limits),
            command_count_limits=list(feedback.pipeline.stream.count_limits),
            source_sha256={str(p.relative_to(root,walk_up=True)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            rollout_graph=feedback.graph_enabled,reset_graph=feedback.reset_graph_enabled,
            reset_graph_chunk=feedback.reset_graph_chunk,
            compact_reset=feedback.compact_reset),sort_keys=True,separators=(',',':'))
    return result
