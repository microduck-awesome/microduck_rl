"""Read-only command quantization audit, not an actuator reachability test."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import numpy as np
import torch
import mjlab.tasks
from mjlab_microduck.sc0090_v2.artifacts import TrainingArchive
from mjlab_microduck.sc0090_v2.feedback import WholeRobotFeedback
from mjlab_microduck.actuator.sc0090_v5 import load_v5_model

root = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
output = root / 'head-command-lattice-audit.json'
assert not output.exists()
run_path = root / 'paired-stdcap-allocation-001/run.json'
run = json.loads(run_path.read_text())
action = run['environment']['actions']['joint_pos']
assert action['scale'] == 1 and action['offset'] == 0 and action['command_rate_rad_s'] is None
archive = TrainingArchive.load(action['archive_path'])
mapping = np.asarray(archive.observer_mapping, dtype=np.float64)
scale = .005624596222247351
model = load_v5_model()
bias, motor_scale = model.feedback_bias.value, model.feedback_scale.value
values = [0, -.005, .005, -.03, .03, -.04, .04, -.11, .11, -.39, .39, -.49, .49,
          -.4857776, 1.1, -1.1, .904656, -1.4, 1.4, -.31, .31]
desired64 = np.asarray(values, dtype=np.float64)
ideal_count = (desired64 - mapping[0]) / mapping[1] / scale + 512
expected_count = np.rint(np.clip(ideal_count, 20, 1001))
results = {}
for device in ['cpu', 'cuda:1']:
    stream = SimpleNamespace(model=SimpleNamespace(mapping=torch.tensor(mapping, dtype=torch.float32, device=device)),
                             scale=scale, zero=512., count_limits=(20, 1001))
    feedback = SimpleNamespace(pipeline=SimpleNamespace(stream=stream))
    desired = torch.tensor(values, device=device, dtype=torch.float32)
    count = WholeRobotFeedback.counts(feedback, desired)
    physical = WholeRobotFeedback.physical_target(feedback, count)
    motor_input = WholeRobotFeedback.physical_goal_for_motor(feedback, count, bias, motor_scale)
    results[device] = [x.detach().cpu().numpy() for x in [count, physical, motor_input]]
    np.testing.assert_array_equal(results[device][0], expected_count)
np.testing.assert_array_equal(results['cpu'][0], results['cuda:1'][0])
max_abs = []
for cpu, gpu in zip(results['cpu'][1:], results['cuda:1'][1:]):
    # FP32 affine/division evaluation can differ by an ULP between devices.
    # Keep discrete counts exact; bound floating results by four FP32 epsilons.
    bound = 4 * np.finfo(np.float32).eps * np.maximum(1, np.abs(cpu))
    assert np.all(np.abs(cpu-gpu) <= bound)
    max_abs.append(float(np.max(np.abs(cpu-gpu))))
count, physical, motor_input = results['cuda:1']
rows = []
for index, value in enumerate(values):
    band = max(.1 * abs(value), .002)
    rows.append(dict(requested_rad=value, native_count=int(count[index]),
                     o0_target_rad=float(physical[index]), motor_api_target_rad=float(motor_input[index]),
                     o0_target_rounding_error_rad=float(physical[index] - value),
                     illustrative_zero_to_target_t90_band_rad=band,
                     rounded_o0_target_within_band=abs(float(physical[index]) - value) <= band))
physical_lsb = float(mapping[1] * scale)
report = dict(status='PASS_CPU_CUDA_PRODUCTION_MAPPING_AND_FLOAT64_COUNT_ORACLE',
    scope='Frozen command map only. No physics stepping, no policy change, no measurements or true actor state substituted.',
    limitation='O0 requested-target lattice is not a physical static-equilibrium error floor. BAM torque, friction, load, estimator noise, dynamics and history affect achieved q. T90 bands here only illustrate steps from zero; official trajectory metrics remain unchanged.',
    o0_mapping=mapping.tolist(), native_register_radians_per_count=scale,
    o0_requested_target_radians_per_count=physical_lsb,
    o0_requested_target_degrees_per_count=math.degrees(physical_lsb),
    motor_api_radians_per_count=float(scale / motor_scale),
    motor_feedback_bias=float(bias), motor_feedback_scale=float(motor_scale),
    cpu_cuda_counts_bitwise_equal=True, cpu_cuda_target_max_abs_difference_rad=max_abs,
    head_cap_001_in_o0_target_counts=.01 / physical_lsb,
    small_003_tolerance_in_o0_target_counts=.003 / physical_lsb,
    head_fine_004_in_o0_target_counts=.04 / physical_lsb,
    rows=rows,
    provenance=dict(run=str(run_path), run_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
                    observer_weights=str(archive.weight_path), observer_weights_sha256=hashlib.sha256(archive.weight_path.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','provenance']}))
print(json.dumps([row for row in rows if abs(row['requested_rad']) in [0, .005, .03, .04]]))
