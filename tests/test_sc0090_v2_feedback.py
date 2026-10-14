"""Exercise measured data, independent histories and coherent robot-level supply."""

from pathlib import Path
import hashlib
import torch

from mjlab_microduck.sc0090_v2.feedback import WholeRobotFeedback


ROOT = Path(__file__).resolve().parents[1] / "src/mjlab_microduck"
ARCHIVE = ROOT / "custom_v2/training-assets"


def test_frozen_runtime_matches_p04_software_evidence():
    expected = {
        "registers.py": "5dc1b99ee04d38bcf08d792caa9bf4798212c665286858c5447c9d0b83faf2e2",
        "stream.py": "f4d4a48af4d2d03b24aa45233b7660de21925dcb5a1cee8c1f26c42a0cc1c6f6",
        "observer.py": "7b9da38512efda458621b63387bc317ed1b676e5d4a3e8b98a506a95cd5c06c7",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "sc0090" / name).read_bytes()).hexdigest() == digest


def test_shared_supply_partial_reset_and_read_only_query():
    torch.set_num_threads(2)
    feedback = WholeRobotFeedback(ARCHIVE, 4, device="cpu", seed=7, stage="measured")
    pose = torch.linspace(-0.1, 0.1, 60).reshape(4, 15)
    feedback.reset(torch.arange(4), pose)
    assert feedback.ready.all()
    p = feedback.pipeline
    assert torch.equal(p.voltage.reshape(4, 15), feedback.voltage[:, None].expand(-1, 15))
    assert set(feedback.voltage.tolist()) <= {float(torch.tensor(v)) for v in (10.2, 11.1, 12.4)}
    # Advance two real control periods; querying repeatedly cannot advance time,
    # consume donor residuals or change the emitted observation.
    for _ in range(2):
        feedback.command(feedback.counts(pose))
        for _ in range(4):
            feedback.advance(pose, torch.zeros_like(pose))
    before = {
        name: value.clone()
        for name, value in vars(p.stream).items()
        if isinstance(value, torch.Tensor)
    }
    values = feedback.last_q.clone()
    for _ in range(5):
        feedback.refresh()
    assert torch.equal(values, feedback.last_q)
    for name, value in before.items():
        assert torch.equal(value, getattr(p.stream, name)), name
    untouched = torch.cat((torch.arange(15), torch.arange(30, 60)))
    before = {
        name: value[untouched].clone()
        for name, value in vars(p.stream).items()
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 60
    }
    physics = p.physics[untouched].clone()
    clock = feedback.clock.clone()
    feedback.reset(torch.tensor([1]), pose)
    assert torch.equal(feedback.clock[[0, 2, 3]], clock[[0, 2, 3]])
    assert torch.equal(p.physics[untouched], physics)
    for name, value in before.items():
        assert torch.equal(value, getattr(p.stream, name)[untouched]), name
    assert torch.equal(p.voltage.reshape(4, 15), feedback.voltage[:, None].expand(-1, 15))


def test_nominal_stage_keeps_s0_o0_active():
    torch.set_num_threads(2)
    feedback = WholeRobotFeedback(ARCHIVE, 2, device="cpu", stage="nominal")
    pose = torch.zeros(2, 15)
    feedback.reset(torch.arange(2), pose)
    assert torch.equal(feedback.voltage, torch.full((2,), 11.1))
    assert feedback.ready.all()
    assert feedback.pipeline.stream.model.mode == "O0"
    assert feedback.pipeline.event_count.min() >= 20
    assert feedback.last_q.abs().max() > 1e-4  # Real residuals/quantization, not ideal pose.
    target = torch.tensor([[-2.96706, 2.96706, 0.0]])
    count = feedback.counts(target)
    assert count.min() == 20 and count.max() == 1001
    restored = feedback.physical_target(count)
    assert restored[0, 0] > target[0, 0] and restored[0, 1] < target[0, 1]
    assert abs(restored[0, 2]) <= 0.005624596222247351 * 0.958647 / 2
