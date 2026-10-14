"""Fail closed before loading a partial or different sensor experiment."""

import pytest
from mjlab_microduck.sc0090_v2.artifacts import FILES, TrainingArchive


def test_missing_archive_lists_required_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="original P04 research archive") as error:
        TrainingArchive.load(tmp_path)
    assert "residual-library.npz" in str(error.value)
    assert "R1-frozen-v1/model.json" in str(error.value)


def test_corrupt_archive_cannot_be_used_as_nominal_training(tmp_path):
    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TrainingArchive.load(tmp_path)
