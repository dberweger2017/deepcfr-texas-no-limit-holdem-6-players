"""Safety checks for staging the fixed v0.4 inference export."""

import hashlib

import pytest

from src.policies import v040 as verify_v04_model


def test_verify_requires_exact_bytes_and_hash(tmp_path, monkeypatch):
    model = tmp_path / "model.json.gz"
    content = b"known compressed bytes"
    model.write_bytes(content)
    monkeypatch.setattr(verify_v04_model, "EXPECTED_BYTES", len(content))
    monkeypatch.setattr(verify_v04_model, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest())
    assert verify_v04_model.verify(model) == hashlib.sha256(content).hexdigest()

    model.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="byte count"):
        verify_v04_model.verify(model)

    model.write_bytes(content[:-1] + b"X")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_v04_model.verify(model)

    model.unlink()
    model.symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="regular file"):
        verify_v04_model.verify(model)
