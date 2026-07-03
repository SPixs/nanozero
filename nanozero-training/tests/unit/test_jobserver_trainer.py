"""Unit tests for jobserver_client.trainer (Phase 13.5)."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest
from nanozero_training.jobserver_client.trainer import (
    INPUT_PLANES_SHAPE,
    POLICY_TARGET_SHAPE,
    DecodedPosition,
    ShouldTrainStatus,
    TrainerClient,
    decode_position,
)

# -----------------------------------------------------------------------------
# decode_position helper
# -----------------------------------------------------------------------------


def _b64_floats(floats: np.ndarray) -> str:
    """Encode np.float32 array as base64-of-little-endian-bytes (worker format)."""
    return base64.b64encode(floats.astype("<f4").tobytes()).decode("ascii")


def _make_position_payload(
    planes: np.ndarray | None = None,
    policy: np.ndarray | None = None,
    outcome: float = 0.0,
    ply: int = 0,
    model_version: int = 1,
) -> dict:
    if planes is None:
        planes = np.zeros(INPUT_PLANES_SHAPE, dtype=np.float32)
    if policy is None:
        policy = np.zeros(POLICY_TARGET_SHAPE, dtype=np.float32)
    return {
        "ply": ply,
        "fen": "",
        "input_planes_b64": _b64_floats(planes.flatten()),
        "policy_target_b64": _b64_floats(policy),
        "outcome": outcome,
        "model_version": model_version,
        "game_id": "test-game",
    }


def test_decode_position_returns_correct_shapes():
    payload = _make_position_payload()
    pos = decode_position(payload)

    assert isinstance(pos, DecodedPosition)
    assert pos.input_planes.shape == INPUT_PLANES_SHAPE
    assert pos.input_planes.dtype == np.float32
    assert pos.policy_target.shape == POLICY_TARGET_SHAPE
    assert pos.policy_target.dtype == np.float32


def test_decode_position_preserves_blob_values():
    """Roundtrip: build a known float array, encode, decode, compare elementwise."""
    planes = np.random.default_rng(42).uniform(-1, 1, INPUT_PLANES_SHAPE).astype(np.float32)
    policy = np.random.default_rng(43).uniform(0, 1, POLICY_TARGET_SHAPE).astype(np.float32)
    payload = _make_position_payload(planes=planes, policy=policy, outcome=0.7, ply=12)

    pos = decode_position(payload)

    np.testing.assert_array_equal(pos.input_planes, planes)
    np.testing.assert_array_equal(pos.policy_target, policy)
    assert pos.value_target == 0.7
    assert pos.ply == 12


def test_decode_position_handles_negative_outcome():
    payload = _make_position_payload(outcome=-1.0)
    pos = decode_position(payload)
    assert pos.value_target == -1.0


def test_decode_position_missing_optional_fields_uses_defaults():
    """ply, model_version, outcome are optional with default 0."""
    payload = _make_position_payload()
    del payload["ply"]
    del payload["model_version"]
    del payload["outcome"]
    pos = decode_position(payload)
    assert pos.ply == 0
    assert pos.model_version == 0
    assert pos.value_target == 0.0


def test_decode_position_invalid_base64_raises():
    payload = _make_position_payload()
    payload["input_planes_b64"] = "not-valid-base64!@#"
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_position(payload)


def test_decode_position_wrong_planes_size_raises():
    bad_planes = np.zeros(100, dtype=np.float32)  # not 119*8*8
    payload = _make_position_payload()
    payload["input_planes_b64"] = _b64_floats(bad_planes)
    with pytest.raises(ValueError, match="input_planes size"):
        decode_position(payload)


def test_decode_position_wrong_policy_size_raises():
    bad_policy = np.zeros(100, dtype=np.float32)  # not 4672
    payload = _make_position_payload()
    payload["policy_target_b64"] = _b64_floats(bad_policy)
    with pytest.raises(ValueError, match="policy_target size"):
        decode_position(payload)


def test_decode_position_missing_blob_field_raises():
    payload = _make_position_payload()
    del payload["input_planes_b64"]
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_position(payload)


# -----------------------------------------------------------------------------
# TrainerClient — HTTP wrapper
# -----------------------------------------------------------------------------


def _make_http_mock(json_body: dict, status_code: int = 200) -> MagicMock:
    """Create a MagicMock httpx-like Client that returns the given JSON."""
    http = MagicMock()
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    http.get.return_value = resp
    http.post.return_value = resp
    return http


def test_should_train_parses_response():
    http = _make_http_mock({"should_train": True, "new_positions": 30000, "threshold": 25000})
    client = TrainerClient("http://x", "k", http_client=http)
    status = client.should_train(threshold=25000)

    assert isinstance(status, ShouldTrainStatus)
    assert status.should_train is True
    assert status.new_positions == 30000
    assert status.threshold == 25000


def test_should_train_passes_query_params():
    http = _make_http_mock({"should_train": False, "new_positions": 0, "threshold": 100})
    client = TrainerClient("http://x", "k", http_client=http)
    client.should_train(threshold=100, since_version=5)

    http.get.assert_called_with(
        "/training/should_train",
        params={"threshold": 100, "since_version": 5},
    )


def test_fetch_sample_decodes_positions():
    """Server returns 2 positions ; client must decode both into DecodedPosition."""
    p1 = _make_position_payload(outcome=0.5, ply=10)
    p2 = _make_position_payload(outcome=-0.5, ply=20)
    http = _make_http_mock({"positions": [p1, p2], "requested": 2, "returned": 2})
    client = TrainerClient("http://x", "k", http_client=http)

    positions = client.fetch_sample(n=2)

    assert len(positions) == 2
    assert positions[0].ply == 10
    assert positions[0].value_target == 0.5
    assert positions[1].ply == 20
    assert positions[1].value_target == -0.5


def test_fetch_sample_passes_query_params():
    http = _make_http_mock({"positions": [], "requested": 100, "returned": 0})
    client = TrainerClient("http://x", "k", http_client=http)
    client.fetch_sample(n=100, window=10, current_version=7)
    http.get.assert_called_with(
        "/replay/sample", params={"n": 100, "window": 10, "current_version": 7}
    )


def test_fetch_sample_empty_returns_empty_list():
    http = _make_http_mock({"positions": [], "requested": 100, "returned": 0})
    client = TrainerClient("http://x", "k", http_client=http)
    assert client.fetch_sample(n=100) == []


def test_current_model_returns_none_on_404():
    http = MagicMock()
    resp = MagicMock(status_code=404)
    resp.raise_for_status = MagicMock()
    http.get.return_value = resp
    client = TrainerClient("http://x", "k", http_client=http)
    assert client.current_model() is None


def test_current_model_returns_dict_on_200():
    http = _make_http_mock(
        {
            "version": 5,
            "name": "gen-005-trained",
            "sha256_onnx": "abc",
            "promoted_at": "2026-05-16Z",
            "parent_version": 4,
            "created_at": "2026-05-16Z",
        }
    )
    client = TrainerClient("http://x", "k", http_client=http)
    m = client.current_model()
    assert m is not None
    assert m["version"] == 5


def test_register_model_posts_full_body():
    http = _make_http_mock(
        {
            "version": 7,
            "name": "gen-007-trained",
            "sha256_onnx": "abc",
            "promoted_at": None,
            "parent_version": 6,
            "created_at": "2026-05-16Z",
        }
    )
    client = TrainerClient("http://x", "k", http_client=http)
    client.register_model(
        version=7,
        name="gen-007-trained",
        onnx_path="/m/v7.onnx",
        sha256_onnx="abc",
        parent_version=6,
    )
    http.post.assert_called_with(
        "/models/register",
        json={
            "version": 7,
            "name": "gen-007-trained",
            "onnx_path": "/m/v7.onnx",
            "sha256_onnx": "abc",
            "parent_version": 6,
        },
    )


def test_register_model_omits_parent_when_none():
    http = _make_http_mock(
        {
            "version": 1,
            "name": "gen-001-init",
            "sha256_onnx": "a",
            "promoted_at": None,
            "parent_version": None,
            "created_at": "2026-05-16Z",
        }
    )
    client = TrainerClient("http://x", "k", http_client=http)
    client.register_model(version=1, name="gen-001-init", onnx_path="/p", sha256_onnx="a")
    call = http.post.call_args
    assert "parent_version" not in call.kwargs["json"]


def test_promote_model_posts_no_body():
    http = _make_http_mock({"status": "promoted", "version": 5})
    client = TrainerClient("http://x", "k", http_client=http)
    result = client.promote_model(version=5)
    http.post.assert_called_with("/models/5/promote")
    assert result["status"] == "promoted"


def test_upload_model_file_not_implemented():
    """Phase 13.6+ feature — currently raises NotImplementedError."""
    from pathlib import Path

    client = TrainerClient("http://x", "k", http_client=MagicMock())
    with pytest.raises(NotImplementedError, match="multipart"):
        client.upload_model_file(version=1, onnx_path=Path("/tmp/x.onnx"))
