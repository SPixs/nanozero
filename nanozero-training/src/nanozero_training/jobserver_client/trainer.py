"""Trainer-side HTTP client + payload decoders for the jobserver.

The trainer's job (Phase 13.5) is the inverse of the worker's :
  - poll /training/should_train regularly
  - when ready, pull a batch via /replay/sample
  - decode base64-float32 BLOBs back into numpy arrays
  - train one epoch, export a new .onnx
  - POST /models/register + /models/{v}/promote

This module covers the HTTP wrapper + the decode side (positions → np.ndarrays).
The actual training loop integration is wired in Phase 13.5b.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt

LOG = logging.getLogger(__name__)

# Shape constants — kept in sync with nanozero-board GameState.NN_PLANES
# and nanozero-nn MoveEncoding.POLICY_INDICES.
INPUT_PLANES_SHAPE = (119, 8, 8)
POLICY_TARGET_SHAPE = (4672,)


@dataclass(frozen=True)
class DecodedPosition:
    """One training sample after decoding from the wire format.

    Attributes:
        input_planes: np.float32, shape (119, 8, 8).
        policy_target: np.float32, shape (4672,).
        value_target: scalar in {-1.0, 0.0, +1.0} (after backfill).
        ply: half-move index within the source game.
        model_version: version of the model that played this position.
    """

    input_planes: npt.NDArray[np.float32]
    policy_target: npt.NDArray[np.float32]
    value_target: float
    ply: int
    model_version: int


@dataclass(frozen=True)
class ShouldTrainStatus:
    """Response of /training/should_train."""

    should_train: bool
    new_positions: int
    threshold: int


def decode_position(payload: dict[str, Any]) -> DecodedPosition:
    """Convert one JSON position dict (with base64 BLOBs) into a typed DecodedPosition.

    Args:
        payload: dict with keys: ply, fen, input_planes_b64, policy_target_b64,
            outcome, model_version, game_id (some optional).

    Returns:
        DecodedPosition with numpy arrays ready for PyTorch.

    Raises:
        ValueError: malformed base64 or wrong array shape.
    """
    try:
        planes_bytes = base64.b64decode(payload["input_planes_b64"])
        policy_bytes = base64.b64decode(payload["policy_target_b64"])
    except (KeyError, ValueError, base64.binascii.Error) as e:  # type: ignore[attr-defined]
        raise ValueError(f"Invalid base64 in position payload: {e}") from e

    # Worker writes little-endian float32 (cf. nanozero-worker Sample.base64FloatArray).
    input_planes = np.frombuffer(planes_bytes, dtype="<f4")
    policy_target = np.frombuffer(policy_bytes, dtype="<f4")

    expected_planes = INPUT_PLANES_SHAPE[0] * INPUT_PLANES_SHAPE[1] * INPUT_PLANES_SHAPE[2]
    if input_planes.size != expected_planes:
        raise ValueError(
            f"Wrong input_planes size: expected {expected_planes} floats, got {input_planes.size}"
        )
    if policy_target.size != POLICY_TARGET_SHAPE[0]:
        raise ValueError(
            f"Wrong policy_target size: expected {POLICY_TARGET_SHAPE[0]} floats, "
            f"got {policy_target.size}"
        )

    return DecodedPosition(
        input_planes=input_planes.reshape(INPUT_PLANES_SHAPE),
        policy_target=policy_target,
        value_target=float(payload.get("outcome", 0.0)),
        ply=int(payload.get("ply", 0)),
        model_version=int(payload.get("model_version", 0)),
    )


class TrainerClient:
    """HTTP wrapper for the trainer's interactions with the jobserver.

    Auth pattern : X-API-Key header on every request (jobserver enforces).
    Returns parsed Python objects (dicts/dataclasses), never raw httpx responses.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
    ):
        self._owns_client = http_client is None
        if http_client is None:
            http_client = httpx.Client(
                base_url=server_url.rstrip("/"),
                headers={"X-API-Key": api_key, "X-Worker-Id": "trainer"},
                timeout=timeout_seconds,
            )
        self.http = http_client

    def close(self) -> None:
        if self._owns_client:
            self.http.close()

    def __enter__(self) -> TrainerClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- training trigger

    def should_train(self, threshold: int, since_version: int | None = None) -> ShouldTrainStatus:
        """GET /training/should_train. Returns parsed status."""
        params: dict[str, Any] = {"threshold": threshold}
        if since_version is not None:
            params["since_version"] = since_version
        resp = self.http.get("/training/should_train", params=params)
        resp.raise_for_status()
        body = resp.json()
        return ShouldTrainStatus(
            should_train=bool(body["should_train"]),
            new_positions=int(body["new_positions"]),
            threshold=int(body["threshold"]),
        )

    # ---------------------------------------------------------------- replay sample

    def fetch_sample(
        self,
        n: int,
        window: int = 5,
        current_version: int | None = None,
    ) -> list[DecodedPosition]:
        """GET /replay/sample?n=&window=&current_version=, returns decoded positions.

        Returns up to N positions (may be fewer if the buffer is small). Each
        BLOB is decoded eagerly — the caller gets typed np arrays.
        """
        params: dict[str, Any] = {"n": n, "window": window}
        if current_version is not None:
            params["current_version"] = current_version
        resp = self.http.get("/replay/sample", params=params)
        resp.raise_for_status()
        body = resp.json()
        positions = body.get("positions", [])
        return [decode_position(p) for p in positions]

    # ---------------------------------------------------------------- model registry

    def current_model(self) -> dict[str, Any] | None:
        """GET /models/current. Returns metadata dict, or None if 404 (no promoted)."""
        resp = self.http.get("/models/current")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def register_model(
        self,
        version: int,
        name: str,
        onnx_path: str,
        sha256_onnx: str,
        parent_version: int | None = None,
    ) -> dict[str, Any]:
        """POST /models/register."""
        body: dict[str, Any] = {
            "version": version,
            "name": name,
            "onnx_path": onnx_path,
            "sha256_onnx": sha256_onnx,
        }
        if parent_version is not None:
            body["parent_version"] = parent_version
        resp = self.http.post("/models/register", json=body)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def promote_model(self, version: int) -> dict[str, Any]:
        """POST /models/{version}/promote."""
        resp = self.http.post(f"/models/{version}/promote")
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def enqueue_jobs(
        self,
        count: int,
        model_version: int,
        num_sims: int = 200,
        opening_fen: str | None = None,
        dirichlet_seed_base: int | None = None,
    ) -> dict[str, Any]:
        """POST /jobs/enqueue. Bulk-enqueue {count} pending jobs.

        Used by the trainer after each promote (sustain the worker queue) and
        by the initial-setup script to seed the run.
        """
        body: dict[str, Any] = {
            "count": count,
            "model_version": model_version,
            "num_sims": num_sims,
        }
        if opening_fen is not None:
            body["opening_fen"] = opening_fen
        if dirichlet_seed_base is not None:
            body["dirichlet_seed_base"] = dirichlet_seed_base
        resp = self.http.post("/jobs/enqueue", json=body)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        """GET /stats. Used by orchestrators / dashboards."""
        resp = self.http.get("/stats")
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def upload_model_file(self, version: int, onnx_path: Path) -> None:
        """Phase 13.5b placeholder : upload the .onnx blob to a server-controlled path.

        Currently the jobserver does NOT expose a multipart upload endpoint — models
        are referenced by filesystem path (the server reads them from disk for the
        /models/{v}/download). On a shared filesystem (W3090 S:\\ via Tailscale)
        this works. On distributed setups we'll need to add POST /models/{v}/upload.
        """
        raise NotImplementedError(
            "upload_model_file requires jobserver multipart endpoint (Phase 13.6+). "
            "For now, ensure onnx_path is reachable by the jobserver via shared filesystem."
        )
