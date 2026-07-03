"""Integration tests for scripts/generate_gen0_model.py.

Slow opt-in — invoque python subprocess + initialise tout le NanoZeroResNet
+ export .npz. Skip via pytest default (sans -m slow).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_gen0_model.py"


@pytest.mark.slow()
def test_generate_gen0_produces_npz(tmp_path: Path) -> None:
    target = tmp_path / "test-gen0.npz"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seed",
            "42",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert target.exists(), f"Output .npz not produced. stderr={result.stderr}"
    data = np.load(target, allow_pickle=True)
    assert "_meta_export_date" in data.files
    assert "_meta_seed" in data.files
    assert int(data["_meta_seed"].item()) == 42
    assert int(data["_meta_training_step"].item()) == 0
    assert int(data["_meta_generation"].item()) == 1
    assert str(data["_meta_init_method"].item()) == "kaiming_standard"


@pytest.mark.slow()
def test_generate_gen0_reproducible(tmp_path: Path) -> None:
    """Deux invocations même seed -> tensors identiques (metadata export_date OK différents)."""
    target_a = tmp_path / "gen-a.npz"
    target_b = tmp_path / "gen-b.npz"
    for target in (target_a, target_b):
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--seed",
                "42",
                "--output",
                str(target),
            ],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
    data_a = np.load(target_a, allow_pickle=True)
    data_b = np.load(target_b, allow_pickle=True)
    # Tensors should be identical (init reproducible avec même seed)
    for key in data_a.files:
        if key.startswith("_meta_"):
            # Metadata varies (export_date timestamp).
            # _meta_seed, _meta_training_step, etc. doivent matcher logiquement
            # mais le test est cible sur les tensors poids ici.
            continue
        np.testing.assert_array_equal(
            data_a[key],
            data_b[key],
            err_msg=f"Tensor {key} differs between two runs with same seed",
        )
