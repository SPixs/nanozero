#!/usr/bin/env python
"""Pre-flight environment check for W3090 before running prod training.

Verifies (in order):
  1. Python version >= 3.10
  2. JDK 25 available (java --version)
  3. fastchess binary in PATH
  4. UCI JAR built and accessible
  5. CUDA available for PyTorch (GPU acceleration — non-critical, CPU fallback)
  6. Training paths (run_root) writable

Usage on W3090 (from nanozero-training/ root):
  poetry run python scripts/validate_w3090_env.py --config configs/w3090-run.yaml

Exit code 0 if all critical checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    critical: bool = True


def check_python_version() -> CheckResult:
    v = sys.version_info
    return CheckResult(
        name="Python >= 3.10",
        ok=v >= (3, 10),
        message=f"Python {v.major}.{v.minor}.{v.micro}",
        critical=True,
    )


def check_java_version() -> CheckResult:
    try:
        result = subprocess.run(
            ["java", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            name="JDK 25+",
            ok=False,
            message="java not found in PATH or timeout",
            critical=True,
        )

    first_line = (result.stdout or result.stderr).strip().split("\n")[0]
    major = -1
    for tok in first_line.split():
        if tok.replace(".", "").isdigit():
            major = int(tok.split(".")[0])
            break
    return CheckResult(
        name="JDK 25+",
        ok=major >= 25,
        message=f"java: {first_line} (major={major})",
        critical=True,
    )


def check_fastchess() -> CheckResult:
    path = shutil.which("fastchess")
    if path:
        return CheckResult(
            name="fastchess in PATH",
            ok=True,
            message=f"found at {path}",
            critical=True,
        )
    return CheckResult(
        name="fastchess in PATH",
        ok=False,
        message="not found (install per scripts/install-fastchess-w3090.md)",
        critical=True,
    )


def check_uci_jar(jar_path: Path) -> CheckResult:
    if jar_path.exists():
        return CheckResult(
            name="UCI JAR built",
            ok=True,
            message=f"found at {jar_path}",
            critical=True,
        )
    return CheckResult(
        name="UCI JAR built",
        ok=False,
        message=(
            f"not found at {jar_path} " "(build via 'mvn -pl nanozero-uci -am package -DskipTests')"
        ),
        critical=True,
    )


def check_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="CUDA available",
            ok=False,
            message="torch not importable",
            critical=True,
        )

    if not torch.cuda.is_available():
        return CheckResult(
            name="CUDA available",
            ok=False,
            message="torch.cuda.is_available()=False (training falls back to CPU, very slow)",
            critical=False,
        )

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
    return CheckResult(
        name="CUDA available",
        ok=True,
        message=f"{device_count} GPU(s): {device_name}",
        critical=False,
    )


def check_run_root_writable(run_root: Path) -> CheckResult:
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        test_file = run_root / ".write-test"
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        return CheckResult(
            name="run_root writable",
            ok=False,
            message=f"{run_root}: {e}",
            critical=True,
        )
    return CheckResult(
        name="run_root writable",
        ok=True,
        message=f"{run_root} OK",
        critical=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight env check for W3090 training",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to config YAML (used to extract paths)",
    )
    args = parser.parse_args()

    try:
        from nanozero_training.config.loader import load_config
    except ImportError:
        print("ERROR: nanozero_training package not importable. Activate poetry env first.")
        return 1

    cfg = load_config(args.config)
    run_root = Path(cfg.paths.run_root)
    uci_jar_str = cfg.paths.uci_jar
    uci_jar = Path(uci_jar_str) if Path(uci_jar_str).is_absolute() else run_root / uci_jar_str

    print("=== W3090 Environment Validation ===")
    print(f"Config:    {args.config}")
    print(f"Run root:  {run_root}")
    print(f"UCI JAR:   {uci_jar}")
    print()

    checks = [
        check_python_version(),
        check_java_version(),
        check_fastchess(),
        check_uci_jar(uci_jar),
        check_cuda(),
        check_run_root_writable(run_root),
    ]

    all_critical_ok = True
    for c in checks:
        icon = "OK  " if c.ok else ("FAIL" if c.critical else "WARN")
        critical_tag = " (critical)" if c.critical and not c.ok else ""
        print(f"  [{icon}] {c.name}{critical_tag}: {c.message}")
        if not c.ok and c.critical:
            all_critical_ok = False

    print()
    if all_critical_ok:
        print("All critical checks passed. Ready to run.")
        return 0
    print("Critical checks failed. Fix above issues before running training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
