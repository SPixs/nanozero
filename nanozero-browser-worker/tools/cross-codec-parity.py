#!/usr/bin/env python3
"""cross-codec-parity.py — parité bi-langage (Story A.1).

Vérifie que le décodage binaire serveur (``submit_codec.decode``) produit des bytes denses
IDENTIQUES au chemin JSON base64 (``input_planes_b64`` / ``policy_target_b64`` / ``outcome``)
pour la fixture écrite par ``cross-codec-parity.mjs``.

Usage : node tools/cross-codec-parity.mjs && python3 tools/cross-codec-parity.py
"""

import base64
import json
import sys
from pathlib import Path

# nanozero-browser-worker/tools/ -> repo root -> nanozero-jobserver/src
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "nanozero-jobserver" / "src"))

from nanozero_jobserver.submit_codec import decode  # noqa: E402

_here = Path(__file__).resolve().parent
data = (_here / "parity-fixture.bin").read_bytes()
json_path = json.loads((_here / "parity-fixture.json").read_text())

decoded = decode(data)
ok = len(decoded) == len(json_path)
for (planes_bytes, policy_bytes, outcome), ref in zip(decoded, json_path):
    if planes_bytes != base64.b64decode(ref["input_planes_b64"]):
        ok = False
        print("  ✗ planes != chemin JSON")
    if policy_bytes != base64.b64decode(ref["policy_target_b64"]):
        ok = False
        print("  ✗ policy densifiée != chemin JSON")
    if outcome != ref["outcome"]:
        ok = False
        print("  ✗ outcome != chemin JSON")

print("PARITÉ OK — decode binaire == chemin JSON base64" if ok else "PARITÉ ÉCHEC")
sys.exit(0 if ok else 1)
