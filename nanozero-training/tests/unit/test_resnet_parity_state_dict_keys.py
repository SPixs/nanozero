"""Parity test: training NanoZeroResNet state_dict keys must match nn reference exactly.

Critical for ADR-003 duplication contrôlée — any divergence in naming or order
breaks the bit-perfect parity with parity-model.npz (phase 1.0.0-3 test).

The expected keys list is a frozen snapshot captured from nn Python reference
(nanozero-nn/scripts/python/nanozero_resnet.py:NanoZeroResNet) on 2026-05-13.
"""

from __future__ import annotations

from nanozero_training.network.resnet import NanoZeroResNet

# Snapshot of state_dict keys from nn Python reference (NanoZeroResNet, 8x96).
# Generated via `poetry run python -c "from nanozero_resnet import NanoZeroResNet;
# m = NanoZeroResNet(); [print(k) for k in m.state_dict().keys()]"` on the
# nn reference script. Frozen as the contract for ADR-003 parity.
EXPECTED_KEYS: list[str] = [
    "input_conv.weight",
    "input_conv.bias",
    "input_bn.weight",
    "input_bn.bias",
    "input_bn.running_mean",
    "input_bn.running_var",
    "input_bn.num_batches_tracked",
    *(
        key
        for i in range(8)
        for key in (
            f"blocks.{i}.conv1.weight",
            f"blocks.{i}.conv1.bias",
            f"blocks.{i}.bn1.weight",
            f"blocks.{i}.bn1.bias",
            f"blocks.{i}.bn1.running_mean",
            f"blocks.{i}.bn1.running_var",
            f"blocks.{i}.bn1.num_batches_tracked",
            f"blocks.{i}.conv2.weight",
            f"blocks.{i}.conv2.bias",
            f"blocks.{i}.bn2.weight",
            f"blocks.{i}.bn2.bias",
            f"blocks.{i}.bn2.running_mean",
            f"blocks.{i}.bn2.running_var",
            f"blocks.{i}.bn2.num_batches_tracked",
        )
    ),
    "policy_conv.weight",
    "policy_conv.bias",
    "value_conv.weight",
    "value_conv.bias",
    "value_fc1.weight",
    "value_fc1.bias",
    "value_fc2.weight",
    "value_fc2.bias",
]


def test_resnet_state_dict_keys_match_nn_reference() -> None:
    """state_dict() keys must exactly match nn Python reference (ADR-003)."""
    model = NanoZeroResNet()
    actual_keys = list(model.state_dict().keys())
    if actual_keys != EXPECTED_KEYS:
        first_diff = next(
            (i for i, (a, e) in enumerate(zip(actual_keys, EXPECTED_KEYS, strict=False)) if a != e),
            "N/A",
        )
        raise AssertionError(
            f"state_dict keys mismatch (len ours={len(actual_keys)}, "
            f"len ref={len(EXPECTED_KEYS)}):\n"
            f"  first diff at idx {first_diff}\n"
            f"  ours[:5]: {actual_keys[:5]}\n"
            f"  ref[:5]:  {EXPECTED_KEYS[:5]}"
        )


def test_resnet_state_dict_count_127() -> None:
    """Exactly 127 entries in state_dict (sanity check on snapshot)."""
    model = NanoZeroResNet()
    # 7 (input_conv + input_bn) + 8 blocs * 14 entries + 8 (heads) = 7 + 112 + 8 = 127
    assert len(model.state_dict()) == 127
    assert len(EXPECTED_KEYS) == 127
