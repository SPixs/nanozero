"""Tests for data/dataset.py — NpzDataset + LazyNpzDataset + load_replay_buffer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from nanozero_training.data.dataset import (
    BatchedNpzDataset,
    LazyNpzDataset,
    NpzDataset,
    _glob_browser_shards,
    _select_browser_shards,
    _shard_n_samples,
    load_replay_buffer,
    load_replay_buffer_batched,
    load_replay_buffer_lazy,
)
from nanozero_training.data.npz_writer import make_batch_filename, write_batch
from nanozero_training.data.sample import (
    BOARD_SIZE,
    N_INPUT_PLANES,
    N_POLICY,
    Sample,
    make_valid_sample_arrays,
)


def _make_sample(ply: int = 0, turn: int = 0, value: float = 0.0) -> Sample:
    args = make_valid_sample_arrays()
    args["ply"] = ply
    args["turn"] = turn
    args["value_target"] = value
    return Sample(**args)


def _write_batch_at(tmp_path: Path, gen: int, batch: int, n_samples: int = 3) -> Path:
    target = tmp_path / make_batch_filename(gen, batch)
    samples = [_make_sample(ply=i) for i in range(n_samples)]
    write_batch(samples=samples, path=target, gen=gen, batch_idx=batch, n_games=1)
    return target


# ----- NpzDataset -----


def test_npzdataset_loads_existing_file(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=5)
    ds = NpzDataset(path)
    assert len(ds) == 5


def test_npzdataset_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        NpzDataset(tmp_path / "missing.npz")


def test_npzdataset_getitem_returns_tensor_dict(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    ds = NpzDataset(path)
    item = ds[0]
    assert set(item.keys()) == {"input_planes", "policy_target", "value_target", "turn", "ply"}
    for v in item.values():
        assert isinstance(v, torch.Tensor)


def test_npzdataset_getitem_index_out_of_range(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    ds = NpzDataset(path)
    with pytest.raises(IndexError, match="out of range"):
        ds[3]


def test_npzdataset_getitem_negative_index(tmp_path: Path) -> None:
    """v1.0.0 doesn't support negative indices (no wrap-around)."""
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    ds = NpzDataset(path)
    with pytest.raises(IndexError, match="out of range"):
        ds[-1]


def test_npzdataset_dtype_conversions(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=1)
    ds = NpzDataset(path)
    item = ds[0]
    assert item["input_planes"].dtype == torch.float32
    assert item["policy_target"].dtype == torch.float32
    assert item["value_target"].dtype == torch.float32
    # turn / ply converted to long for nn.CrossEntropyLoss compatibility downstream.
    assert item["turn"].dtype == torch.int64
    assert item["ply"].dtype == torch.int64


def test_npzdataset_shape_per_sample(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=1)
    ds = NpzDataset(path)
    item = ds[0]
    assert item["input_planes"].shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert item["policy_target"].shape == (N_POLICY,)
    assert item["value_target"].shape == ()
    assert item["turn"].shape == ()
    assert item["ply"].shape == ()


def test_npzdataset_ply_values_preserved(tmp_path: Path) -> None:
    """Roundtrip : ply value per sample should match original."""
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    ds = NpzDataset(path)
    for i in range(4):
        assert int(ds[i]["ply"].item()) == i


# ----- load_replay_buffer -----


def test_load_replay_buffer_combines_files(tmp_path: Path) -> None:
    # Write 3 batches across gen 1, 2, 3 (each 3 samples).
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=3, batch=0, n_samples=3)

    buf = load_replay_buffer(tmp_path, current_gen=3, window=10)
    assert len(buf) == 9


def test_load_replay_buffer_window_filters(tmp_path: Path) -> None:
    """window=5 from current_gen=15 → only gens 11-15 included."""
    for g in range(1, 16):  # gens 1 through 15
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=2)

    buf = load_replay_buffer(tmp_path, current_gen=15, window=5)
    # Gens 11-15 = 5 gens x 2 samples = 10 samples
    assert len(buf) == 10


def test_load_replay_buffer_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No batch files"):
        load_replay_buffer(tmp_path, current_gen=1, window=10)


def test_load_replay_buffer_directory_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Datasets directory"):
        load_replay_buffer(tmp_path / "nonexistent", current_gen=1, window=10)


def test_load_replay_buffer_current_gen_smaller_than_window(tmp_path: Path) -> None:
    """current_gen=3 window=10 → range max(1, 3-10+1)=max(1,-6)=1 → gens [1, 3]."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=2)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=2)
    _write_batch_at(tmp_path, gen=3, batch=0, n_samples=2)

    buf = load_replay_buffer(tmp_path, current_gen=3, window=10)
    assert len(buf) == 6  # gens 1, 2, 3 all included


def test_load_replay_buffer_multiple_batches_per_gen(tmp_path: Path) -> None:
    """Several batches in same gen should all be loaded."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=1, batch=1, n_samples=2)
    _write_batch_at(tmp_path, gen=1, batch=2, n_samples=1)

    buf = load_replay_buffer(tmp_path, current_gen=1, window=1)
    assert len(buf) == 6  # 3 + 2 + 1


# ----- LazyNpzDataset -----


def _collect_samples(ds: LazyNpzDataset) -> list[dict[str, torch.Tensor]]:
    return list(ds)


def test_lazynpzdataset_yields_all_samples(tmp_path: Path) -> None:
    """Total samples iterated should equal sum of shard sizes (no loss, no dup)."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    _write_batch_at(tmp_path, gen=1, batch=1, n_samples=3)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=5)
    paths = sorted(tmp_path.glob("*.npz"))
    ds = LazyNpzDataset(paths, buffer_size=10, shuffle=True, seed=0)
    samples = _collect_samples(ds)
    assert len(samples) == 12
    assert ds.total_samples() == 12


def test_lazynpzdataset_no_shuffle_preserves_order(tmp_path: Path) -> None:
    """shuffle=False : ply values come out in (shard order, ply order) = 0..N."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    paths = sorted(tmp_path.glob("*.npz"))
    ds = LazyNpzDataset(paths, buffer_size=10, shuffle=False, seed=0)
    plies = [int(s["ply"].item()) for s in ds]
    assert plies == [0, 1, 2, 3]


def test_lazynpzdataset_shape_per_sample(tmp_path: Path) -> None:
    """Each sample has the expected tensor shapes and dtypes."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=2)
    paths = sorted(tmp_path.glob("*.npz"))
    ds = LazyNpzDataset(paths, buffer_size=10, shuffle=False, seed=0)
    sample = next(iter(ds))
    assert sample["input_planes"].shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert sample["input_planes"].dtype == torch.float32
    assert sample["policy_target"].shape == (N_POLICY,)
    assert sample["policy_target"].dtype == torch.float32
    assert sample["value_target"].dtype == torch.float32
    assert sample["turn"].dtype == torch.long
    assert sample["ply"].dtype == torch.long


def test_lazynpzdataset_deterministic_with_seed(tmp_path: Path) -> None:
    """Same seed → same emission order (single-worker case)."""
    for g in range(1, 4):
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=5)
    paths = sorted(tmp_path.glob("*.npz"))
    ds_a = LazyNpzDataset(paths, buffer_size=4, shuffle=True, seed=123)
    ds_b = LazyNpzDataset(paths, buffer_size=4, shuffle=True, seed=123)
    plies_a = [int(s["ply"].item()) for s in ds_a]
    plies_b = [int(s["ply"].item()) for s in ds_b]
    assert plies_a == plies_b


def test_lazynpzdataset_different_seeds_differ(tmp_path: Path) -> None:
    """Different seeds → different emission order (with very high probability)."""
    for g in range(1, 6):
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=5)
    paths = sorted(tmp_path.glob("*.npz"))
    ds_a = LazyNpzDataset(paths, buffer_size=4, shuffle=True, seed=1)
    ds_b = LazyNpzDataset(paths, buffer_size=4, shuffle=True, seed=2)
    plies_a = [int(s["ply"].item()) for s in ds_a]
    plies_b = [int(s["ply"].item()) for s in ds_b]
    assert plies_a != plies_b


def test_lazynpzdataset_empty_paths_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paths must be non-empty"):
        LazyNpzDataset(paths=[], buffer_size=10)


def test_lazynpzdataset_invalid_buffer_size_raises(tmp_path: Path) -> None:
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=2)
    paths = sorted(tmp_path.glob("*.npz"))
    with pytest.raises(ValueError, match="buffer_size must be >= 1"):
        LazyNpzDataset(paths=paths, buffer_size=0)


def test_lazynpzdataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        LazyNpzDataset(paths=[tmp_path / "missing.npz"], buffer_size=10)


def test_lazynpzdataset_buffer_smaller_than_dataset(tmp_path: Path) -> None:
    """buffer_size < total samples : should still yield every sample exactly once."""
    for g in range(1, 5):
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=10)
    paths = sorted(tmp_path.glob("*.npz"))
    ds = LazyNpzDataset(paths, buffer_size=5, shuffle=True, seed=0)
    samples = _collect_samples(ds)
    assert len(samples) == 40


def test_lazynpzdataset_buffer_larger_than_dataset(tmp_path: Path) -> None:
    """buffer_size > total : full draining at end yields everything."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    paths = sorted(tmp_path.glob("*.npz"))
    ds = LazyNpzDataset(paths, buffer_size=1000, shuffle=True, seed=0)
    samples = _collect_samples(ds)
    assert len(samples) == 3


def test_lazynpzdataset_matches_eager_sample_set(tmp_path: Path) -> None:
    """Lazy emission must produce the same multiset of samples as eager (just reordered).

    We hash each sample by (ply, turn, value, policy.sum()) and compare counters.
    """
    from collections import Counter

    for g in range(1, 4):
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=5)
    paths = sorted(tmp_path.glob("*.npz"))

    def signature(s: dict[str, torch.Tensor]) -> tuple[int, int, float, float]:
        return (
            int(s["ply"].item()),
            int(s["turn"].item()),
            float(s["value_target"].item()),
            float(s["policy_target"].sum().item()),
        )

    # Eager set
    eager_sigs: list[tuple[int, int, float, float]] = []
    for p in paths:
        ds = NpzDataset(p)
        for i in range(len(ds)):
            eager_sigs.append(signature(ds[i]))

    # Lazy set
    lazy_ds = LazyNpzDataset(paths, buffer_size=4, shuffle=True, seed=42)
    lazy_sigs = [signature(s) for s in lazy_ds]

    assert Counter(eager_sigs) == Counter(lazy_sigs)


# ----- load_replay_buffer_lazy -----


def test_load_replay_buffer_lazy_combines_window(tmp_path: Path) -> None:
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=3, batch=0, n_samples=2)
    ds = load_replay_buffer_lazy(tmp_path, current_gen=3, window=10, buffer_size=5)
    assert ds.total_samples() == 9


def test_load_replay_buffer_lazy_window_filters(tmp_path: Path) -> None:
    for g in range(1, 6):
        _write_batch_at(tmp_path, gen=g, batch=0, n_samples=2)
    ds = load_replay_buffer_lazy(tmp_path, current_gen=5, window=2, buffer_size=4)
    assert ds.total_samples() == 4  # gens 4-5


def test_load_replay_buffer_lazy_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No batch files"):
        load_replay_buffer_lazy(tmp_path, current_gen=1, window=10, buffer_size=10)


def test_load_replay_buffer_lazy_directory_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Datasets directory"):
        load_replay_buffer_lazy(tmp_path / "nonexistent", current_gen=1, window=10, buffer_size=10)


# ----- browser_fraction (D.3 cloisonnement) -----


def _write_browser_batch_at(tmp_path: Path, gen: int, batch: int, n_samples: int = 2) -> Path:
    """Write a browser-cohort shard (chantier 1 naming : browser-genNNN-batch-MMM.npz)."""
    target = tmp_path / f"browser-gen{gen:03d}-batch-{batch:03d}.npz"
    samples = [_make_sample(ply=i) for i in range(n_samples)]
    write_batch(samples=samples, path=target, gen=gen, batch_idx=batch, n_games=1)
    return target


def test_shard_n_samples_reads_meta(tmp_path: Path) -> None:
    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=7)
    assert _shard_n_samples(path) == 7


def test_glob_browser_shards_only_matches_browser(tmp_path: Path) -> None:
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=3)  # fleet → ignored
    _write_browser_batch_at(tmp_path, gen=2, batch=0)
    _write_browser_batch_at(tmp_path, gen=2, batch=1)
    found = _glob_browser_shards(tmp_path, gen_start=1, current_gen=2)
    assert len(found) == 2
    assert all(p.name.startswith("browser-gen") for p in found)


def test_browser_fraction_zero_is_fleet_only(tmp_path: Path) -> None:
    """browser_fraction=0.0 (défaut) ignore les shards browser : chemin prod inchangé."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    for b in range(5):
        _write_browser_batch_at(tmp_path, gen=1, batch=b, n_samples=2)
    buf = load_replay_buffer(tmp_path, current_gen=1, window=1)
    assert len(buf) == 10  # browser shards NOT mixed in


def test_browser_fraction_caps_browser_share(tmp_path: Path) -> None:
    """browser ≤ f/(1-f) × fleet_n. fleet_n=10, f=0.1 → cap≈1.11 → 1 shard (2 samples)."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    for b in range(10):
        _write_browser_batch_at(tmp_path, gen=1, batch=b, n_samples=2)
    buf = load_replay_buffer(tmp_path, current_gen=1, window=1, browser_fraction=0.1)
    assert len(buf) == 12  # 10 fleet + 1 browser shard (greedy stops once cum >= cap)


def test_browser_fraction_half(tmp_path: Path) -> None:
    """f=0.5, fleet_n=10 → cap=10 → 5 browser shards (10 samples)."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    for b in range(10):
        _write_browser_batch_at(tmp_path, gen=1, batch=b, n_samples=2)
    buf = load_replay_buffer(tmp_path, current_gen=1, window=1, browser_fraction=0.5)
    assert len(buf) == 20  # 10 fleet + 10 browser samples


def test_browser_fraction_one_includes_all(tmp_path: Path) -> None:
    """f>=1.0 = no cap : every browser shard mixed in."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    for b in range(4):
        _write_browser_batch_at(tmp_path, gen=1, batch=b, n_samples=2)
    buf = load_replay_buffer(tmp_path, current_gen=1, window=1, browser_fraction=1.0)
    assert len(buf) == 18  # 10 fleet + 8 browser (all)


def test_browser_fraction_no_browser_shards_is_noop(tmp_path: Path) -> None:
    """browser_fraction>0 but zero browser shards present → fleet-only, no error."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=5)
    buf = load_replay_buffer(tmp_path, current_gen=1, window=1, browser_fraction=0.5)
    assert len(buf) == 5


def test_browser_fraction_respects_window(tmp_path: Path) -> None:
    """Browser shards outside the gen window are not globbed."""
    _write_batch_at(tmp_path, gen=5, batch=0, n_samples=10)
    _write_browser_batch_at(tmp_path, gen=1, batch=0, n_samples=2)  # out of window
    _write_browser_batch_at(tmp_path, gen=5, batch=0, n_samples=2)  # in window
    buf = load_replay_buffer(tmp_path, current_gen=5, window=1, browser_fraction=1.0)
    assert len(buf) == 12  # 10 fleet + only the in-window browser shard


def test_browser_fraction_lazy_path(tmp_path: Path) -> None:
    """The lazy loader mixes browser shards through the same cap logic."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    for b in range(10):
        _write_browser_batch_at(tmp_path, gen=1, batch=b, n_samples=2)
    ds = load_replay_buffer_lazy(
        tmp_path, current_gen=1, window=1, buffer_size=8, browser_fraction=0.5
    )
    assert ds.total_samples() == 20  # 10 fleet + 10 browser


def test_select_browser_shards_seed_deterministic(tmp_path: Path) -> None:
    """Same seed → identical shard selection ; cap respected (overshoot ≤ one shard)."""
    fleet = [_write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)]
    browser = [
        _write_browser_batch_at(tmp_path, gen=1, batch=i, n_samples=s)
        for i, s in enumerate([1, 2, 3, 4, 5])
    ]
    a = _select_browser_shards(fleet, browser, browser_fraction=0.5, seed=42)
    b = _select_browser_shards(fleet, browser, browser_fraction=0.5, seed=42)
    assert a == b  # reproducible
    # cap = 0.5/0.5 * 10 = 10 ; greedy stops once cum >= 10 → cum in [10, 14].
    assert 10 <= sum(_shard_n_samples(p) for p in a) <= 14


def test_select_browser_shards_edge_cases(tmp_path: Path) -> None:
    fleet = [_write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)]
    browser = [_write_browser_batch_at(tmp_path, gen=1, batch=0, n_samples=2)]
    assert _select_browser_shards(fleet, browser, browser_fraction=0.0, seed=1) == []
    assert _select_browser_shards(fleet, [], browser_fraction=0.5, seed=1) == []
    assert _select_browser_shards(fleet, browser, browser_fraction=1.0, seed=1) == browser


# ----- BatchedNpzDataset -----


def _collect_batched_plies(ds: BatchedNpzDataset) -> list[int]:
    plies: list[int] = []
    for batch in ds:
        plies.extend(int(p) for p in batch["ply"])
    return plies


def test_batched_yields_all_positions(tmp_path: Path) -> None:
    paths = [_write_batch_at(tmp_path, gen=1, batch=b, n_samples=5) for b in range(3)]
    ds = BatchedNpzDataset(paths, batch_size=4, shuffle=True, seed=0)
    plies = _collect_batched_plies(ds)
    assert len(plies) == 15  # 3 shards × 5
    assert sorted(plies) == sorted([0, 1, 2, 3, 4] * 3)


def test_batched_same_positions_as_lazy(tmp_path: Path) -> None:
    # Le batched yield exactement les mêmes positions que le lazy (à l'ordre près).
    paths = [_write_batch_at(tmp_path, gen=1, batch=b, n_samples=7) for b in range(2)]
    lazy = LazyNpzDataset(paths, buffer_size=10, shuffle=True, seed=1)
    batched = BatchedNpzDataset(paths, batch_size=4, shuffle=True, seed=1)
    assert sorted(int(s["ply"]) for s in lazy) == sorted(_collect_batched_plies(batched))


def test_batched_batch_shapes_and_dtypes(tmp_path: Path) -> None:
    paths = [_write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)]
    batches = list(BatchedNpzDataset(paths, batch_size=4, shuffle=False, seed=0))
    assert [b["input_planes"].shape[0] for b in batches] == [4, 4, 2]  # 10 → 4,4,2
    b0 = batches[0]
    assert b0["input_planes"].shape == (4, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert b0["policy_target"].shape == (4, N_POLICY)
    assert b0["value_target"].shape == (4,)
    assert b0["input_planes"].dtype == torch.float32
    assert b0["policy_target"].dtype == torch.float32
    assert b0["turn"].dtype == torch.int64
    assert b0["ply"].dtype == torch.int64


def test_batched_drop_last(tmp_path: Path) -> None:
    paths = [_write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)]
    ds = BatchedNpzDataset(paths, batch_size=4, shuffle=False, seed=0, drop_last=True)
    assert [b["input_planes"].shape[0] for b in ds] == [4, 4]  # partial last (2) dropped


def test_batched_no_shuffle_preserves_order(tmp_path: Path) -> None:
    paths = [_write_batch_at(tmp_path, gen=1, batch=0, n_samples=6)]
    ds = BatchedNpzDataset(paths, batch_size=3, shuffle=False, seed=0)
    assert _collect_batched_plies(ds) == [0, 1, 2, 3, 4, 5]


def test_batched_batch_size_validation(tmp_path: Path) -> None:
    paths = [_write_batch_at(tmp_path, gen=1, batch=0)]
    with pytest.raises(ValueError, match="batch_size"):
        BatchedNpzDataset(paths, batch_size=0)


def test_batched_empty_paths_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BatchedNpzDataset([], batch_size=4)


def test_load_replay_buffer_batched_combines_window(tmp_path: Path) -> None:
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=5)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=5)
    ds = load_replay_buffer_batched(tmp_path, current_gen=2, batch_size=4, window=2)
    assert len(_collect_batched_plies(ds)) == 10
