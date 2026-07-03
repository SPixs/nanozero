"""PyTorch Dataset over selfplay-data .npz files (ADR-002).

Two implementations cohabit :

- ``NpzDataset`` (eager, v1.0.0) : load 1 .npz fully in memory on construction.
  Combined with ``ConcatDataset`` for the full replay buffer.
- ``LazyNpzDataset`` (streaming, post-v1.0.0) : IterableDataset that opens each
  .npz on the fly, performs local shard-shuffle, then mixes shards through a
  rolling shuffle buffer (Lc0 style). Used when the eager replay buffer would
  not fit in RAM (e.g. ``replay_window`` > 3 on W3090 80 GB).

The lazy implementation accepts the standard PyTorch ``DataLoader`` contract
(``num_workers`` round-robin partition of shards, ``shuffle`` ignored because
shuffling is done internally — DataLoader ``shuffle`` MUST be False for
IterableDataset). Determinism : reproducible with a fixed ``seed`` per
worker, modulo OS scheduling that may reorder DataLoader batches between
workers across runs.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset

LOG = logging.getLogger(__name__)


class NpzDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset wrapping one selfplay-data .npz file.

    Eager loading: entire batch is loaded in memory on __init__.
    Returns dicts with keys 'input_planes', 'policy_target', 'value_target',
    'turn', 'ply' as torch.Tensor.
    """

    def __init__(self, path: str | Path) -> None:
        """Load a .npz batch file into memory."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        with np.load(path, allow_pickle=False) as data:
            self.input_planes = torch.from_numpy(data["input_planes"]).float()
            self.policy_target = torch.from_numpy(data["policy_target"]).float()
            self.value_target = torch.from_numpy(data["value_target"]).float()
            self.turn = torch.from_numpy(data["turn"]).long()
            self.ply = torch.from_numpy(data["ply"]).long()
            self._n = int(data["_meta_n_samples"].item())
        # Sanity check stacked sizes consistent.
        if len(self.input_planes) != self._n:
            raise ValueError(
                f"Inconsistent _meta_n_samples ({self._n}) vs actual "
                f"({len(self.input_planes)}) in {path}"
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")
        return {
            "input_planes": self.input_planes[idx],
            "policy_target": self.policy_target[idx],
            "value_target": self.value_target[idx],
            "turn": self.turn[idx],
            "ply": self.ply[idx],
        }


def _shard_n_samples(path: Path) -> int:
    """Read ``_meta_n_samples`` from a shard without fully loading its arrays.

    ``np.load`` on a ``.npz`` returns a lazy ``NpzFile`` : accessing one key
    decompresses only that small scalar, not the (119,8,8)/(4672,) sample arrays.
    """
    with np.load(path) as data:
        return int(data["_meta_n_samples"].item())


def _glob_browser_shards(datasets_dir: Path, gen_start: int, current_gen: int) -> list[Path]:
    """Browser-source shards for the window (chantier 1 naming : ``browser-gen{N}-batch-*``)."""
    paths: list[Path] = []
    for g in range(gen_start, current_gen + 1):
        paths.extend(sorted(datasets_dir.glob(f"browser-gen{g:03d}-batch-*.npz")))
    return paths


def _select_browser_shards(
    fleet_paths: list[Path],
    browser_paths: list[Path],
    browser_fraction: float,
    seed: int,
) -> list[Path]:
    """Pick browser shards so browser samples are ≤ ``browser_fraction`` of the total (D.3).

    ``browser / (fleet + browser) ≤ f``  ⟺  ``browser ≤ f/(1-f) × fleet_n``. Greedy fill of a
    seeded shuffle of the browser shards until the cap. ``f >= 1.0`` includes all (no cap) ;
    ``f <= 0`` includes none (the data stays quarantined — bit-for-bit fleet-only path).

    Args:
        fleet_paths: the fleet shards already selected (defines ``fleet_n``).
        browser_paths: candidate browser shards for the window.
        browser_fraction: target max fraction of browser samples in the mix.
        seed: seeds the browser-shard shuffle for reproducible experiments.

    Returns:
        The subset of browser shards to mix in (may be empty).
    """
    if not browser_paths or browser_fraction <= 0.0:
        return []
    if browser_fraction >= 1.0:
        return list(browser_paths)
    fleet_n = sum(_shard_n_samples(p) for p in fleet_paths)
    cap = browser_fraction / (1.0 - browser_fraction) * fleet_n
    shuffled = list(browser_paths)
    random.Random(seed).shuffle(shuffled)
    selected: list[Path] = []
    cum = 0
    for p in shuffled:
        if cum >= cap:
            break
        selected.append(p)
        cum += _shard_n_samples(p)
    return selected


def load_replay_buffer(
    datasets_dir: str | Path,
    current_gen: int,
    window: int = 10,
    browser_fraction: float = 0.0,
    seed: int = 42,
) -> ConcatDataset[dict[str, torch.Tensor]]:
    """Load the last K generations of selfplay-data into a ConcatDataset.

    Args:
        datasets_dir: directory containing selfplay-genNNN-batch-MMM.npz files
                      (typically S:\\nano2\\datasets\\).
        current_gen: current generation index. Generations
                     [max(1, current_gen-window+1), current_gen] are included.
        window: number of generations to include (default 10).
        browser_fraction: D.3 — fraction MAX de samples ``browser-gen*`` à mélanger au corpus
                      (0.0 défaut = fleet only = chemin v1.0.0 bit-pour-bit préservé). >0 mélange
                      la cohorte navigateur cloisonnée (chantier 1) à cette fraction max.
        seed: seed du sous-échantillonnage browser (expériences reproductibles).

    Returns:
        ConcatDataset combining all matching .npz files.

    Raises:
        FileNotFoundError: if datasets_dir doesn't exist.
        ValueError: if no batch files found for the requested window.
    """
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    gen_start = max(1, current_gen - window + 1)
    paths: list[Path] = []
    for g in range(gen_start, current_gen + 1):
        pattern = f"selfplay-gen{g:03d}-batch-*.npz"
        paths.extend(sorted(datasets_dir.glob(pattern)))

    if not paths:
        raise ValueError(
            f"No batch files found for gens [{gen_start}, {current_gen}] in {datasets_dir}"
        )

    if browser_fraction > 0.0:
        browser = _select_browser_shards(
            paths,
            _glob_browser_shards(datasets_dir, gen_start, current_gen),
            browser_fraction,
            seed,
        )
        if browser:
            LOG.info(
                "Replay mix : %d fleet shards + %d browser shards (browser_fraction=%.3f cap)",
                len(paths),
                len(browser),
                browser_fraction,
            )
            paths.extend(browser)

    datasets = [NpzDataset(p) for p in paths]
    return ConcatDataset(datasets)


class LazyNpzDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Streaming dataset over a list of .npz shards (Lc0-style shuffle buffer).

    Each shard is opened on demand, locally shuffled, then funneled through a
    bounded rolling buffer that mixes samples between shards before emission.
    Memory usage is bounded by ``buffer_size`` samples per worker (≈ 30 KB per
    sample, so 50k buffer ≈ 1.5 GB / worker).

    DataLoader contract :

    - Pass ``shuffle=False`` to DataLoader (IterableDataset is incompatible
      with DataLoader-level shuffle).
    - ``num_workers >= 1`` is supported : each worker takes a round-robin
      partition of the shards and an independent shuffle buffer.
    - ``__len__`` is NOT provided (IterableDataset convention). Use
      ``total_samples()`` to inspect the total count if needed.

    Args :
        paths : list of .npz shard paths to stream over. Order is irrelevant
                (paths are shuffled per worker if ``shuffle=True``).
        buffer_size : capacity of the rolling shuffle buffer per worker.
                Larger = better global shuffle, more RAM. Default 50_000.
        shuffle : if True, shuffle paths order, shuffle samples within each
                shard, and shuffle through the rolling buffer. If False, emit
                samples in deterministic order (path order, sample order
                within shard, no buffer mixing).
        seed : base random seed. Each worker derives its own RNG from
                ``seed + worker_id`` for reproducibility.
    """

    def __init__(
        self,
        paths: list[Path],
        buffer_size: int = 50_000,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        if not paths:
            raise ValueError("paths must be non-empty")
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")
        self.paths = [Path(p) for p in paths]
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed

    def total_samples(self) -> int:
        """Return total number of samples across all shards (reads metadata only)."""
        total = 0
        for p in self.paths:
            with np.load(p, allow_pickle=False) as data:
                total += int(data["_meta_n_samples"].item())
        return total

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_paths = list(self.paths)
            worker_id = 0
        else:
            # Round-robin partition : worker i gets shards [i, i+W, i+2W, ...].
            my_paths = self.paths[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id
        if not my_paths:
            return
        rng = np.random.RandomState(self.seed + worker_id)
        if self.shuffle:
            # rng.shuffle does not accept list[Path] (numpy strict types) ;
            # permute via indices instead.
            perm = rng.permutation(len(my_paths))
            my_paths = [my_paths[i] for i in perm]

        # Rolling shuffle buffer : a list, popped via swap-and-pop O(1).
        buffer: list[dict[str, torch.Tensor]] = []

        for path in my_paths:
            # Open shard, materialize numpy arrays (decompress if .npz compressed).
            with np.load(path, allow_pickle=False) as data:
                n = int(data["_meta_n_samples"].item())
                planes = np.asarray(data["input_planes"])
                policy = np.asarray(data["policy_target"])
                value = np.asarray(data["value_target"])
                turn = np.asarray(data["turn"])
                ply = np.asarray(data["ply"])
                if len(planes) != n:
                    raise ValueError(
                        f"Inconsistent _meta_n_samples ({n}) vs actual "
                        f"({len(planes)}) in {path}"
                    )
                local_indices = np.arange(n)
                if self.shuffle:
                    rng.shuffle(local_indices)

                for i in local_indices:
                    # .copy() : rend chaque sample indépendant du shard numpy parent.
                    # Sans ça, torch.from_numpy(planes[i]) garde une VUE sur le tableau
                    # décompressé entier (planes float32) → le shard reste en RAM tant
                    # qu'un de ses samples est dans le shuffle buffer. Avec des shards
                    # volumineux (jusqu'à ~5 GB décompressés) × num_workers, la RAM
                    # explose. Le .copy() borne la RAM au seul buffer.
                    sample = {
                        "input_planes": torch.from_numpy(planes[i].copy()).float(),
                        "policy_target": torch.from_numpy(policy[i].copy()).float(),
                        "value_target": torch.tensor(float(value[i])),
                        "turn": torch.tensor(int(turn[i]), dtype=torch.long),
                        "ply": torch.tensor(int(ply[i]), dtype=torch.long),
                    }
                    if not self.shuffle:
                        yield sample
                        continue
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        # Swap-and-pop : O(1) random removal.
                        idx = int(rng.randint(0, len(buffer)))
                        out = buffer[idx]
                        buffer[idx] = buffer[-1]
                        buffer.pop()
                        yield out

        # Drain remaining buffer in random (or insertion) order.
        if self.shuffle:
            while buffer:
                idx = int(rng.randint(0, len(buffer)))
                out = buffer[idx]
                buffer[idx] = buffer[-1]
                buffer.pop()
                yield out
        # (If shuffle=False, buffer is always empty here.)


class BatchedNpzDataset(IterableDataset[dict[str, torch.Tensor]]):
    """IterableDataset yielding PRE-STACKED batches (use DataLoader ``batch_size=None``).

    Optimisation pipeline (2026-06-27, cf. mémoire ``training-data-pipeline-strategy``).
    Le ``LazyNpzDataset`` produit 1 dict PAR position → ~``batch_size`` appels Python
    par batch (création de dict + ``.copy()``/``.float()`` ×5 + ops shuffle-buffer) PUIS
    une collation ``batch_size``-items dans le process principal mono-thread : c'est le
    mur GIL (~8800 pos/s, GPU affamé à ~30 % mesuré). Ici chaque worker découpe son shard
    en batches déjà empilés par UNE indexation numpy vectorisée par batch, et yield le
    batch complet. Avec ``batch_size=None`` le DataLoader ne re-batch pas : il ne fait que
    recevoir un handle mémoire-partagée → supprime le per-item Python ET la collation.

    Mémoire : bornée à ~1 shard/worker. Chaque batch est ``planes[batch_indices]`` =
    une indexation *fancy* numpy → tableau contigu à storage PROPRE (pas une vue du shard),
    donc l'IPC envoie ~17 Mo/batch et non le shard entier, et aucun shard n'est épinglé
    après son traitement. (Contraste avec ``LazyNpzDataset`` dont le buffer roulant à
    éviction aléatoire impose le ``.copy()`` par-item pour ne pas épingler le shard.)

    Shuffle : ordre des shards (par worker) + permutation intra-shard. Le mélange
    cross-shard du buffer roulant est abandonné — acceptable car les shards sont déjà
    mêlés à la source (assignation aléatoire des fragments de parties par le jobserver).
    ⚠️ À valider empiriquement (courbe de loss intra-gen + SPRT) avant de remplacer le
    défaut ``LazyNpzDataset``.

    Sortie identique (shapes/dtypes/valeurs) à la collation du ``LazyNpzDataset``, à
    l'ORDRE des positions près.
    """

    def __init__(
        self,
        paths: list[Path],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not paths:
            raise ValueError("paths must be non-empty")
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")
        self.paths = [Path(p) for p in paths]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_paths = list(self.paths)
            worker_id = 0
        else:
            # Round-robin partition : worker i gets shards [i, i+W, i+2W, ...].
            my_paths = self.paths[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id
        if not my_paths:
            return
        rng = np.random.RandomState(self.seed + worker_id)
        if self.shuffle:
            perm = rng.permutation(len(my_paths))
            my_paths = [my_paths[i] for i in perm]

        for path in my_paths:
            with np.load(path, allow_pickle=False) as data:
                n = int(data["_meta_n_samples"].item())
                planes = np.asarray(data["input_planes"])
                policy = np.asarray(data["policy_target"])
                value = np.asarray(data["value_target"])
                turn = np.asarray(data["turn"])
                ply = np.asarray(data["ply"])
                if len(planes) != n:
                    raise ValueError(
                        f"Inconsistent _meta_n_samples ({n}) vs actual "
                        f"({len(planes)}) in {path}"
                    )
                order = rng.permutation(n) if self.shuffle else np.arange(n)
                for start in range(0, n, self.batch_size):
                    bidx = order[start : start + self.batch_size]
                    if self.drop_last and len(bidx) < self.batch_size:
                        break
                    # Indexation fancy = copie vectorisée vers un tableau contigu à
                    # storage propre (1 memcpy par champ, pas N appels Python).
                    yield {
                        "input_planes": torch.from_numpy(planes[bidx]).float(),
                        "policy_target": torch.from_numpy(policy[bidx]).float(),
                        "value_target": torch.from_numpy(value[bidx]).float(),
                        "turn": torch.from_numpy(turn[bidx]).long(),
                        "ply": torch.from_numpy(ply[bidx]).long(),
                    }


def load_replay_buffer_lazy(
    datasets_dir: str | Path,
    current_gen: int,
    window: int = 10,
    buffer_size: int = 50_000,
    shuffle: bool = True,
    seed: int = 42,
    browser_fraction: float = 0.0,
) -> LazyNpzDataset:
    """Streaming analog of ``load_replay_buffer`` returning a LazyNpzDataset.

    Same window semantics : generations
    ``[max(1, current_gen - window + 1), current_gen]`` are included. Shards
    are listed in deterministic path order ; the LazyNpzDataset then handles
    per-worker partition + shuffle. ``browser_fraction`` (D.3) mixes the cloisonné
    browser cohort at a bounded fraction — the rolling shuffle buffer blends the
    browser + fleet shards at the sample level for free. 0.0 = fleet only.

    Raises :
        FileNotFoundError : if ``datasets_dir`` doesn't exist.
        ValueError : if no batch files found for the requested window.
    """
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    gen_start = max(1, current_gen - window + 1)
    paths: list[Path] = []
    for g in range(gen_start, current_gen + 1):
        pattern = f"selfplay-gen{g:03d}-batch-*.npz"
        paths.extend(sorted(datasets_dir.glob(pattern)))

    if not paths:
        raise ValueError(
            f"No batch files found for gens [{gen_start}, {current_gen}] in {datasets_dir}"
        )

    if browser_fraction > 0.0:
        browser = _select_browser_shards(
            paths,
            _glob_browser_shards(datasets_dir, gen_start, current_gen),
            browser_fraction,
            seed,
        )
        if browser:
            LOG.info(
                "Replay mix (lazy) : %d fleet + %d browser shards (browser_fraction=%.3f cap)",
                len(paths),
                len(browser),
                browser_fraction,
            )
            paths.extend(browser)

    return LazyNpzDataset(paths=paths, buffer_size=buffer_size, shuffle=shuffle, seed=seed)


def load_replay_buffer_batched(
    datasets_dir: str | Path,
    current_gen: int,
    batch_size: int,
    window: int = 10,
    shuffle: bool = True,
    seed: int = 42,
    browser_fraction: float = 0.0,
    drop_last: bool = False,
) -> BatchedNpzDataset:
    """Pre-batched analog of ``load_replay_buffer_lazy`` returning a BatchedNpzDataset.

    Même sémantique de fenêtre et de collecte de shards (fleet + browser cap) que
    ``load_replay_buffer_lazy`` ; seul le dataset diffère (yield de batches pré-empilés
    au lieu de samples individuels). Cf. ``BatchedNpzDataset`` pour le compromis shuffle.

    Raises :
        FileNotFoundError : si ``datasets_dir`` n'existe pas.
        ValueError : si aucun shard trouvé pour la fenêtre.
    """
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    gen_start = max(1, current_gen - window + 1)
    paths: list[Path] = []
    for g in range(gen_start, current_gen + 1):
        pattern = f"selfplay-gen{g:03d}-batch-*.npz"
        paths.extend(sorted(datasets_dir.glob(pattern)))

    if not paths:
        raise ValueError(
            f"No batch files found for gens [{gen_start}, {current_gen}] in {datasets_dir}"
        )

    if browser_fraction > 0.0:
        browser = _select_browser_shards(
            paths,
            _glob_browser_shards(datasets_dir, gen_start, current_gen),
            browser_fraction,
            seed,
        )
        if browser:
            LOG.info(
                "Replay mix (batched) : %d fleet + %d browser shards (browser_fraction=%.3f cap)",
                len(paths),
                len(browser),
                browser_fraction,
            )
            paths.extend(browser)

    return BatchedNpzDataset(
        paths=paths, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last
    )
