"""Re-shard un corpus self-play en shards de taille UNIFORME.

Motivation
----------
Les shards NPZ produits par le jobserver sont de tailles très inégales :
quelques « monstres » (~100k positions = ~5 GB décompressés, ratio zlib ~131x)
côtoient une majorité de shards minuscules. Le ``LazyNpzDataset`` décompresse un
shard entier d'un coup ; tomber sur un monstre fait staller un worker plusieurs
secondes (0 production → GPU affamé) et fait gonfler la RAM (un monstre
décompressé × num_workers → risque d'OOM en conteneur mémoire-limité).

Re-découper tout en pièces uniformes (~10k positions) lisse le feeding
(décompression courte et régulière) et borne la RAM par worker, ce qui permet
de monter ``num_workers`` et de saturer un GPU rapide.

Le format est CONSERVÉ À L'IDENTIQUE (mêmes clés, dtypes, ``savez_compressed``)
→ le ``LazyNpzDataset`` et la loss sont inchangés : aucun risque sur la qualité
d'entraînement, c'est purement un re-découpage.

Usage
-----
    python scripts/reshard_uniform.py \
        --src S:/nano2/runs/w3090-v1.0.0/datasets \
        --dst S:/nano2/runs/w3090-v1.0.0/datasets_u \
        --gen 31 --target 10000 --nproc 6

⚠️ ``--nproc`` borne la RAM : chaque worker peut tenir un shard monstre
décompressé (~5 GB) → nproc=6 ≈ 30 GB de pic. Adapter à la RAM dispo.
Les petits shards (< ``--monster-mb``) sont hardlinkés (instantané, même FS) ;
seuls les monstres sont décompressés/re-compressés.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

KEYS = ["input_planes", "policy_target", "value_target", "turn", "ply"]


def split_monster(args: tuple[str, str, int, int]) -> int:
    """Découpe un shard monstre en pièces de ``target`` positions.

    Args:
        args: (chemin_source, dossier_dest, target_positions, model_version).

    Returns:
        Nombre de positions traitées.
    """
    src, dst, target, model_version = args
    base = os.path.basename(src)[:-4]
    with np.load(src) as data:
        n = int(data["_meta_n_samples"])
        arrs = {k: np.asarray(data[k]) for k in KEYS}
    for j in range((n + target - 1) // target):
        a, b = j * target, min((j + 1) * target, n)
        out = {k: arrs[k][a:b] for k in KEYS}
        meta = {
            "_meta_n_samples": np.int64(b - a),
            "_meta_model_version": np.int64(model_version),
            "_meta_batch_idx": np.int64(j),
        }
        np.savez_compressed(f"{dst}/{base}_p{j:02d}.npz", **out, **meta)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-shard un corpus en tailles uniformes.")
    ap.add_argument("--src", required=True, help="Dossier des shards source.")
    ap.add_argument("--dst", required=True, help="Dossier de sortie (créé si absent).")
    ap.add_argument("--gen", type=int, required=True, help="N° de gen (glob selfplay-genNNN).")
    ap.add_argument("--target", type=int, default=10_000, help="Positions par shard de sortie.")
    ap.add_argument("--monster-mb", type=float, default=5.0, help="Seuil 'monstre' (MB compressé).")
    ap.add_argument("--nproc", type=int, default=6, help="Workers parallèles (borne la RAM).")
    ap.add_argument("--model-version", type=int, default=None, help="Override _meta_model_version.")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    pattern = f"{args.src}/selfplay-gen{args.gen:03d}-batch-*.npz"
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"Aucun shard pour {pattern}")
    mv = args.model_version if args.model_version is not None else args.gen - 1

    threshold = args.monster_mb * 1e6
    monsters = [s for s in shards if os.path.getsize(s) > threshold]
    smalls = [s for s in shards if os.path.getsize(s) <= threshold]

    for s in smalls:
        link = f"{args.dst}/{os.path.basename(s)}"
        if not os.path.exists(link):
            os.link(s, link)

    print(f"{len(smalls)} petits hardlinkés | {len(monsters)} monstres splittés "
          f"(target={args.target}, nproc={args.nproc})", flush=True)

    t0 = time.perf_counter()
    work = [(s, args.dst, args.target, mv) for s in monsters]
    with Pool(args.nproc) as pool:
        res = pool.map(split_monster, work)
    n_out = len(glob.glob(f"{args.dst}/selfplay-gen{args.gen:03d}-batch-*.npz"))
    print(f"OK: {sum(res)} positions re-shardées | {n_out} shards dans {args.dst} "
          f"| {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
