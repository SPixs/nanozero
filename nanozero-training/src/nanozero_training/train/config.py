"""Training configuration. Defaults alignés SPEC-training §4."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters (AlphaZero-style training).

    Defaults alignés SPEC-training §4.

    Attributes:
        batch_size: taille batch DataLoader.
        learning_rate: LR initial AdamW.
        total_epochs: nombre d'epochs par génération.
        l2_reg: coefficient weight_decay AdamW (decoupled, exclu BN+biases).
                Diagnostic 2026-05-21 gen-13 : valeur 1e-4 (héritage AGZ/AZ
                calibré SGD) trop forte pour Adam-family + cycle court 20
                epochs → 17 % poids en denormal. Réduite à 5e-5 (KataGo
                publié 3e-5). Cf. mémoire `value-head-improvements`.
        lr_min_ratio: ratio eta_min/lr_init pour CosineAnnealingLR.
                      Default 0.05 → eta_min = lr × 0.05 = 5e-5 (vs lr 1e-3).
                      Empêche LR → 0 en fin de schedule, élimine la pression
                      du weight_decay sans contre-gradient (cause des
                      denormals fin-de-training).
        value_target_smoothing: label smoothing de la cible value (z*(1-eps)),
                      dans [0, 1). Calibration anti-sur-confiance du value head
                      (v1.4.0). 0.0 = off. Conseillé ~0.05.
        replay_window: nombre de générations dans replay buffer.
        num_workers: workers DataLoader.
        shuffle: shuffle DataLoader.
        pin_memory: speedup GPU (cuda).
        max_grad_norm: gradient clipping (style Lc0, default 10000.0 :
                       safety net défensif, pas contrainte active en flow normal).
        use_cosine_schedule: CosineAnnealingLR T_max=total_steps.
        train_seed: reproductibilité.
    """

    # Architecture du réseau (rebump expérimental). Défaut 8×96 = archi figée
    # historique (rétrocompat bit-pour-bit). Permet d'entraîner un body plus
    # large (ex. 10×120) pour tester la capacité (cf. distillation Stockfish
    # 2026-06-03). Un net non-8×96 ne joue QUE via le backend ONNX Runtime
    # (le chemin SIMD .npz hardcode 8×96) — OK pour SPRT (les moteurs y passent).
    body_blocks: int = 8
    body_channels: int = 96

    # Core training
    batch_size: int = 256
    learning_rate: float = 1e-3
    total_epochs: int = 10
    l2_reg: float = 5e-5  # was 1e-4 — réduit suite diagnostic denormals 2026-05-21
    lr_min_ratio: float = 0.05  # eta_min = lr × ratio (floor cosine)

    # Value head calibration (v1.4.0) — label smoothing de la cible value : z*(1-eps).
    # Décourage la saturation tanh / sur-confiance du value head (diagnostic gen-24
    # king-safety + Lc0 WDL). 0.0 = off (v1.3.0). Valeur conseillée ~0.05 (cible ±0.95).
    value_target_smoothing: float = 0.0

    # Replay buffer
    replay_window: int = 10

    # D.3 cloisonnement (browser self-play) — fraction MAX de samples de la cohorte
    # navigateur ('browser-gen*', flushée séparément côté jobserver, hors corpus 'selfplay')
    # à mélanger au replay. 0.0 (défaut) = fleet only = chemin de prod inchangé, bit-pour-bit.
    # >0 = expérience A/B (ex. 0.10) ; le sous-échantillonnage est seedé par train_seed.
    browser_fraction: float = 0.0

    # DataLoader
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    # Gradient clipping (Lc0 convention : default généreux, log metric)
    max_grad_norm: float = 10000.0

    # LR schedule
    use_cosine_schedule: bool = True

    # Reproducibility
    train_seed: int = 42

    # Lazy replay buffer (post-v1.0.0 — eager NpzDataset = défaut bit-pour-bit identique).
    # Activer pour replay_window > 3 sur W3090 80 GB (eager OOM-swap au-delà).
    lazy_dataset: bool = False
    # Capacité du rolling shuffle buffer (samples par worker). 50k ≈ 1.5 GB/worker.
    # Augmenter (jusqu'à 500k) pour shuffle de meilleure qualité au prix de la RAM.
    lazy_buffer_size: int = 50_000

    # Pre-batched loader (2026-06-27, cf. mémoire training-data-pipeline-strategy).
    # Si True, utilise BatchedNpzDataset (yield de batches déjà empilés, DataLoader
    # batch_size=None) au lieu de LazyNpzDataset : supprime le per-item Python + la
    # collation mono-thread (mur GIL ~8800 pos/s → GPU affamé). Shuffle = ordre des
    # shards + permutation intra-shard (pas de buffer roulant cross-shard). RAM bornée
    # à ~1 shard/worker. Default False = chemin bit-pour-bit identique (LazyNpzDataset).
    # Prend le pas sur lazy_dataset quand les deux sont True.
    batched_dataset: bool = False

    # Phase A3 (2026-05-23) — continued training inter-générations.
    # Si True, run_train_phase charge les poids de gen-(N-1) avant d'entraîner gen-N
    # (load_model_for_resume depuis train-state-gen-(N-1)-epoch-(total_epochs-1).pt).
    # Pattern aligné Lc0/KataGo/AlphaZero (single network updated continually).
    # Default False = comportement bit-pour-bit identique pré-A3 (init aléatoire chaque gen).
    # L'optimizer state est volontairement RESET (Gupta 2023 — re-warm reset moments),
    # combiné à un warmup linéaire (warmup_epochs) avant le cosine.
    continued_training: bool = False
    # Nombre d'epochs en warmup linéaire (start_factor=0.01 → 1.0). Seulement actif si
    # continued_training=True. Default 1 epoch ≈ ~3 % des steps totaux (20 epochs).
    warmup_epochs: int = 1
