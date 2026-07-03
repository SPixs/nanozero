package org.nanozero.uci.internal;

import java.util.List;

/**
 * Holder mutable des options UCI courantes (cf. SPEC §6.2, ADR-003, §12 phase 1).
 *
 * <p>Stocke les valeurs en cours des options UCI déclarées au boot ({@code Ponder}, {@code Move
 * Overhead}). Modifié à la réception d'une commande {@code setoption}, lu par les autres composants
 * UCI (typiquement {@code TimeManagementPolicy} en phase 5).
 *
 * <p><strong>Concurrence</strong> : les champs sont {@code volatile}. En pratique, l'écriture (via
 * {@code setoption}) et la lecture sont toutes deux sur le main thread (cf. SPEC §7.1 threading
 * model), mais la convention défensive est cohérente avec les patterns du module {@code
 * nanozero-engine} (cf. ADR-004 engine).
 *
 * <p><strong>Tolérance UCI</strong> ({@link #set}) : conformément au protocole UCI, les options
 * inconnues, valeurs invalides (parse {@link NumberFormatException}), et arguments {@code null}
 * sont silencieusement ignorés — pas d'exception propagée vers la boucle UCI principale.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciOptionsState {

  /** Default Ponder (cf. ADR-003). */
  private static final boolean DEFAULT_PONDER = true;

  /** Default Move Overhead en millisecondes (cf. ADR-003 § Move Overhead). */
  private static final int DEFAULT_MOVE_OVERHEAD_MS = 30;

  /** Borne min Move Overhead. */
  private static final int MOVE_OVERHEAD_MIN_MS = 0;

  /** Borne max Move Overhead. */
  private static final int MOVE_OVERHEAD_MAX_MS = 5000;

  /** Default DirichletAlpha (spin x1000). 0 = désactivé (cf. SPEC §6.4, ADR-003 v1.1.0). */
  private static final int DEFAULT_DIRICHLET_ALPHA = 0;

  /** Borne max DirichletAlpha (spin x1000 → 10.0 après division). */
  private static final int DIRICHLET_ALPHA_MAX = 10000;

  /** Default DirichletEpsilon (spin x1000). 0 = désactivé. */
  private static final int DEFAULT_DIRICHLET_EPSILON = 0;

  /** Borne max DirichletEpsilon (spin x1000 → 1.0 après division). */
  private static final int DIRICHLET_EPSILON_MAX = 1000;

  /** Default DirichletSeed (string parsée en long). "0" = déterministe. */
  private static final String DEFAULT_DIRICHLET_SEED = "0";

  /** Default Threads (cf. engine v1.2.0). 1 = mono-thread strict (compat v1.1.2 bit-for-bit). */
  private static final int DEFAULT_THREADS = 1;

  /** Borne min Threads. */
  private static final int THREADS_MIN = 1;

  /** Borne max Threads (cf. EngineConfig.MAX_SEARCH_THREADS). */
  private static final int THREADS_MAX = 128;

  /**
   * (v1.3.0) Default BatchSize : sentinelle {@code 0} = "non set" → auto-config selon {@code
   * useCuda} dans {@link UciAdapterState#buildEngineConfigFromOptions}. Une fois set par {@code
   * setoption}, valeur ∈ [{@link #BATCH_SIZE_MIN}, {@link #BATCH_SIZE_MAX}]. Sentinelle nécessaire
   * pour distinguer "default" de "set à 1" (override explicite). Cf. ADR-008-uci.
   */
  private static final int DEFAULT_BATCH_SIZE = 0;

  /** Borne min BatchSize quand set. */
  private static final int BATCH_SIZE_MIN = 1;

  /** Borne max BatchSize (cohérente avec Network.MAX_BATCH côté engine). */
  private static final int BATCH_SIZE_MAX = 64;

  /**
   * Default UCI option {@code BatchSize} displayed = {@code 1} (auto-config quand non override).
   */
  private static final int DEFAULT_BATCH_SIZE_DECLARED = 1;

  /** (ADR-018) Default {@code NNCacheSize} : {@code 0} = cache d'évaluation NN désactivé. */
  private static final int DEFAULT_NN_CACHE_SIZE = 0;

  /** Borne min {@code NNCacheSize}. {@code 0} = désactivé. */
  private static final int NN_CACHE_SIZE_MIN = 0;

  /** Borne max {@code NNCacheSize} (≈ 16 M entrées). */
  private static final int NN_CACHE_SIZE_MAX = 1 << 24;

  private volatile boolean ponder = DEFAULT_PONDER;
  private volatile int moveOverheadMs = DEFAULT_MOVE_OVERHEAD_MS;
  private volatile int dirichletAlpha = DEFAULT_DIRICHLET_ALPHA;
  private volatile int dirichletEpsilon = DEFAULT_DIRICHLET_EPSILON;
  private volatile String dirichletSeed = DEFAULT_DIRICHLET_SEED;
  private volatile int threads = DEFAULT_THREADS;
  private volatile int batchSize = DEFAULT_BATCH_SIZE;
  private volatile int nnCacheSize = DEFAULT_NN_CACHE_SIZE;

  /** Valeur courante de l'option {@code Ponder} (cf. ADR-003). Lecture lock-free. */
  public boolean ponder() {
    return ponder;
  }

  /**
   * Valeur courante de l'option {@code Move Overhead} en millisecondes (cf. ADR-003). Lecture
   * lock-free, valeur garantie dans {@code [0, 5000]}.
   */
  public int moveOverheadMs() {
    return moveOverheadMs;
  }

  /**
   * (v1.1.0) Valeur courante de l'option hidden {@code DirichletAlpha} (spin x1000, cf. SPEC §6.4).
   * Range garanti {@code [0, 10000]} → {@code [0.0, 10.0]} après division. 0 = Dirichlet désactivé.
   */
  public int dirichletAlpha() {
    return dirichletAlpha;
  }

  /**
   * (v1.1.0) Valeur courante de l'option hidden {@code DirichletEpsilon} (spin x1000). Range
   * garanti {@code [0, 1000]} → {@code [0.0, 1.0]} après division. 0 = Dirichlet désactivé.
   */
  public int dirichletEpsilon() {
    return dirichletEpsilon;
  }

  /**
   * (v1.1.0) Valeur courante de l'option hidden {@code DirichletSeed} (string parsée en long). "0"
   * par défaut.
   */
  public String dirichletSeed() {
    return dirichletSeed;
  }

  /**
   * Valeur courante de l'option {@code Threads} (cf. engine v1.2.0). Range garanti {@code [1,
   * 128]}. 1 = mono-thread strict (bit-for-bit identique au comportement v1.1.2). {@code > 1} =
   * MCTS multi-thread (tree parallelization + virtual loss).
   */
  public int threads() {
    return threads;
  }

  /**
   * (v1.3.0) Override explicite de {@code batchSize} via setoption UCI. Retourne {@code 0} quand
   * non set par l'utilisateur (= auto-config selon {@code useCuda} dans {@link
   * UciAdapterState#buildEngineConfigFromOptions}). Retourne ∈ [1, {@link #BATCH_SIZE_MAX}] quand
   * set explicitement. Cf. ADR-008-uci.
   */
  public int batchSizeOverride() {
    return batchSize;
  }

  /**
   * (ADR-018) Valeur courante de l'option {@code NNCacheSize} (nombre d'entrées du cache
   * d'évaluation NN). Range garanti {@code [0, 16M]}. {@code 0} = cache désactivé (défaut),
   * comportement strictement préservé bit-pour-bit. Pré-amorçable au boot via le flag CLI {@code
   * --nncache} (cf. {@code UciMain}), surchargeable runtime via {@code setoption}.
   */
  public int nnCacheSize() {
    return nnCacheSize;
  }

  /**
   * Modifie une option par nom. Comportement strictement tolérant conformément au protocole UCI :
   *
   * <ul>
   *   <li>{@code name == null} : silencieusement ignoré, aucune option modifiée.
   *   <li>option inconnue : silencieusement ignorée.
   *   <li>{@code Ponder} : {@code value} parsé via {@link Boolean#parseBoolean} (case-insensitive,
   *       toute valeur non-"true" donne {@code false}). Si {@code value == null}, ignoré.
   *   <li>{@code Move Overhead} : {@code value} parsé via {@link Integer#parseInt} après {@code
   *       trim()}. Si parse échoue ({@link NumberFormatException}), silencieusement ignoré. Si
   *       {@code value == null}, ignoré. La valeur est clampée dans {@code [0, 5000]} pour absorber
   *       les GUIs mal calibrés.
   *   <li>(v1.1.0 hidden) {@code DirichletAlpha} : spin x1000, clampé dans {@code [0, 10000]}.
   *   <li>(v1.1.0 hidden) {@code DirichletEpsilon} : spin x1000, clampé dans {@code [0, 1000]}.
   *   <li>(v1.1.0 hidden) {@code DirichletSeed} : string trimmée, stockée telle quelle. La
   *       conversion en long est faite à la lecture (consommateur).
   * </ul>
   *
   * @param name nom de l'option (peut être {@code null}, ignoré)
   * @param value valeur (peut être {@code null}, ignoré)
   */
  public void set(String name, String value) {
    if (name == null) {
      return;
    }
    switch (name) {
      case "Ponder" -> {
        if (value != null) {
          ponder = Boolean.parseBoolean(value.trim());
        }
      }
      case "Move Overhead" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            moveOverheadMs = Math.max(MOVE_OVERHEAD_MIN_MS, Math.min(MOVE_OVERHEAD_MAX_MS, parsed));
          } catch (NumberFormatException ignored) {
            // UCI tolérance : silently ignore invalid integer values.
          }
        }
      }
      // v1.1.0 hidden options (cf. SPEC §6.4, ADR-003 mise à jour v1.1.0).
      case "DirichletAlpha" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            dirichletAlpha = Math.max(0, Math.min(DIRICHLET_ALPHA_MAX, parsed));
          } catch (NumberFormatException ignored) {
            // Silently ignore invalid (UCI tolérance).
          }
        }
      }
      case "DirichletEpsilon" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            dirichletEpsilon = Math.max(0, Math.min(DIRICHLET_EPSILON_MAX, parsed));
          } catch (NumberFormatException ignored) {
            // Silently ignore invalid.
          }
        }
      }
      case "DirichletSeed" -> {
        if (value != null) {
          dirichletSeed = value.trim();
        }
      }
      case "Threads" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            threads = Math.max(THREADS_MIN, Math.min(THREADS_MAX, parsed));
          } catch (NumberFormatException ignored) {
            // UCI tolérance : silently ignore invalid integer values.
          }
        }
      }
      // (v1.3.0) BatchSize override pour mode batched GPU (cf. ADR-008-uci).
      case "BatchSize" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            batchSize = Math.max(BATCH_SIZE_MIN, Math.min(BATCH_SIZE_MAX, parsed));
          } catch (NumberFormatException ignored) {
            // UCI tolérance : silently ignore invalid integer values.
          }
        }
      }
      // (ADR-018) NNCacheSize : taille du cache d'évaluation NN optionnel. 0 = désactivé.
      case "NNCacheSize" -> {
        if (value != null) {
          try {
            int parsed = Integer.parseInt(value.trim());
            nnCacheSize = Math.max(NN_CACHE_SIZE_MIN, Math.min(NN_CACHE_SIZE_MAX, parsed));
          } catch (NumberFormatException ignored) {
            // UCI tolérance : silently ignore invalid integer values.
          }
        }
      }
      default -> {
        // UCI tolérance : silently ignore unknown options (cf. SPEC §6.2).
      }
    }
  }

  /**
   * Liste des options déclarées au protocole UCI au boot (cf. ADR-003). Émise après réception de
   * {@code uci} pour annoncer au GUI les options disponibles.
   *
   * <p>v1.0.0 : 2 options exposées :
   *
   * <ul>
   *   <li>{@code Check Ponder} default {@code true}.
   *   <li>{@code Spin "Move Overhead"} default {@code 30}, range {@code [0, 5000]}.
   * </ul>
   *
   * <p>Hyperparamètres MCTS ({@code c_puct}, {@code fpu_value}, etc.) NON exposés en v1.0.0 (cf.
   * ADR-003) — réservés au CLI {@link org.nanozero.uci.UciMain} pour empêcher le tuning runtime via
   * GUI.
   */
  public static List<UciOption> declaredOptions() {
    return List.of(
        new UciOption.Check("Ponder", DEFAULT_PONDER),
        new UciOption.Spin(
            "Move Overhead", DEFAULT_MOVE_OVERHEAD_MS, MOVE_OVERHEAD_MIN_MS, MOVE_OVERHEAD_MAX_MS),
        new UciOption.Spin("Threads", DEFAULT_THREADS, THREADS_MIN, THREADS_MAX),
        new UciOption.Spin(
            "BatchSize", DEFAULT_BATCH_SIZE_DECLARED, BATCH_SIZE_MIN, BATCH_SIZE_MAX),
        new UciOption.Spin(
            "NNCacheSize", DEFAULT_NN_CACHE_SIZE, NN_CACHE_SIZE_MIN, NN_CACHE_SIZE_MAX));
  }
}
