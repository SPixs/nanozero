package org.nanozero.engine;

import org.nanozero.nn.Network;

/**
 * Hyperparamètres immutables d'une instance {@code Engine} (cf. SPEC §3.5, §6, §15.3).
 *
 * <p>Trois paramètres v1.0.0 (alignés avec la simplicité revendiquée du moteur : pas de
 * température, pas de transposition table — cf. ADR-008, ADR-009) + trois paramètres v1.1.0 pour
 * configurer le Dirichlet noise au root MCTS (cf. ADR-012) + trois paramètres v1.2.0 pour
 * configurer le mode batched multi-thread (cf. ADR-013, ADR-014, ADR-015).
 *
 * <p>Cohabitation stricte v1.1.2 ↔ v1.2.0 : les défauts v1.2.0 ({@code searchThreads=1,
 * batchSize=1, virtualLoss=0.0f}) préservent strictement le comportement v1.1.2 bit-pour-bit. Le
 * mode batched s'active uniquement quand {@code searchThreads ≥ 2}.
 *
 * @param cPuct constante d'exploration de la formule PUCT (cf. §5.2). Valeurs typiques [1.0, 4.0],
 *     défaut 2.5. {@code cPuct} faible privilégie l'exploitation, élevé privilégie l'exploration.
 *     DOIT être strictement positif.
 * @param fpuValue valeur Q par défaut pour un child non-visité (First Play Urgency, cf. §6.2).
 *     Valeurs typiques [-0.3, +0.3], défaut 0.0 (FPU neutre). DOIT être dans {@code [-1, +1]} pour
 *     être cohérent avec le domaine de Q.
 * @param treeInitialCapacity hint dimensionnement initial du buffer interne de l'arbre (nombre de
 *     nœuds réservés à la création). N'impacte pas la sémantique, uniquement l'allocation initiale.
 *     Défaut 1024. DOIT être {@code ≥ 1}.
 * @param dirichletAlpha (v1.1.0) paramètre α de la distribution Dirichlet pour le noise au root
 *     (cf. ADR-012, SPEC §3.5). Défaut 0.0 (Dirichlet désactivé). DOIT être {@code ≥ 0}. Si {@code
 *     dirichletEpsilon > 0}, DOIT être {@code > 0} (typique chess AlphaZero : 0.3).
 * @param dirichletEpsilon (v1.1.0) mix factor entre priors NN et noise Dirichlet (cf. ADR-012). 0.0
 *     désactive Dirichlet (comportement v1.0.0 strictement préservé). Défaut 0.0. DOIT être dans
 *     {@code [0, 1]} (typique chess AlphaZero : 0.25).
 * @param randomSeed (v1.1.0) seed du {@code Random} utilisé pour le sampling Dirichlet (cf.
 *     ADR-012). Défaut 0L. Pour reproductibilité bit-à-bit fixer une valeur connue ; pour self-play
 *     diversifié, passer {@code System.nanoTime()} ou un compteur externe.
 * @param searchThreads (v1.2.0) nombre de threads de recherche concurrents (cf. ADR-013, ADR-015).
 *     Défaut 1 (mono-thread, court-circuite l'orchestrateur batched et retombe sur le code path
 *     v1.1.2 strict bit-pour-bit). {@code ≥ 2} active le mode batched. DOIT être dans {@code [1,
 *     128]}.
 * @param batchSize (v1.2.0) nombre de feuilles MCTS collectées avant chaque {@code
 *     Network.forward(planes, K, output)} (cf. ADR-013). Défaut 1 (batch=1 systématique,
 *     comportement v1.1.2 strict). DOIT être dans {@code [1, Network.MAX_BATCH]}.
 * @param virtualLoss (v1.2.0) pénalité Q appliquée aux nœuds in-flight pour décourager les threads
 *     concurrents de converger sur la même feuille (style KataGo, cf. ADR-013). Défaut 0.0
 *     (désactivé, équivalent mono-thread). Valeurs typiques pour multi-thread : 3.0 (recommandé,
 *     KataGo) ou 1.0 (Lc0 default). DOIT être dans {@code [0, 10]} et fini.
 * @param nnCacheSize (ADR-018) taille (nombre d'entrées) du cache d'évaluation NN OPTIONNEL. Défaut
 *     0 ⇒ cache désactivé : comportement strictement préservé bit-pour-bit (aucun calcul de clé,
 *     aucun overhead). {@code > 0} active un cache borné style Lc0 {@code NNCache} (mutualise
 *     l'évaluation NN d'une feuille, PAS les stats MCTS — l'arbre reste un arbre, cf. ADR-008/009).
 *     DOIT être {@code ≥ 0}.
 */
public record EngineConfig(
    float cPuct,
    float fpuValue,
    int treeInitialCapacity,
    float dirichletAlpha,
    float dirichletEpsilon,
    long randomSeed,
    int searchThreads,
    int batchSize,
    float virtualLoss,
    int nnCacheSize) {

  /** Maximum supporté pour {@link #searchThreads}, sanity cap. */
  public static final int MAX_SEARCH_THREADS = 128;

  /** Maximum supporté pour {@link #virtualLoss}, sanity cap. */
  public static final float MAX_VIRTUAL_LOSS = 10.0f;

  /**
   * Configuration par défaut conforme aux choix v1.0.0 + v1.1.0 + v1.2.0 : {@code cPuct=2.5},
   * {@code fpuValue=0.0}, {@code treeInitialCapacity=1024}, Dirichlet désactivé ({@code
   * dirichletAlpha=0.0, dirichletEpsilon=0.0, randomSeed=0L}), mode batched désactivé ({@code
   * searchThreads=1, batchSize=1, virtualLoss=0.0}) — comportement v1.1.2 strictement préservé.
   */
  public static EngineConfig defaults() {
    return new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, 0);
  }

  /**
   * Validation des arguments au compact constructor.
   *
   * @throws IllegalArgumentException si une contrainte est violée (cf. Javadoc de chaque champ)
   */
  public EngineConfig {
    // v1.1.2 — Float.isFinite() checks AVANT les bornes : NaN comparisons retournent toujours
    // false, donc NaN passerait silencieusement les checks < 0 ou > 1. isFinite rejette
    // NaN, +∞, -∞ en un seul test.
    if (!Float.isFinite(cPuct)) {
      throw new IllegalArgumentException("cPuct must be finite, got " + cPuct);
    }
    if (cPuct <= 0f) {
      throw new IllegalArgumentException("cPuct must be > 0, got " + cPuct);
    }
    if (!Float.isFinite(fpuValue)) {
      throw new IllegalArgumentException("fpuValue must be finite, got " + fpuValue);
    }
    if (fpuValue < -1f || fpuValue > 1f) {
      throw new IllegalArgumentException("fpuValue must be in [-1, 1], got " + fpuValue);
    }
    if (treeInitialCapacity < 1) {
      throw new IllegalArgumentException(
          "treeInitialCapacity must be >= 1, got " + treeInitialCapacity);
    }
    // v1.1.0 — validations Dirichlet, étendues v1.1.2 avec isFinite
    if (!Float.isFinite(dirichletAlpha)) {
      throw new IllegalArgumentException("dirichletAlpha must be finite, got " + dirichletAlpha);
    }
    if (dirichletAlpha < 0f) {
      throw new IllegalArgumentException("dirichletAlpha must be >= 0, got " + dirichletAlpha);
    }
    if (!Float.isFinite(dirichletEpsilon)) {
      throw new IllegalArgumentException(
          "dirichletEpsilon must be finite, got " + dirichletEpsilon);
    }
    if (dirichletEpsilon < 0f || dirichletEpsilon > 1f) {
      throw new IllegalArgumentException(
          "dirichletEpsilon must be in [0, 1], got " + dirichletEpsilon);
    }
    if (dirichletEpsilon > 0f && dirichletAlpha == 0f) {
      throw new IllegalArgumentException("dirichletAlpha must be > 0 when dirichletEpsilon > 0");
    }
    // v1.2.0 — validations mode batched
    if (searchThreads < 1 || searchThreads > MAX_SEARCH_THREADS) {
      throw new IllegalArgumentException(
          "searchThreads must be in [1, " + MAX_SEARCH_THREADS + "], got " + searchThreads);
    }
    if (batchSize < 1 || batchSize > Network.MAX_BATCH) {
      throw new IllegalArgumentException(
          "batchSize must be in [1, " + Network.MAX_BATCH + "], got " + batchSize);
    }
    if (!Float.isFinite(virtualLoss)) {
      throw new IllegalArgumentException("virtualLoss must be finite, got " + virtualLoss);
    }
    if (virtualLoss < 0f || virtualLoss > MAX_VIRTUAL_LOSS) {
      throw new IllegalArgumentException(
          "virtualLoss must be in [0, " + MAX_VIRTUAL_LOSS + "], got " + virtualLoss);
    }
    // ADR-018 — validation nnCacheSize.
    if (nnCacheSize < 0) {
      throw new IllegalArgumentException("nnCacheSize must be >= 0, got " + nnCacheSize);
    }
  }

  /**
   * (v1.1.2) Constructor secondaire à 3 args délégant. Équivalent à {@link #defaults()} avec
   * valeurs personnalisées sur les 3 premiers champs ; Dirichlet désactivé ({@code
   * dirichletAlpha=0.0}, {@code dirichletEpsilon=0.0}, {@code randomSeed=0L}) et mode batched
   * désactivé ({@code searchThreads=1, batchSize=1, virtualLoss=0.0f}). Restaure la compatibilité
   * source avec les callers v1.0.0 utilisant le constructor positionnel.
   *
   * @param cPuct constante d'exploration PUCT (doit être {@code > 0} et fini)
   * @param fpuValue valeur Q par défaut pour child non-visité (doit être dans {@code [-1, +1]} et
   *     fini)
   * @param treeInitialCapacity hint dimensionnement initial ({@code ≥ 1})
   * @throws IllegalArgumentException si une contrainte est violée (via délégation au compact
   *     constructor)
   */
  public EngineConfig(float cPuct, float fpuValue, int treeInitialCapacity) {
    this(cPuct, fpuValue, treeInitialCapacity, 0.0f, 0.0f, 0L, 1, 1, 0.0f);
  }

  /**
   * (v1.2.0) Constructor secondaire à 6 args délégant. Équivalent au constructor positionnel v1.1.0
   * avec mode batched désactivé ({@code searchThreads=1, batchSize=1, virtualLoss=0.0f}). Restaure
   * la compatibilité source avec les callers v1.1.0/v1.1.2 utilisant les 6 premiers champs.
   *
   * @param cPuct constante d'exploration PUCT
   * @param fpuValue First Play Urgency
   * @param treeInitialCapacity hint dimensionnement initial
   * @param dirichletAlpha paramètre α Dirichlet
   * @param dirichletEpsilon mix factor priors/noise
   * @param randomSeed seed du Random Dirichlet
   * @throws IllegalArgumentException si une contrainte est violée (via délégation au compact
   *     constructor)
   */
  public EngineConfig(
      float cPuct,
      float fpuValue,
      int treeInitialCapacity,
      float dirichletAlpha,
      float dirichletEpsilon,
      long randomSeed) {
    this(
        cPuct,
        fpuValue,
        treeInitialCapacity,
        dirichletAlpha,
        dirichletEpsilon,
        randomSeed,
        1,
        1,
        0.0f);
  }

  /**
   * (ADR-018) Constructor secondaire à 9 args délégant. Équivalent au constructor positionnel
   * v1.2.0 (avant l'ajout de {@code nnCacheSize}) avec cache d'évaluation NN désactivé ({@code
   * nnCacheSize=0}). Restaure la compatibilité source <strong>bit-pour-bit</strong> avec TOUS les
   * call sites v1.2.0/v1.3.0 utilisant les 9 premiers champs (cache off) ⇒ aucun site existant ni
   * {@link #defaults()} ne change de sémantique.
   *
   * @param cPuct constante d'exploration PUCT
   * @param fpuValue First Play Urgency
   * @param treeInitialCapacity hint dimensionnement initial
   * @param dirichletAlpha paramètre α Dirichlet
   * @param dirichletEpsilon mix factor priors/noise
   * @param randomSeed seed du Random Dirichlet
   * @param searchThreads nombre de threads de recherche
   * @param batchSize taille de batch NN
   * @param virtualLoss pénalité Q in-flight
   * @throws IllegalArgumentException si une contrainte est violée (via délégation au compact
   *     constructor)
   */
  public EngineConfig(
      float cPuct,
      float fpuValue,
      int treeInitialCapacity,
      float dirichletAlpha,
      float dirichletEpsilon,
      long randomSeed,
      int searchThreads,
      int batchSize,
      float virtualLoss) {
    this(
        cPuct,
        fpuValue,
        treeInitialCapacity,
        dirichletAlpha,
        dirichletEpsilon,
        randomSeed,
        searchThreads,
        batchSize,
        virtualLoss,
        0);
  }
}
