package org.nanozero.engine.internal;

import java.util.concurrent.atomic.LongAdder;
import org.nanozero.board.GameState;
import org.nanozero.board.Position;

/**
 * Cache d'évaluation NN OPTIONNEL, calqué sur la {@code NNCache} de Lc0 (cf. ADR-018).
 *
 * <p><strong>Ce que c'est</strong> : un cache d'<em>évaluation</em> pur. Pour une feuille MCTS
 * donnée, il mémorise le résultat du forward réseau ({@code value} scalaire + {@code priors[n]}
 * décodés sur les coups légaux), de sorte qu'une seconde visite d'une feuille « équivalente » évite
 * {@code toPlanes → forward → decodePolicy}.
 *
 * <p><strong>Ce que ce n'est PAS</strong> : ni une table de transposition, ni un DAG. On NE partage
 * JAMAIS les statistiques MCTS ({@code N}, {@code Q}) — l'arbre reste un arbre pur (ADR-008/009).
 * Lc0 lui-même refuse le partage de nœuds à cause de la dépendance du réseau à l'historique (cf.
 * ADR-018 §contexte).
 *
 * <p><strong>Clé (approximation assumée façon-Lc0, {@code CacheHistoryLength=0})</strong> : un mix
 * 64-bit de la clé via {@link #key(GameState)} = {@code HashCat(zobristHash, aux)} où {@code aux}
 * empaquette le {@code rule50} (halfmoveClock) et 2 bits de répétition (twofold/threefold). Le
 * Zobrist complet est <strong>préservé</strong> (pas de troncature de bits) ; seul le « cosmétique
 * » — le chemin des 7 derniers plateaux que le réseau consomme via les plans d'historique — est
 * délibérément ignoré. La partie de l'historique qui change la valeur du jeu (répétition + rule50)
 * reste dans la clé.
 *
 * <p><strong>Éviction</strong> : table direct-mapped bornée (taille = puissance de deux ≥ capacité
 * demandée). Une collision de slot écrase l'entrée présente (éviction « simple » au sens
 * d'ADR-018). Borné par construction, zéro boxing (clé primitive {@code long}), zéro allocation sur
 * HIT.
 *
 * <p><strong>Thread-safety</strong> : conçu thread-safe (verrous stripés par slot) pour permettre
 * le câblage mode B futur. Le SPRT initial est en mode A (mono-thread, {@code Threads=1}) où aucune
 * contention n'existe. Un HIT effectue uniquement un {@link System#arraycopy} sous le verrou de
 * stripe — aucune allocation.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module. La
 *     classe est {@code public} uniquement pour être instanciée par {@code Engine} (package parent)
 *     et passée aux {@code SearcherCore} ; son API ({@link #key}, {@link #lookup}, {@link #store},
 *     compteurs) reste package-private.
 */
public final class NNEvalCache {

  /**
   * Constante du nombre d'or pour le hash-combine (Lc0 {@code HashCat} / boost {@code
   * hash_combine}). Mélange add+shift+xor : préserve l'entropie des 64 bits du Zobrist (aucune
   * troncature).
   */
  private static final long HASH_MIX = 0x9E3779B97F4A7C15L;

  /** Plafond du nombre de verrous de stripe (puissance de deux). */
  private static final int MAX_LOCKS = 64;

  /**
   * Entrée immutable du cache : la value scalaire, le nombre de coups légaux {@code n} (garde
   * anti-collision) et la copie défensive des {@code n} priors. La clé est stockée pour valider le
   * slot direct-mapped (deux clés distinctes peuvent viser le même slot).
   */
  private static final class Entry {
    final long key;
    final float value;
    final int n;
    final float[] priors;

    Entry(long key, float value, int n, float[] priors) {
      this.key = key;
      this.value = value;
      this.n = n;
      this.priors = priors;
    }
  }

  /** Table direct-mapped (taille = {@link #capacity}, puissance de deux). */
  private final Entry[] slots;

  /** Masque d'indexation de slot ({@code capacity - 1}). */
  private final int slotMask;

  /** Verrous de stripe (puissance de deux ≤ {@link #MAX_LOCKS}). */
  private final Object[] locks;

  /** Masque d'indexation de verrou ({@code locks.length - 1}). */
  private final int lockMask;

  /** Capacité effective (puissance de deux ≥ taille demandée). */
  private final int capacity;

  private final LongAdder lookups = new LongAdder();
  private final LongAdder hits = new LongAdder();

  /**
   * Construit un cache borné à au moins {@code requestedSize} entrées. La capacité effective est
   * arrondie à la puissance de deux supérieure ou égale (indexation direct-mapped sans modulo).
   *
   * @param requestedSize nombre d'entrées souhaité (≥ 1)
   * @throws IllegalArgumentException si {@code requestedSize < 1}
   */
  public NNEvalCache(int requestedSize) {
    if (requestedSize < 1) {
      throw new IllegalArgumentException("nnCacheSize must be >= 1, got " + requestedSize);
    }
    this.capacity = nextPowerOfTwo(requestedSize);
    this.slotMask = capacity - 1;
    this.slots = new Entry[capacity];
    int numLocks = Math.min(capacity, MAX_LOCKS); // capacité et MAX_LOCKS sont des puissances de 2
    this.locks = new Object[numLocks];
    for (int i = 0; i < numLocks; i++) {
      this.locks[i] = new Object();
    }
    this.lockMask = numLocks - 1;
  }

  /**
   * Calcule la clé 64-bit d'une position de feuille MCTS (cf. ADR-018). {@code HashCat(zobristHash,
   * aux)} où {@code aux} empaquette {@code rule50} (bits 0..31) + répétition twofold (bit 32) +
   * threefold (bit 33). Le Zobrist complet est mélangé sans troncature.
   *
   * @param state position à clé (l'historique sert à {@link GameState#isRepetition(int)}), non null
   * @return clé 64-bit
   */
  static long key(GameState state) {
    Position pos = state.currentPosition();
    long zobrist = pos.zobristHash();
    long aux = pos.halfmoveClock() & 0xFFFFFFFFL;
    if (state.isRepetition(2)) {
      aux |= 1L << 32;
    }
    if (state.isRepetition(3)) {
      aux |= 1L << 33;
    }
    return hashCat(zobrist, aux);
  }

  /**
   * Hash-combine façon Lc0 {@code HashCat} de deux mots 64-bit (add+shift+xor avec la constante du
   * nombre d'or). Préserve l'entropie complète de {@code a} (le Zobrist) : départ {@code h=0} ⇒
   * première passe {@code h = HASH_MIX + a}, seconde passe mélange {@code b} avec les décalages de
   * {@code h}.
   */
  private static long hashCat(long a, long b) {
    long h = 0L;
    h ^= HASH_MIX + a + (h << 6) + (h >> 2);
    h ^= HASH_MIX + b + (h << 6) + (h >> 2);
    return h;
  }

  /**
   * Recherche une entrée. Sur HIT, remplit {@code priorsDest[0..numLegalMoves)} via {@link
   * System#arraycopy} (zéro allocation) et retourne la value mémorisée. Sur MISS, retourne {@link
   * Float#NaN} et ne touche pas {@code priorsDest}.
   *
   * <p>Une entrée n'est un HIT que si la clé correspond ET {@code entry.n == numLegalMoves} (garde
   * anti-collision Zobrist 64-bit : protège contre une collision où la grille différerait — cf.
   * ADR-018). {@link Float#NaN} est un sentinel sûr car une value NN est toujours finie dans {@code
   * [-1, +1]} (jamais stockée si NaN, cf. {@link #store}).
   *
   * @param key clé issue de {@link #key(GameState)}
   * @param numLegalMoves nombre de coups légaux attendu (longueur de refill)
   * @param priorsDest buffer destination ; rempli sur HIT, intact sur MISS
   * @return la value mémorisée sur HIT, {@link Float#NaN} sur MISS
   */
  float lookup(long key, int numLegalMoves, float[] priorsDest) {
    lookups.increment();
    int slot = (int) (key & slotMask);
    synchronized (locks[slot & lockMask]) {
      Entry e = slots[slot];
      if (e != null && e.key == key && e.n == numLegalMoves) {
        System.arraycopy(e.priors, 0, priorsDest, 0, numLegalMoves);
        hits.increment();
        return e.value;
      }
    }
    return Float.NaN;
  }

  /**
   * Mémorise (ou écrase, direct-mapped) l'évaluation d'une feuille. Copie défensive des {@code n}
   * premiers priors (allocation tolérée : chemin MISS uniquement, cf. ADR-018).
   *
   * <p>Une {@code value} {@link Float#isNaN(float) NaN} n'est jamais stockée (NaN est le sentinel
   * de MISS de {@link #lookup}).
   *
   * @param key clé issue de {@link #key(GameState)}
   * @param value value NN scalaire à mémoriser (finie)
   * @param priors buffer source des priors normalisés
   * @param n nombre de coups légaux (priors valides dans {@code [0, n)})
   */
  void store(long key, float value, float[] priors, int n) {
    if (Float.isNaN(value)) {
      return;
    }
    float[] copy = new float[n];
    System.arraycopy(priors, 0, copy, 0, n);
    Entry e = new Entry(key, value, n, copy);
    int slot = (int) (key & slotMask);
    synchronized (locks[slot & lockMask]) {
      slots[slot] = e;
    }
  }

  /** Nombre total d'appels à {@link #lookup} (HIT + MISS). Instrumentation hit-rate. */
  long lookups() {
    return lookups.sum();
  }

  /** Nombre d'appels à {@link #lookup} ayant produit un HIT. Instrumentation hit-rate. */
  long hits() {
    return hits.sum();
  }

  /** Capacité effective (puissance de deux ≥ taille demandée). */
  int capacity() {
    return capacity;
  }

  /** Arrondit à la puissance de deux ≥ {@code n} (n ≥ 1), plafonnée à {@code 1 << 30}. */
  private static int nextPowerOfTwo(int n) {
    if (n >= (1 << 30)) {
      return 1 << 30;
    }
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }
}
