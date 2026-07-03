package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;

/**
 * Tests {@link NNEvalCache} (cf. ADR-018). Couvre : store/lookup, garde anti-collision n-mismatch,
 * éviction direct-mapped à capacité, déterminisme + transposition de la clé, sentinel NaN,
 * instrumentation lookups/hits, et validations.
 */
class NNEvalCacheTest {

  // -------------------------------------------------------------------------------------------
  // store / lookup
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("store puis lookup même clé/n → HIT : value + priors restitués")
  void storeThenLookupRoundtrip() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {0.1f, 0.2f, 0.7f};
    cache.store(0xCAFEBABEL, 0.42f, priors, 3);

    float[] dest = new float[3];
    float value = cache.lookup(0xCAFEBABEL, 3, dest);

    assertThat(Float.isNaN(value)).as("HIT → value finie").isFalse();
    assertThat(value).isEqualTo(0.42f);
    assertThat(dest).containsExactly(0.1f, 0.2f, 0.7f);
  }

  @Test
  @DisplayName("lookup clé absente → MISS (NaN) et priorsDest intact")
  void lookupMissReturnsNaNAndLeavesDestUntouched() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] dest = {-1f, -1f, -1f};
    float value = cache.lookup(123L, 3, dest);

    assertThat(Float.isNaN(value)).as("MISS → NaN sentinel").isTrue();
    assertThat(dest).as("MISS ne touche pas priorsDest").containsExactly(-1f, -1f, -1f);
  }

  @Test
  @DisplayName("store fait une copie défensive des priors (mutation source post-store sans effet)")
  void storeCopiesPriorsDefensively() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {0.25f, 0.75f};
    cache.store(7L, 0.0f, priors, 2);

    // Mutation de la source APRÈS le store : le cache ne doit pas être affecté.
    priors[0] = 9.9f;
    priors[1] = -9.9f;

    float[] dest = new float[2];
    cache.lookup(7L, 2, dest);
    assertThat(dest).containsExactly(0.25f, 0.75f);
  }

  @Test
  @DisplayName("store n < priors.length : seuls les n premiers priors sont mémorisés/restitués")
  void storeRespectsNLength() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {0.5f, 0.5f, 999f}; // priors[2] hors plage n=2
    cache.store(11L, 0.1f, priors, 2);

    float[] dest = new float[2];
    cache.lookup(11L, 2, dest);
    assertThat(dest).containsExactly(0.5f, 0.5f);
  }

  // -------------------------------------------------------------------------------------------
  // garde anti-collision : n mismatch → MISS
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("garde anti-collision : entrée présente mais n != numLegalMoves → MISS")
  void nMismatchTreatedAsMiss() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {0.1f, 0.2f, 0.7f};
    cache.store(55L, 0.3f, priors, 3);

    // Même clé, mais n différent (simule une collision Zobrist sur une grille différente).
    float[] dest = new float[2];
    float value = cache.lookup(55L, 2, dest);
    assertThat(Float.isNaN(value)).as("n mismatch → MISS").isTrue();

    // n correct → toujours HIT.
    float[] dest3 = new float[3];
    assertThat(cache.lookup(55L, 3, dest3)).isEqualTo(0.3f);
  }

  // -------------------------------------------------------------------------------------------
  // éviction direct-mapped à capacité
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("éviction direct-mapped : une collision de slot écrase l'ancienne entrée")
  void evictionOnSlotCollision() {
    // Capacité 2 (puissance de deux) → slotMask=1. Les clés 0 et 2 visent le même slot (k & 1 ==
    // 0).
    NNEvalCache cache = new NNEvalCache(2);
    assertThat(cache.capacity()).isEqualTo(2);

    float[] p1 = {1.0f};
    cache.store(0L, 0.5f, p1, 1);
    float[] dest = new float[1];
    assertThat(cache.lookup(0L, 1, dest)).as("clé 0 présente").isEqualTo(0.5f);

    // Stocke la clé 2 (même slot) → évince la clé 0.
    float[] p2 = {2.0f};
    cache.store(2L, -0.5f, p2, 1);

    assertThat(Float.isNaN(cache.lookup(0L, 1, dest))).as("clé 0 évincée").isTrue();
    assertThat(cache.lookup(2L, 1, dest)).as("clé 2 présente").isEqualTo(-0.5f);

    // La clé 1 (slot 1) cohabite sans collision.
    cache.store(1L, 0.9f, p1, 1);
    assertThat(cache.lookup(1L, 1, dest)).isEqualTo(0.9f);
    assertThat(cache.lookup(2L, 1, dest)).as("clé 2 toujours là (slot distinct)").isEqualTo(-0.5f);
  }

  @Test
  @DisplayName("capacité arrondie à la puissance de deux ≥ taille demandée")
  void capacityRoundedUpToPowerOfTwo() {
    assertThat(new NNEvalCache(1).capacity()).isEqualTo(1);
    assertThat(new NNEvalCache(3).capacity()).isEqualTo(4);
    assertThat(new NNEvalCache(200_000).capacity()).isEqualTo(262_144); // 2^18
  }

  // -------------------------------------------------------------------------------------------
  // sentinel NaN : value NaN jamais stockée
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("store value=NaN : ignoré (NaN = sentinel de MISS)")
  void storeNaNValueIgnored() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {1.0f};
    cache.store(99L, Float.NaN, priors, 1);

    float[] dest = new float[1];
    assertThat(Float.isNaN(cache.lookup(99L, 1, dest))).as("NaN jamais stockée → MISS").isTrue();
  }

  // -------------------------------------------------------------------------------------------
  // instrumentation lookups / hits
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("compteurs lookups/hits comptent tous les lookups et les HIT")
  void lookupsAndHitsCounters() {
    NNEvalCache cache = new NNEvalCache(16);
    float[] priors = {0.5f, 0.5f};
    cache.store(1L, 0.1f, priors, 2);

    float[] dest = new float[2];
    cache.lookup(1L, 2, dest); // HIT
    cache.lookup(2L, 2, dest); // MISS (clé absente)
    cache.lookup(1L, 2, dest); // HIT
    cache.lookup(1L, 3, dest); // MISS (n mismatch)

    assertThat(cache.lookups()).isEqualTo(4);
    assertThat(cache.hits()).isEqualTo(2);
  }

  // -------------------------------------------------------------------------------------------
  // clé : déterminisme + transposition + sensibilité
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("key(state) déterministe : deux appels sur le même état → clé identique")
  void keyIsDeterministic() {
    GameState state = new GameState();
    long k1 = NNEvalCache.key(state);
    long k2 = NNEvalCache.key(state);
    assertThat(k1).isEqualTo(k2);

    // Deux startpos distincts → même clé.
    assertThat(NNEvalCache.key(new GameState())).isEqualTo(k1);
  }

  @Test
  @DisplayName("key(state) change après un coup (entropie du Zobrist préservée)")
  void keyChangesAfterMove() {
    GameState state = new GameState();
    long before = NNEvalCache.key(state);
    state.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
    long after = NNEvalCache.key(state);
    assertThat(after).isNotEqualTo(before);
  }

  @Test
  @DisplayName("key(state) : transposition (même grille, même rule50) → clé identique")
  void keyEqualForTransposition() {
    // Ordre A : Nf3 Nc6 Nc3 Nf6 — uniquement des coups de cavalier (rule50 = 4 à la fin).
    GameState a = new GameState();
    a.applyMove(Move.encode(Square.G1, Square.F3, MoveType.NORMAL, 0));
    a.applyMove(Move.encode(Square.B8, Square.C6, MoveType.NORMAL, 0));
    a.applyMove(Move.encode(Square.B1, Square.C3, MoveType.NORMAL, 0));
    a.applyMove(Move.encode(Square.G8, Square.F6, MoveType.NORMAL, 0));

    // Ordre B : Nc3 Nf6 Nf3 Nc6 — même position finale, même rule50, sans répétition.
    GameState b = new GameState();
    b.applyMove(Move.encode(Square.B1, Square.C3, MoveType.NORMAL, 0));
    b.applyMove(Move.encode(Square.G8, Square.F6, MoveType.NORMAL, 0));
    b.applyMove(Move.encode(Square.G1, Square.F3, MoveType.NORMAL, 0));
    b.applyMove(Move.encode(Square.B8, Square.C6, MoveType.NORMAL, 0));

    assertThat(a.currentPosition().zobristHash())
        .as("sanity : transposition ⇒ Zobrist identiques")
        .isEqualTo(b.currentPosition().zobristHash());
    assertThat(NNEvalCache.key(a)).isEqualTo(NNEvalCache.key(b));
  }

  @Test
  @DisplayName("key : la clé HIT restitue exactement les priors stockés sous cette clé")
  void keyDrivesCorrectRefill() {
    NNEvalCache cache = new NNEvalCache(1024);
    GameState state = new GameState();
    long key = NNEvalCache.key(state);

    float[] priors = {0.05f, 0.10f, 0.85f};
    cache.store(key, -0.2f, priors, 3);

    float[] dest = new float[3];
    float value = cache.lookup(NNEvalCache.key(new GameState()), 3, dest);
    assertThat(value).isCloseTo(-0.2f, within(1e-7f));
    assertThat(dest).containsExactly(0.05f, 0.10f, 0.85f);
  }

  // -------------------------------------------------------------------------------------------
  // validations
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("constructeur taille < 1 → IllegalArgumentException")
  void constructorRejectsNonPositiveSize() {
    assertThatThrownBy(() -> new NNEvalCache(0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nnCacheSize");
    assertThatThrownBy(() -> new NNEvalCache(-1)).isInstanceOf(IllegalArgumentException.class);
  }
}
