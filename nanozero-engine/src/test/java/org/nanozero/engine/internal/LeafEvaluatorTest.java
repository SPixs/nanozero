package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.Color;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;

/**
 * Tests {@link LeafEvaluator} (cf. SPEC §5.3, §12 phase 4).
 *
 * <p>Critère §12 phase 4 : sur 100 positions diverses + un mock retournant priors uniformes et
 * value=0, {@code LeafEvaluator} retourne des priors normalisés (sum=1) et value=0.
 *
 * <p>Le mock {@link MockNetworkProvider} est self-contained dans le test (pas de Mockito) :
 * politiques injectables (uniforme, focus, custom random), value injectable.
 */
class LeafEvaluatorTest {

  // -------------------------------------------------------------------------------------------
  // Mock NetworkProvider self-contained
  // -------------------------------------------------------------------------------------------

  /**
   * Implémentation de test du contrat {@link NetworkProvider}. Recopie un float[] de logits + une
   * value scalaire dans le {@code NNOutput} fourni à chaque appel.
   */
  private static final class MockNetworkProvider implements NetworkProvider {

    /** Logits pour le sample 0 (longueur 4672, copiés vers output.logits[0..4672)). */
    final float[] logitsForSample0;

    /** Value pour le sample 0. */
    float value;

    /** Compteur d'appels — sanity sur batch=1 (un appel par evaluate). */
    int forwardCallCount = 0;

    /** Snapshot des planes lus au dernier appel — sanity sur la pré-allocation. */
    float[] lastPlanesSnapshot;

    MockNetworkProvider() {
      this.logitsForSample0 = new float[MoveEncoding.POLICY_INDICES];
      this.value = 0.0f;
    }

    @Override
    public void forward(float[] planes, NNOutput output) {
      forwardCallCount++;
      lastPlanesSnapshot = planes.clone(); // copie pour permettre l'assertion ultérieure
      // Copie des logits dans output via réflexion sur le champ package-private 'logits' :
      // NNOutput est dans nanozero-nn, ses champs ne sont pas accessibles depuis ici. On utilise
      // la mécanique exposée publiquement : NNOutput.logits est package-private dans nn, mais
      // on peut écrire via des méthodes publiques OU contourner via reflection (peu propre).
      // Choix retenu : passer par reflection sur le champ logits (un accès public en écriture
      // n'existe pas car NNOutput est conçu en lecture seule pour les callers).
      try {
        java.lang.reflect.Field logitsField = NNOutput.class.getDeclaredField("logits");
        java.lang.reflect.Field valuesField = NNOutput.class.getDeclaredField("values");
        logitsField.setAccessible(true);
        valuesField.setAccessible(true);
        float[] outLogits = (float[]) logitsField.get(output);
        float[] outValues = (float[]) valuesField.get(output);
        System.arraycopy(logitsForSample0, 0, outLogits, 0, MoveEncoding.POLICY_INDICES);
        outValues[0] = value;
      } catch (ReflectiveOperationException e) {
        throw new AssertionError("MockNetworkProvider reflection setup failed", e);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  /** Remplit les logits du mock avec une distribution uniforme (tous 0 → softmax uniforme). */
  private static void setUniformLogits(MockNetworkProvider mock) {
    java.util.Arrays.fill(mock.logitsForSample0, 0f);
  }

  /** Remplit les logits avec un focus sur un index unique (logit=100 sur cet index, 0 ailleurs). */
  private static void setFocusedLogits(MockNetworkProvider mock, int focusIndex) {
    java.util.Arrays.fill(mock.logitsForSample0, 0f);
    mock.logitsForSample0[focusIndex] = 100f;
  }

  /** Génère les coups légaux d'une position dans un buffer fraîchement alloué. */
  private static int[] legalMovesOf(GameState state, int[] outCount) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buf, 0);
    outCount[0] = n;
    return buf;
  }

  // -------------------------------------------------------------------------------------------
  // Tests fonctionnels (priors normalisés, value transmise)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Logits uniformes, value=0 → priors uniformes (1/n), value=0 (critère §12 phase 4)")
  void uniformLogitsYieldsUniformPriorsAndZeroValue() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    int n = count[0];
    assertThat(n).isEqualTo(20); // sanity startpos

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    mock.value = 0.0f;

    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[n];
    float value = evaluator.evaluate(state, legalMoves, n, priors);

    // Value transmise telle quelle.
    assertThat(value).isEqualTo(0.0f);
    // Priors uniformes 1/20 = 0.05 chacun, sum = 1 ± 1e-6.
    float expected = 1.0f / n;
    double sum = 0;
    for (int i = 0; i < n; i++) {
      assertThat(priors[i]).isCloseTo(expected, org.assertj.core.api.Assertions.within(1e-6f));
      sum += priors[i];
    }
    assertThat((float) sum).isCloseTo(1.0f, org.assertj.core.api.Assertions.within(1e-6f));
  }

  @Test
  @DisplayName("Focus sur un coup légal → prior ≈ 1 sur ce coup, ≈ 0 ailleurs")
  void focusedLogitConcentratesPriorOnSingleMove() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    int n = count[0];

    int focusedMove = legalMoves[3]; // 4e coup légal
    int focusedIndex = MoveEncoding.encode(focusedMove, Color.WHITE);

    MockNetworkProvider mock = new MockNetworkProvider();
    setFocusedLogits(mock, focusedIndex);
    mock.value = 0.42f;

    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[n];
    float value = evaluator.evaluate(state, legalMoves, n, priors);

    // Value transmise.
    assertThat(value).isEqualTo(0.42f);
    // Prior concentré sur le coup index 3.
    assertThat(priors[3]).isGreaterThan(0.999f);
    for (int i = 0; i < n; i++) {
      if (i != 3) {
        assertThat(priors[i]).isLessThan(0.001f);
      }
    }
    // Sum reste 1.
    double sum = 0;
    for (int i = 0; i < n; i++) {
      sum += priors[i];
    }
    assertThat((float) sum).isCloseTo(1.0f, org.assertj.core.api.Assertions.within(1e-6f));
  }

  @Test
  @DisplayName("Value scalaire transmise du mock à l'appelant (cas négatif inclus)")
  void valueIsTransmittedAsIs() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[count[0]];

    for (float v : new float[] {-1.0f, -0.7f, 0.0f, 0.42f, 0.99f}) {
      mock.value = v;
      float result = evaluator.evaluate(state, legalMoves, count[0], priors);
      assertThat(result).isEqualTo(v);
    }
  }

  @Test
  @DisplayName("Forward appelé exactement 1 fois par evaluate (batch=1 conforme ADR-011)")
  void forwardCalledOncePerEvaluate() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[count[0]];

    evaluator.evaluate(state, legalMoves, count[0], priors);
    assertThat(mock.forwardCallCount).isEqualTo(1);

    evaluator.evaluate(state, legalMoves, count[0], priors);
    assertThat(mock.forwardCallCount).isEqualTo(2);
  }

  @Test
  @DisplayName("planeBuffer pré-alloué : même array entre 2 appels (zéro alloc)")
  void planeBufferIsReusedAcrossCalls() {
    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    LeafEvaluator evaluator = new LeafEvaluator(mock);

    float[] reference = evaluator.planeBufferForTest();
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    float[] priors = new float[count[0]];

    evaluator.evaluate(state, legalMoves, count[0], priors);
    assertThat(evaluator.planeBufferForTest()).isSameAs(reference);

    evaluator.evaluate(state, legalMoves, count[0], priors);
    assertThat(evaluator.planeBufferForTest()).isSameAs(reference);
  }

  @Test
  @DisplayName("Black to move : decodePolicy reçoit sideToMove=BLACK (perspective inversée)")
  void blackToMoveAppliesPerspectiveInversion() {
    // Position après 1.e4 : noirs au trait, 20 coups légaux.
    GameState state = new GameState();
    state.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
    assertThat(state.currentPosition().sideToMove()).isEqualTo(Color.BLACK);

    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    int n = count[0];
    assertThat(n).isEqualTo(20);

    // Focus sur le 1er coup noir : on encode l'index avec sideToMove=BLACK.
    int focusedMove = legalMoves[0];
    int focusedIndex = MoveEncoding.encode(focusedMove, Color.BLACK);

    MockNetworkProvider mock = new MockNetworkProvider();
    setFocusedLogits(mock, focusedIndex);
    mock.value = -0.3f;

    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[n];
    float value = evaluator.evaluate(state, legalMoves, n, priors);

    // Si decodePolicy avait reçu Color.WHITE par erreur, l'index encodé serait différent
    // (perspective inversée XOR 56 absente) et le prior 0 ne serait PAS proche de 1.
    assertThat(priors[0]).isGreaterThan(0.999f);
    assertThat(value).isEqualTo(-0.3f);
  }

  // -------------------------------------------------------------------------------------------
  // Critère §12 phase 4 — 100 positions diverses
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Critère §12 phase 4 : 100 positions random play, mock priors uniformes value=0 → "
          + "sum priors == 1 et value == 0")
  void hundredPositionsUniformPriorsAndZeroValue() {
    Random walk = new Random(0xC0FFEE17L);
    int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    float[] priors = new float[MoveGen.RECOMMENDED_BUFFER_SIZE];

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    mock.value = 0.0f;
    LeafEvaluator evaluator = new LeafEvaluator(mock);

    GameState gs = new GameState();
    int collected = 0;
    int safety = 0;
    while (collected < 100) {
      if (++safety > 50_000) {
        throw new AssertionError("random walk : 50k pas sans 100 positions non-terminales");
      }
      int n = gs.generateMoves(legalMoves, 0);
      if (n == 0) {
        gs = new GameState();
        continue;
      }

      float value = evaluator.evaluate(gs, legalMoves, n, priors);
      assertThat(value).as("position %d : value", collected).isEqualTo(0.0f);

      double sum = 0;
      for (int i = 0; i < n; i++) {
        assertThat(priors[i])
            .as("position %d : prior[%d] >= 0", collected, i)
            .isGreaterThanOrEqualTo(0f);
        sum += priors[i];
      }
      assertThat((float) sum)
          .as("position %d : sum priors == 1 ± 1e-6 (n=%d)", collected, n)
          .isCloseTo(1.0f, org.assertj.core.api.Assertions.within(1e-6f));

      // Avance d'un coup random.
      gs.applyMove(legalMoves[walk.nextInt(n)]);
      collected++;
    }
    assertThat(collected).isEqualTo(100);
  }

  // -------------------------------------------------------------------------------------------
  // Validation arguments
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructor null provider → NPE")
  void constructorNullProviderRejected() {
    assertThatThrownBy(() -> new LeafEvaluator(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("networkProvider");
  }

  @Test
  @DisplayName("evaluate(null state) → NPE")
  void evaluateNullStateRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(null, new int[1], 1, new float[1]))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("state");
  }

  @Test
  @DisplayName("evaluate(*, null legalMoves, ...) → NPE")
  void evaluateNullLegalMovesRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), null, 1, new float[1]))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("legalMoves");
  }

  @Test
  @DisplayName("evaluate(*, *, *, null priorsDest) → NPE")
  void evaluateNullPriorsDestRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), new int[1], 1, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("priorsDest");
  }

  @Test
  @DisplayName("evaluate(numLegalMoves=0) → IAE explicite (terminal must be detected by caller)")
  void evaluateZeroLegalMovesRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), new int[1], 0, new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numLegalMoves")
        .hasMessageContaining("terminal");
  }

  @Test
  @DisplayName("evaluate(numLegalMoves=-1) → IAE")
  void evaluateNegativeNumLegalMovesRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), new int[1], -1, new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numLegalMoves");
  }

  @Test
  @DisplayName("evaluate(numLegalMoves > legalMoves.length) → IAE")
  void evaluateNumLegalMovesExceedsBufferRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), new int[5], 10, new float[10]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exceeds legalMoves buffer");
  }

  @Test
  @DisplayName("evaluate(priorsDest.length < numLegalMoves) → IAE")
  void evaluatePriorsDestTooSmallRejected() {
    LeafEvaluator e = new LeafEvaluator(new MockNetworkProvider());
    assertThatThrownBy(() -> e.evaluate(new GameState(), new int[5], 5, new float[3]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("priorsDest length");
  }

  // -------------------------------------------------------------------------------------------
  // ADR-018 — cache d'évaluation NN optionnel
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "ADR-018 cache : 2e evaluate de la même clé → MÊME value + priors, forward NON rappelé")
  void cacheHitReturnsSameValueAndPriorsAndSkipsForward() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    int n = count[0];

    MockNetworkProvider mock = new MockNetworkProvider();
    setFocusedLogits(mock, MoveEncoding.encode(legalMoves[2], Color.WHITE));
    mock.value = 0.33f;

    NNEvalCache cache = new NNEvalCache(1024);
    LeafEvaluator evaluator = new LeafEvaluator(mock, cache);

    float[] priors1 = new float[n];
    float v1 = evaluator.evaluate(state, legalMoves, n, priors1);
    assertThat(mock.forwardCallCount).as("1er evaluate : MISS → forward").isEqualTo(1);

    // Mute le réseau : un re-forward produirait une sortie DIFFÉRENTE. Le cache doit primer.
    setFocusedLogits(mock, MoveEncoding.encode(legalMoves[5], Color.WHITE));
    mock.value = -0.99f;

    float[] priors2 = new float[n];
    float v2 = evaluator.evaluate(state, legalMoves, n, priors2);

    assertThat(mock.forwardCallCount).as("2e evaluate : HIT → forward NON rappelé").isEqualTo(1);
    assertThat(v2).as("value servie depuis le cache (pas le réseau muté)").isEqualTo(v1);
    assertThat(priors2).as("priors servis depuis le cache").containsExactly(priors1);
    assertThat(cache.lookups()).isEqualTo(2);
    assertThat(cache.hits()).isEqualTo(1);
  }

  @Test
  @DisplayName("ADR-018 cache=null (nnCacheSize=0) : forward rappelé à chaque evaluate (inchangé)")
  void nullCacheBehavesLikeNoCache() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);
    int n = count[0];

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    mock.value = 0.1f;

    // cache == null : chemin v1.x strict (bit-pour-bit).
    LeafEvaluator evaluator = new LeafEvaluator(mock, null);

    float[] priors = new float[n];
    float v1 = evaluator.evaluate(state, legalMoves, n, priors);
    assertThat(v1).isEqualTo(0.1f);

    // Sans cache, un changement de réseau est reflété ET le forward est rappelé.
    mock.value = -0.4f;
    float v2 = evaluator.evaluate(state, legalMoves, n, priors);
    assertThat(mock.forwardCallCount).as("pas de cache → forward chaque fois").isEqualTo(2);
    assertThat(v2).isEqualTo(-0.4f);
  }

  @Test
  @DisplayName("ADR-018 cache : positions distinctes → MISS distincts (pas de faux partage)")
  void cacheDistinctPositionsDoNotShare() {
    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    mock.value = 0.0f;
    NNEvalCache cache = new NNEvalCache(1024);
    LeafEvaluator evaluator = new LeafEvaluator(mock, cache);

    GameState startpos = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(startpos, count);
    int n = count[0];
    float[] priors = new float[n];

    evaluator.evaluate(startpos, legalMoves, n, priors);
    assertThat(mock.forwardCallCount).isEqualTo(1);

    // Position après 1.e4 : clé différente → MISS → forward rappelé.
    GameState afterE4 = new GameState();
    afterE4.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
    int[] legalMoves2 = legalMovesOf(afterE4, count);
    int n2 = count[0];
    float[] priors2 = new float[n2];
    evaluator.evaluate(afterE4, legalMoves2, n2, priors2);

    assertThat(mock.forwardCallCount).as("position distincte → MISS → forward").isEqualTo(2);
    assertThat(cache.hits()).isZero();
  }

  // -------------------------------------------------------------------------------------------
  // Sanity zéro alloc (informatif)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("100 evaluate sur mêmes buffers : pas de crash, pas d'OOM")
  void hundredCallsNoAllocSanity() {
    GameState state = new GameState();
    int[] count = new int[1];
    int[] legalMoves = legalMovesOf(state, count);

    MockNetworkProvider mock = new MockNetworkProvider();
    setUniformLogits(mock);
    LeafEvaluator evaluator = new LeafEvaluator(mock);
    float[] priors = new float[count[0]];

    for (int i = 0; i < 100; i++) {
      evaluator.evaluate(state, legalMoves, count[0], priors);
    }
    assertThat(mock.forwardCallCount).isEqualTo(100);
  }
}
