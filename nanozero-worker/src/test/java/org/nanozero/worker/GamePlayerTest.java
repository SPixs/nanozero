package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.nanozero.board.Color;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.MoveEncoding;

/**
 * Tests for the pure-function parts of GamePlayer. The full {@link GamePlayer#play} requires a
 * Network + Engine, which is exercised end-to-end in 13.4b.3 against a real model.
 */
class GamePlayerTest {

  // ---------------------------------------------------------------------------------------
  // buildPolicyTarget
  // ---------------------------------------------------------------------------------------

  @Test
  void buildPolicyTargetSumsToOne() {
    // Use moves at valid positions ; encoding requires legal-ish moves but the test only checks
    // arithmetic shape, not move legality. Use simple e2-e4 + g1-f3 style encodings.
    int[] moves = sampleMoves();
    int[] visits = {30, 10};
    float[] policy = GamePlayer.buildPolicyTarget(moves, visits, Color.WHITE);

    float sum = 0;
    for (float v : policy) sum += v;
    assertThat(sum).isCloseTo(1.0f, org.assertj.core.api.Assertions.within(1e-5f));
    assertThat(policy).hasSize(MoveEncoding.POLICY_INDICES);
  }

  @Test
  void buildPolicyTargetZeroVisitsAllZero() {
    int[] moves = sampleMoves();
    int[] visits = {0, 0};
    float[] policy = GamePlayer.buildPolicyTarget(moves, visits, Color.WHITE);
    for (float v : policy) {
      assertThat(v).isEqualTo(0.0f);
    }
  }

  @Test
  void buildPolicyTargetNormalizesUniform() {
    int[] moves = sampleMoves();
    int[] visits = {5, 5};
    float[] policy = GamePlayer.buildPolicyTarget(moves, visits, Color.WHITE);
    int nonZeroCount = 0;
    for (float v : policy) {
      if (v > 0) {
        assertThat(v).isCloseTo(0.5f, org.assertj.core.api.Assertions.within(1e-6f));
        nonZeroCount++;
      }
    }
    assertThat(nonZeroCount).isEqualTo(2);
  }

  // ---------------------------------------------------------------------------------------
  // pickMove
  // ---------------------------------------------------------------------------------------

  @Test
  void pickMoveSingleChoiceReturnsThatMove() {
    int[] moves = {1234};
    int[] visits = {50};
    int chosen =
        GamePlayer.pickMove(moves, visits, /* ply */ 0, /* temp */ 1.0f, 30, new Random(0));
    assertThat(chosen).isEqualTo(1234);
  }

  @Test
  void pickMoveArgmaxAfterTemperatureSwitch() {
    int[] moves = {100, 200, 300};
    int[] visits = {10, 50, 1};
    // ply >= switchPly → argmax regardless of temperature.
    int chosen =
        GamePlayer.pickMove(moves, visits, /* ply */ 30, /* temp */ 1.0f, 30, new Random(42));
    assertThat(chosen).isEqualTo(200); // visits[1] = 50 is max
  }

  @Test
  void pickMoveArgmaxWhenTemperatureNearZero() {
    int[] moves = {100, 200, 300};
    int[] visits = {1, 50, 10};
    int chosen = GamePlayer.pickMove(moves, visits, 0, 0.001f, 30, new Random(0));
    assertThat(chosen).isEqualTo(200);
  }

  @Test
  void pickMoveStochasticBeforeSwitch() {
    // With T=1.0 and visits [50,50,0], the move "300" (visits 0) must never be sampled.
    int[] moves = {100, 200, 300};
    int[] visits = {50, 50, 0};
    Random rng = new Random(42);
    boolean saw100 = false, saw200 = false;
    for (int i = 0; i < 200; i++) {
      int chosen = GamePlayer.pickMove(moves, visits, 0, 1.0f, 30, rng);
      assertThat(chosen).isNotEqualTo(300); // zero-visit must never be picked
      if (chosen == 100) saw100 = true;
      if (chosen == 200) saw200 = true;
    }
    // Roughly 50/50 over 200 trials — both visited.
    assertThat(saw100).isTrue();
    assertThat(saw200).isTrue();
  }

  @Test
  void pickMoveStochasticAllZeroVisitsFallsBackToUniform() {
    // T=1.0, ply < switch, >1 move, ALL visits zero → weighted total is 0 → defensive uniform pick.
    int[] moves = {100, 200, 300};
    int[] visits = {0, 0, 0};
    Random rng = new Random(1);
    for (int i = 0; i < 50; i++) {
      int chosen = GamePlayer.pickMove(moves, visits, 0, 1.0f, 30, rng);
      assertThat(chosen).isIn(100, 200, 300);
    }
  }

  @Test
  void pickMoveEmptyArrayRaises() {
    assertThatThrownBy(() -> GamePlayer.pickMove(new int[0], new int[0], 0, 1.0f, 30, new Random()))
        .isInstanceOf(IllegalStateException.class);
  }

  // ---------------------------------------------------------------------------------------
  // backfillValueTargets
  // ---------------------------------------------------------------------------------------

  @Test
  void backfillValueWhiteWin() {
    List<Sample> samples = new ArrayList<>();
    samples.add(new Sample(new float[1], new float[1], Color.WHITE, 0));
    samples.add(new Sample(new float[1], new float[1], Color.BLACK, 1));
    samples.add(new Sample(new float[1], new float[1], Color.WHITE, 2));

    GamePlayer.backfillValueTargets(samples, +1.0f);

    assertThat(samples.get(0).valueTarget).isEqualTo(+1.0f); // white POV at white-turn ply
    assertThat(samples.get(1).valueTarget).isEqualTo(-1.0f); // black POV : white-won = bad
    assertThat(samples.get(2).valueTarget).isEqualTo(+1.0f);
  }

  @Test
  void backfillValueDraw() {
    List<Sample> samples = new ArrayList<>();
    samples.add(new Sample(new float[1], new float[1], Color.WHITE, 0));
    samples.add(new Sample(new float[1], new float[1], Color.BLACK, 1));

    GamePlayer.backfillValueTargets(samples, 0.0f);
    assertThat(samples).allMatch(s -> s.valueTarget == 0.0f);
  }

  @Test
  void backfillValueBlackWin() {
    List<Sample> samples = new ArrayList<>();
    samples.add(new Sample(new float[1], new float[1], Color.WHITE, 0));
    samples.add(new Sample(new float[1], new float[1], Color.BLACK, 1));

    GamePlayer.backfillValueTargets(samples, -1.0f);

    assertThat(samples.get(0).valueTarget).isEqualTo(-1.0f);
    assertThat(samples.get(1).valueTarget).isEqualTo(+1.0f); // black wins → black POV value = +1
  }

  // ---------------------------------------------------------------------------------------
  // play (end-to-end with FakeNetwork + real Engine)
  // ---------------------------------------------------------------------------------------

  @Test
  void playStopsAtMaxPliesAndBackfillsDrawValue() {
    // Cap at a small number of plies so the game aborts as a draw quickly (uniform-ish policy).
    GamePlayer player = new GamePlayer();
    List<Sample> samples =
        player.play(
            new FakeNetwork(),
            EngineConfig.defaults(),
            /* numSims */ 2,
            /* maxPlies */ 4,
            /* temperature */ 1.0f,
            /* temperatureSwitchPly */ 2,
            new Random(7));

    // Exactly maxPlies samples : the start position cannot be terminal within < 4 plies (fastest
    // mate is fool's mate at ply 4), so the loop always exits on the maxPlies cap here.
    assertThat(samples).hasSize(4);
    // Plies are 0..3, sides alternate WHITE, BLACK, WHITE, BLACK.
    for (int i = 0; i < samples.size(); i++) {
      Sample s = samples.get(i);
      assertThat(s.ply).isEqualTo(i);
      assertThat(s.turn).isEqualTo(i % 2 == 0 ? Color.WHITE : Color.BLACK);
      // Input planes + policy target are populated and base64-encodable.
      assertThat(s.inputPlanes).hasSize(NN_PLANE_FLOATS);
      assertThat(s.policyTarget).hasSize(MoveEncoding.POLICY_INDICES);
      // Value target backfilled (non-NaN) in {-1, 0, +1}.
      assertThat(Float.isNaN(s.valueTarget)).isFalse();
      // Math.abs normalise -0.0f -> 0.0f (un draw côté Noir donne -0.0f, que
      // Float.equals — utilisé par isIn — distingue à tort de 0.0f).
      assertThat(Math.abs(s.valueTarget)).isIn(0.0f, 1.0f);
      // Policy target sums to ~1 (visit distribution normalized).
      float sum = 0;
      for (float v : s.policyTarget) sum += v;
      assertThat(sum).isCloseTo(1.0f, org.assertj.core.api.Assertions.within(1e-4f));
    }
    // base64 round-trip is well-formed (covers Sample encoders via the play path too).
    assertThat(samples.get(0).inputPlanesBase64()).isNotEmpty();
    assertThat(samples.get(0).policyTargetBase64()).isNotEmpty();
  }

  @Test
  void playReachesNaturalTerminalAndBackfillsOutcome() {
    // A long random game terminates naturally (threefold repetition / 50-move / mate / stalemate)
    // well before this cap. Exercises the natural-terminal path of play() + gameOutcomeWhite's
    // switch (a draw, win-white, or win-black arm depending on the game).
    GamePlayer player = new GamePlayer();
    List<Sample> samples =
        player.play(
            new FakeNetwork(),
            EngineConfig.defaults(),
            /* numSims */ 1,
            /* maxPlies */ 1000,
            /* temperature */ 1.0f,
            /* temperatureSwitchPly */ 8,
            new Random(2026));

    // Terminated naturally (did not hit the max-plies cap).
    assertThat(samples.size()).isLessThan(1000);
    assertThat(samples).isNotEmpty();
    // Every sample got a backfilled (non-NaN) value target in {-1, 0, +1}.
    for (Sample s : samples) {
      assertThat(Float.isNaN(s.valueTarget)).isFalse();
      // Math.abs normalise -0.0f -> 0.0f (un draw côté Noir donne -0.0f, que
      // Float.equals — utilisé par isIn — distingue à tort de 0.0f).
      assertThat(Math.abs(s.valueTarget)).isIn(0.0f, 1.0f);
    }
    // White-turn and black-turn samples carry opposite-signed value targets (consistent POV).
    float whiteVal = Float.NaN;
    float blackVal = Float.NaN;
    for (Sample s : samples) {
      if (s.turn == Color.WHITE) whiteVal = s.valueTarget;
      else blackVal = s.valueTarget;
    }
    if (!Float.isNaN(whiteVal) && !Float.isNaN(blackVal)) {
      assertThat(whiteVal).isEqualTo(-blackVal);
    }
  }

  @Test
  void playEngagesTemperatureBeforeSwitchAndArgmaxAfter() {
    // switchPly=1 means ply 0 uses temperature sampling, plies >=1 use argmax. Just assert it runs
    // and produces the expected number of samples (covers both selection branches in one game).
    GamePlayer player = new GamePlayer();
    List<Sample> samples =
        player.play(
            new FakeNetwork(0.0f),
            EngineConfig.defaults(),
            2,
            3,
            1.0f,
            /* temperatureSwitchPly */ 1,
            new Random(99));
    assertThat(samples).hasSize(3);
  }

  @Test
  void playRejectsNonPositiveNumSims() {
    GamePlayer player = new GamePlayer();
    FakeNetwork net = new FakeNetwork();
    EngineConfig cfg = EngineConfig.defaults();
    Random rng = new Random();
    assertThatThrownBy(() -> player.play(net, cfg, 0, 10, 1.0f, 5, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numSims");
  }

  @Test
  void playRejectsNonPositiveMaxPlies() {
    GamePlayer player = new GamePlayer();
    FakeNetwork net = new FakeNetwork();
    EngineConfig cfg = EngineConfig.defaults();
    Random rng = new Random();
    assertThatThrownBy(() -> player.play(net, cfg, 2, 0, 1.0f, 5, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxPlies");
  }

  @Test
  void playRejectsNullRng() {
    GamePlayer player = new GamePlayer();
    FakeNetwork net = new FakeNetwork();
    EngineConfig cfg = EngineConfig.defaults();
    assertThatThrownBy(() -> player.play(net, cfg, 2, 10, 1.0f, 5, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rng");
  }

  // ---------------------------------------------------------------------------------------
  // perGameConfig : per-game Dirichlet re-seed must PRESERVE nnCacheSize (ADR-018)
  // ---------------------------------------------------------------------------------------

  @Test
  void perGameConfigPreservesNnCacheSizeWhenDirichletOn() {
    // Dirichlet on (epsilon > 0) → the config is rebuilt with a fresh per-game seed ; the optional
    // NN-eval cache size must survive that rebuild (it would otherwise silently drop to 0).
    EngineConfig base = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 0L, 1, 1, 0.0f, 131072);
    EngineConfig perGame = GamePlayer.perGameConfig(base, new Random(7));

    assertThat(perGame.nnCacheSize()).isEqualTo(131072);
    // Re-seeded (a fresh seed was drawn) but every other field is preserved.
    assertThat(perGame.randomSeed()).isNotEqualTo(base.randomSeed());
    assertThat(perGame.dirichletAlpha()).isEqualTo(base.dirichletAlpha());
    assertThat(perGame.dirichletEpsilon()).isEqualTo(base.dirichletEpsilon());
  }

  @Test
  void perGameConfigReturnsSameConfigWhenDirichletOff() {
    // Dirichlet off (epsilon == 0) → config returned untouched (no rng draw), cache preserved.
    EngineConfig base = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, 65536);
    EngineConfig perGame = GamePlayer.perGameConfig(base, new Random(7));

    assertThat(perGame).isSameAs(base);
    assertThat(perGame.nnCacheSize()).isEqualTo(65536);
  }

  // ---------------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------------

  private static final int NN_PLANE_FLOATS = 119 * 64;

  /**
   * Two legal-shape encoded moves for testing. e2e4 = from=12 to=28 (no promotion), encoded by the
   * {@code Move} 16-bit format. We use {@code MoveEncoding.encode} to ensure roundtrip consistency
   * in the index logic.
   */
  private static int[] sampleMoves() {
    // Simple Move encoding : low 6 = from, mid 6 = to. e2=12, e4=28 ; g1=6, f3=21.
    int e2e4 = (12) | (28 << 6);
    int g1f3 = (6) | (21 << 6);
    return new int[] {e2e4, g1f3};
  }
}
