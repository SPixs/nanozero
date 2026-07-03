package org.nanozero.sprt.stats;

import static org.assertj.core.api.Assertions.assertThat;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests référence : validation des LLR contre les cas exacts du test suite fastchess ({@code
 * app/tests/sprt_test.cpp}).
 *
 * <p>Source : Disservin/fastchess v1.8.0-alpha, fichier {@code app/tests/sprt_test.cpp}.
 *
 * <p>Tolerance : 0.01 (idem doctest epsilon dans fastchess test). Notre implémentation devrait
 * matcher à 1e-6 ou mieux en pratique, mais on aligne le tolerance sur fastchess pour cohérence.
 */
class SprtStatisticsReferenceTest {

  private static final Offset<Double> TOL = Offset.offset(0.01);

  // ============================ NORMALIZED (default fastchess) ============================

  // Note : fastchess Stats(w, l, d) prend args dans l'ordre (wins, LOSSES, draws).
  // Notre helper build(bounds, w, d, l) prend (wins, draws, losses) — ordre naturel.
  // Donc on swap les 2 derniers args quand on transcrit depuis le test fastchess.

  @Test
  @DisplayName("normalized trinomial 1: balanced near elo1=2")
  void normalizedTrinomial1() {
    // fastchess: Stats(w=36433, l=36027, d=68692)
    // SPRT(0.05, 0.05, 0, 2, "normalized") → LLR ≈ 0.92
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 2.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 36433, 68692, 36027);
    assertThat(stats.llr()).isCloseTo(0.92, TOL);
  }

  @Test
  @DisplayName("normalized trinomial 2: elo0 negative")
  void normalizedTrinomial2() {
    // fastchess: Stats(w=10871, l=10650, d=20431)
    // SPRT(0.05, 0.05, -1.75, 0.25, "normalized") → LLR ≈ 2.30
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, -1.75, 0.25, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 10871, 20431, 10650);
    assertThat(stats.llr()).isCloseTo(2.30, TOL);
  }

  @Test
  @DisplayName("normalized trinomial 3: only wins (extreme case)")
  void normalizedTrinomial3() {
    // fastchess: Stats(w=4250, l=0, d=0) — only wins
    // SPRT(0.05, 0.05, 0, 10, "normalized") → LLR ≈ 120.56
    // Tolerance plus large ici : cas extrême (régularisation 1e-3 sur les compteurs zéro)
    // amplifie les petites différences numériques du solver MLE iteratif (mle_epsilon=1e-4).
    // fastchess utilise doctest::Approx(120.56).epsilon(0.01) qui est une tolérance RELATIVE
    // 1% (1.2056 absolu), pas 0.01 absolu. Notre 0.012 absolu reste largement dans cette
    // marge (10× plus précis que requis).
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 10.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 4250, 0, 0);
    assertThat(stats.llr()).isCloseTo(120.56, org.assertj.core.data.Percentage.withPercentage(1.0));
  }

  // ============================ LOGISTIC ============================

  @Test
  @DisplayName("logistic trinomial 1: balanced, elo0=0.5 elo1=2.5")
  void logisticTrinomial1() {
    // fastchess: Stats(w=21404, l=21184, d=40708)
    // SPRT(0.05, 0.05, 0.5, 2.5, "logistic") → LLR ≈ -1.57
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.5, 2.5, SprtModel.LOGISTIC);
    SprtStatistics stats = build(bounds, 21404, 40708, 21184);
    assertThat(stats.llr()).isCloseTo(-1.57, TOL);
  }

  @Test
  @DisplayName("logistic trinomial 2: balanced, elo0=0 elo1=2")
  void logisticTrinomial2() {
    // fastchess: Stats(w=57433, l=57030, d=106593)
    // SPRT(0.05, 0.05, 0, 2, "logistic") → LLR ≈ -2.59
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 2.0, SprtModel.LOGISTIC);
    SprtStatistics stats = build(bounds, 57433, 106593, 57030);
    assertThat(stats.llr()).isCloseTo(-2.59, TOL);
  }

  // ============================ BAYESIAN ============================

  @Test
  @DisplayName("bayesian trinomial 1: balanced, elo0=0 elo1=2")
  void bayesianTrinomial1() {
    // fastchess: Stats(w=68965, l=68526, d=128429)
    // SPRT(0.05, 0.05, 0, 2, "bayesian") → LLR ≈ -1.26
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 2.0, SprtModel.BAYESIAN);
    SprtStatistics stats = build(bounds, 68965, 128429, 68526);
    assertThat(stats.llr()).isCloseTo(-1.26, TOL);
  }

  @Test
  @DisplayName("bayesian trinomial 2: elo0=0.5 elo1=2.5")
  void bayesianTrinomial2() {
    // fastchess: Stats(w=21629, l=21484, d=41111)
    // SPRT(0.05, 0.05, 0.5, 2.5, "bayesian") → LLR ≈ -1.13
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.5, 2.5, SprtModel.BAYESIAN);
    SprtStatistics stats = build(bounds, 21629, 41111, 21484);
    assertThat(stats.llr()).isCloseTo(-1.13, TOL);
  }

  // ============================ Decisions ============================

  @Test
  @DisplayName("decision H1 if llr ≥ upper")
  void decisionH1Accepted() {
    // Cas extreme : 1000 wins, 0 losses, 0 draws → LLR très haut → H1 ACCEPTED
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 1000, 0, 0);
    assertThat(stats.decision()).isEqualTo(org.nanozero.sprt.SprtDecision.H1_ACCEPTED);
  }

  @Test
  @DisplayName("decision H0 if llr ≤ lower (all losses)")
  void decisionH0Accepted() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 0, 0, 1000);
    assertThat(stats.decision()).isEqualTo(org.nanozero.sprt.SprtDecision.H0_ACCEPTED);
  }

  @Test
  @DisplayName("decision CONTINUE if llr in (lower, upper)")
  void decisionContinue() {
    // ~ balanced → LLR proche 0 → CONTINUE
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 10, 10, 10);
    assertThat(stats.decision()).isEqualTo(org.nanozero.sprt.SprtDecision.CONTINUE);
  }

  // ============================ Edge cases ============================

  @Test
  @DisplayName("zero games → llr 0, CONTINUE")
  void zeroGames() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = new SprtStatistics(bounds);
    assertThat(stats.llr()).isEqualTo(0.0);
    assertThat(stats.decision()).isEqualTo(org.nanozero.sprt.SprtDecision.CONTINUE);
    assertThat(stats.gamesPlayed()).isEqualTo(0);
  }

  @Test
  @DisplayName("bayesian wins=0 → llr 0 (degenerate case)")
  void bayesianAllLossesAndDraws() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.BAYESIAN);
    SprtStatistics stats = build(bounds, 0, 50, 50);
    assertThat(stats.llr()).isEqualTo(0.0);
  }

  @Test
  @DisplayName("bayesian losses=0 → llr 0 (degenerate case)")
  void bayesianAllWinsAndDraws() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.BAYESIAN);
    SprtStatistics stats = build(bounds, 50, 50, 0);
    assertThat(stats.llr()).isEqualTo(0.0);
  }

  // ============================ Accessors ============================

  @Test
  @DisplayName("counters increment correctly via record()")
  void recordIncrementsCounters() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = new SprtStatistics(bounds);
    stats.record(GameOutcome.WIN);
    stats.record(GameOutcome.WIN);
    stats.record(GameOutcome.DRAW);
    stats.record(GameOutcome.LOSS);
    assertThat(stats.wins()).isEqualTo(2);
    assertThat(stats.draws()).isEqualTo(1);
    assertThat(stats.losses()).isEqualTo(1);
    assertThat(stats.gamesPlayed()).isEqualTo(4);
  }

  @Test
  @DisplayName("scoreRate computes (W + D/2) / total")
  void scoreRate() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 10, 20, 10);
    // (10 + 0.5 * 20) / 40 = 20 / 40 = 0.5
    assertThat(stats.scoreRate()).isCloseTo(0.5, Offset.offset(1e-9));
  }

  @Test
  @DisplayName("scoreRate = 0.5 for empty stats")
  void scoreRateEmpty() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = new SprtStatistics(bounds);
    assertThat(stats.scoreRate()).isEqualTo(0.5);
  }

  @Test
  @DisplayName("eloDiff = 0 for 50% score")
  void eloDiffZero() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    SprtStatistics stats = build(bounds, 25, 50, 25);
    // 50% → 0 Elo
    assertThat(stats.eloDiff()).isCloseTo(0.0, Offset.offset(1e-9));
  }

  @Test
  @DisplayName("eloDiff matches logistic formula")
  void eloDiffLogistic() {
    SprtBounds bounds = SprtBounds.of(0.05, 0.05, 0.0, 5.0, SprtModel.NORMALIZED);
    // gen-16 SPRT: 277W/80L/157D, score 69.16%, Elo +140.32
    SprtStatistics stats = build(bounds, 277, 157, 80);
    // -400 * log10(1/0.6916 - 1) ≈ 140.32
    assertThat(stats.eloDiff()).isCloseTo(140.32, Offset.offset(0.5));
  }

  // ============================ Helpers ============================

  /** Helper : build stats with given W/D/L counts. */
  private static SprtStatistics build(SprtBounds bounds, int wins, int draws, int losses) {
    SprtStatistics s = new SprtStatistics(bounds);
    for (int i = 0; i < wins; i++) s.record(GameOutcome.WIN);
    for (int i = 0; i < draws; i++) s.record(GameOutcome.DRAW);
    for (int i = 0; i < losses; i++) s.record(GameOutcome.LOSS);
    return s;
  }
}
