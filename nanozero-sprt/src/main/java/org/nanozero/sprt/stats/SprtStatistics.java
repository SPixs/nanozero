package org.nanozero.sprt.stats;

import org.nanozero.sprt.SprtDecision;
import org.nanozero.sprt.internal.MleSolver;

/**
 * Accumulateur thread-safe pour le SPRT (Sequential Probability Ratio Test) trinomial.
 *
 * <p>Implémente {@code fastchess::SPRT::getLLR(int wins, int draws, int losses)} avec les 3 modèles
 * : {@link SprtModel#LOGISTIC}, {@link SprtModel#BAYESIAN}, {@link SprtModel#NORMALIZED}.
 *
 * <p>Usage standard :
 *
 * <pre>{@code
 * SprtBounds bounds = SprtBounds.standard(0.0, 5.0);  // elo0=0, elo1=5, alpha=beta=0.05
 * SprtStatistics stats = new SprtStatistics(bounds);
 * for (Game game : games) {
 *   stats.record(game.outcome());
 *   if (stats.decision() != SprtDecision.CONTINUE) break;
 * }
 * double llr = stats.llr();
 * }</pre>
 *
 * <p>Thread-safe : tous les accesseurs et {@link #record(GameOutcome)} sont {@code synchronized}.
 * Pour un tournament concurrent, plusieurs threads peuvent appeler {@code record} en parallèle sans
 * corruption.
 *
 * <p>Cf. SPEC-sprt §4.
 */
public final class SprtStatistics {

  /** Scores trinomial : LOSS=0.0, DRAW=0.5, WIN=1.0. */
  private static final double[] TRINOMIAL_SCORES = {0.0, 0.5, 1.0};

  /** Conversion Elo → score logistique : 800 / ln(10). */
  private static final double ELO_DIVISOR = 800.0 / Math.log(10.0);

  /**
   * Régularisation des compteurs zéro pour éviter log(0). Cf. fastchess sprt.cpp::regularize(). Une
   * valeur de 1e-3 est utilisée comme « pseudo-game ».
   */
  private static final double REGULARIZE_ZERO = 1e-3;

  private final SprtBounds bounds;
  private int wins;
  private int draws;
  private int losses;

  /**
   * Construit un accumulateur SPRT vide pour des bornes données.
   *
   * @param bounds bornes pré-calculées (cf. {@link SprtBounds#of}), non null
   * @throws NullPointerException si {@code bounds} null
   */
  public SprtStatistics(SprtBounds bounds) {
    if (bounds == null) {
      throw new NullPointerException("bounds must not be null");
    }
    this.bounds = bounds;
  }

  /**
   * Enregistre une game.
   *
   * @param outcome résultat de la game du POV du challenger, non null
   * @throws NullPointerException si {@code outcome} null
   */
  public synchronized void record(GameOutcome outcome) {
    switch (outcome) {
      case WIN -> wins++;
      case LOSS -> losses++;
      case DRAW -> draws++;
    }
  }

  /** Nombre total de games enregistrées. */
  public synchronized int gamesPlayed() {
    return wins + losses + draws;
  }

  /** Nombre de wins du challenger. */
  public synchronized int wins() {
    return wins;
  }

  /** Nombre de losses du challenger. */
  public synchronized int losses() {
    return losses;
  }

  /** Nombre de nulles. */
  public synchronized int draws() {
    return draws;
  }

  /** Bornes utilisées. */
  public SprtBounds bounds() {
    return bounds;
  }

  /**
   * Calcule le LLR (Log-Likelihood Ratio) cumulé actuel selon le modèle des bornes.
   *
   * <p>Port direct de {@code fastchess::SPRT::getLLR(int, int, int)}. Match bit-pour-bit à 1e-6
   * près sur les cas du Stockfish framework.
   *
   * <p>Cas dégénérés :
   *
   * <ul>
   *   <li>0 games : retourne 0.0
   *   <li>BAYESIAN avec wins=0 OU losses=0 : retourne 0.0 (cas non-identifiable, fastchess
   *       fallback)
   *   <li>autres compteurs à 0 : régularisés à 1e-3 (cf. {@link #REGULARIZE_ZERO})
   * </ul>
   *
   * @return LLR cumulé
   */
  public synchronized double llr() {
    int total = wins + losses + draws;
    if (total == 0) {
      return 0.0;
    }
    if (bounds.model() == SprtModel.BAYESIAN && (wins == 0 || losses == 0)) {
      return 0.0;
    }

    double l = regularize(losses);
    double d = regularize(draws);
    double w = regularize(wins);
    double totalReg = l + d + w;
    double[] probs = {l / totalReg, d / totalReg, w / totalReg};

    return switch (bounds.model()) {
      case LOGISTIC -> {
        double score0 = leloToScore(bounds.elo0());
        double score1 = leloToScore(bounds.elo1());
        yield llrLogistic(totalReg, TRINOMIAL_SCORES, probs, score0, score1);
      }
      case BAYESIAN -> {
        double pL = probs[0];
        double pW = probs[2];
        double drawElo = 200.0 * Math.log10((1.0 - pL) / pL * (1.0 - pW) / pW);
        double score0 = bayesEloToScore(bounds.elo0(), drawElo);
        double score1 = bayesEloToScore(bounds.elo1(), drawElo);
        yield llrLogistic(totalReg, TRINOMIAL_SCORES, probs, score0, score1);
      }
      case NORMALIZED -> {
        double t0 = bounds.elo0() / ELO_DIVISOR;
        double t1 = bounds.elo1() / ELO_DIVISOR;
        yield llrNormalized(totalReg, TRINOMIAL_SCORES, probs, t0, t1);
      }
    };
  }

  /**
   * Décision actuelle basée sur LLR vs bornes.
   *
   * <p>Cf. {@link SprtDecision} pour les valeurs possibles. {@link SprtDecision#CONTINUE} indique «
   * ni convergé ni cap atteint, continuer à enregistrer des games ».
   *
   * <p>Note : la décision {@link SprtDecision#MAX_GAMES} n'est PAS retournée par cette méthode (qui
   * ne connaît pas {@code maxGames}). Le tournament orchestrator gère ce cap externement.
   */
  public synchronized SprtDecision decision() {
    double llr = llr();
    if (llr >= bounds.upper()) {
      return SprtDecision.H1_ACCEPTED;
    }
    if (llr <= bounds.lower()) {
      return SprtDecision.H0_ACCEPTED;
    }
    return SprtDecision.CONTINUE;
  }

  /**
   * Score % du challenger (wins + 0.5 * draws) / total.
   *
   * @return score dans [0, 1], ou 0.5 si 0 games
   */
  public synchronized double scoreRate() {
    int total = wins + losses + draws;
    if (total == 0) return 0.5;
    return (wins + 0.5 * draws) / (double) total;
  }

  /**
   * Elo estimé via formule logistique inversée à partir du score%.
   *
   * <p>Cf. {@code fastchess::Stats::diff(model="logistic")}. Cas dégénéré : score 0 ou 1 →
   * Double.NEGATIVE_INFINITY ou Double.POSITIVE_INFINITY.
   */
  public synchronized double eloDiff() {
    double s = scoreRate();
    if (s <= 0.0) return Double.NEGATIVE_INFINITY;
    if (s >= 1.0) return Double.POSITIVE_INFINITY;
    return -400.0 * Math.log10(1.0 / s - 1.0);
  }

  // ============================== Private helpers ==============================

  private static double regularize(int value) {
    return value == 0 ? REGULARIZE_ZERO : (double) value;
  }

  /** Logistic Elo to expected score : sigma(elo / 400 * ln10). */
  static double leloToScore(double lelo) {
    return 1.0 / (1.0 + Math.pow(10.0, -lelo / 400.0));
  }

  /** BayesElo to expected score using draw_elo. */
  static double bayesEloToScore(double bayesElo, double drawElo) {
    double pwin = 1.0 / (1.0 + Math.pow(10.0, (-bayesElo + drawElo) / 400.0));
    double ploss = 1.0 / (1.0 + Math.pow(10.0, (bayesElo + drawElo) / 400.0));
    double pdraw = 1.0 - pwin - ploss;
    return pwin + 0.5 * pdraw;
  }

  /** LLR via modèle logistique/bayesien (MLE à contrainte de moyenne). */
  static double llrLogistic(double total, double[] scores, double[] probs, double s0, double s1) {
    double[] p0 = MleSolver.mleLogistic(scores, probs, s0);
    double[] p1 = MleSolver.mleLogistic(scores, probs, s1);
    double mean = 0.0;
    for (int i = 0; i < scores.length; i++) {
      mean += probs[i] * (Math.log(p1[i]) - Math.log(p0[i]));
    }
    return total * mean;
  }

  /** LLR via modèle normalized (MLE à contrainte de t = (mu - mu_ref) / sigma). */
  static double llrNormalized(double total, double[] scores, double[] probs, double t0, double t1) {
    double[] p0 = MleSolver.mleNormalized(scores, probs, 0.5, t0);
    double[] p1 = MleSolver.mleNormalized(scores, probs, 0.5, t1);
    double mean = 0.0;
    for (int i = 0; i < scores.length; i++) {
      mean += probs[i] * (Math.log(p1[i]) - Math.log(p0[i]));
    }
    return total * mean;
  }
}
