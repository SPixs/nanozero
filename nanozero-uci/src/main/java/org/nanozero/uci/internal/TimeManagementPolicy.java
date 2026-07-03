package org.nanozero.uci.internal;

import java.time.Duration;
import java.util.Objects;
import java.util.OptionalLong;
import org.nanozero.board.Color;
import org.nanozero.engine.SearchBudget;

/**
 * Politique de calcul de {@link SearchBudget} à partir d'une commande {@code go} parsée (cf. SPEC
 * §5.5, ADR-002, §12 phase 5).
 *
 * <p>API minimaliste : une méthode statique {@link #computeBudget} qui dispatch selon les
 * paramètres de {@link GoArgs} et retourne le {@link SearchBudget} adapté à passer à {@code
 * Engine.startSearch} / {@code Engine.startPonder}.
 *
 * <p><strong>Ordre de priorité</strong> (cf. SPEC §5.5) :
 *
 * <ol>
 *   <li>{@code movetime} présent : durée exacte ({@link SearchBudget#duration}).
 *   <li>{@code nodes} présent : limite simulations ({@link SearchBudget#nodes}).
 *   <li>{@code infinite} ou {@code ponder} : {@link SearchBudget#UNLIMITED}.
 *   <li>Time control standard ({@code wtime/btime}) : formule {@code allocated = myTime / divisor +
 *       myInc - moveOverhead}, bornée par {@code floor=50ms} et {@code cap=0.9×myTime}. Choix
 *       {@code wtime} vs {@code btime} selon {@code sideToMove}.
 *   <li>{@code depth} présent : best-effort {@code 100 × depth} simulations (MCTS n'a pas de notion
 *       claire de profondeur).
 *   <li>Aucun paramètre : fallback {@code 5s} (sécurité pseudo-humaine).
 * </ol>
 *
 * <p><strong>Tous les calculs sont en {@code long}</strong> pour éviter les surprises de promotion.
 * Le {@code cap} est calculé via {@code (long)(myTime * 0.9)} (cast après multiplication).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class TimeManagementPolicy {

  /** Diviseur par défaut pour le time control sans {@code movestogo} explicite. */
  private static final int DEFAULT_DIVISOR = 30;

  /** Plancher minimum d'allocation (ms). Garantit qu'on alloue au moins ce temps. */
  private static final long FLOOR_MS = 50;

  /** Plafond proportionnel au temps restant (90 % de {@code myTime}). */
  private static final double CAP_RATIO = 0.9;

  /** Fallback si aucun paramètre {@code go} n'est fourni : 5 s (sécurité pseudo-humaine). */
  private static final long FALLBACK_MS = 5_000;

  /**
   * Conversion best-effort {@code depth → nodes}. MCTS n'a pas de notion claire de profondeur ; on
   * approxime par un nombre de simulations proportionnel.
   */
  private static final int DEPTH_TO_NODES = 100;

  private TimeManagementPolicy() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Calcule le {@link SearchBudget} à appliquer pour la commande {@code go} reçue.
   *
   * @param args arguments parsés du {@code go}, non null
   * @param sideToMove couleur du joueur au trait dans la position courante, doit être {@link
   *     Color#WHITE} ou {@link Color#BLACK}
   * @param options état des options UCI courantes, non null (lit {@code moveOverheadMs})
   * @return budget conforme à appliquer
   * @throws NullPointerException si {@code args} ou {@code options} est null
   * @throws IllegalArgumentException si {@code sideToMove} n'est pas {@code WHITE} ou {@code BLACK}
   */
  public static SearchBudget computeBudget(GoArgs args, int sideToMove, UciOptionsState options) {
    Objects.requireNonNull(args, "args must not be null");
    Objects.requireNonNull(options, "options must not be null");
    if (sideToMove != Color.WHITE && sideToMove != Color.BLACK) {
      throw new IllegalArgumentException(
          "sideToMove must be Color.WHITE or Color.BLACK, got " + sideToMove);
    }

    // 1. movetime : durée exacte, priorité absolue.
    if (args.movetimeMs().isPresent()) {
      return SearchBudget.duration(Duration.ofMillis(args.movetimeMs().getAsLong()));
    }

    // 2. nodes : limite simulations.
    if (args.nodes().isPresent()) {
      return SearchBudget.nodes(args.nodes().getAsInt());
    }

    // 3. infinite ou ponder : pas de limite (arrêt par stop UCI ou ponderhit).
    if (args.infinite() || args.ponder()) {
      return SearchBudget.UNLIMITED;
    }

    // 4. Time control standard (wtime/btime) : formule allocate.
    OptionalLong myTimeOpt = (sideToMove == Color.WHITE) ? args.wtimeMs() : args.btimeMs();
    OptionalLong myIncOpt = (sideToMove == Color.WHITE) ? args.wincMs() : args.bincMs();
    if (myTimeOpt.isPresent()) {
      long myTime = myTimeOpt.getAsLong();
      long myInc = myIncOpt.orElse(0L);
      long divisor = args.movestogo().orElse(DEFAULT_DIVISOR);
      long moveOverhead = options.moveOverheadMs();

      long allocated = myTime / divisor + myInc - moveOverhead;
      long cap = (long) (myTime * CAP_RATIO);
      // floor prime sur cap si cap < floor (cas low-time avec inc important + overhead) ;
      // l'ordre Math.max(FLOOR, Math.min(allocated, cap)) garantit au moins FLOOR_MS.
      long bounded = Math.max(FLOOR_MS, Math.min(allocated, cap));
      return SearchBudget.duration(Duration.ofMillis(bounded));
    }

    // 5. depth : best-effort, pas de notion claire de profondeur en MCTS.
    if (args.depth().isPresent()) {
      int approximatedNodes = DEPTH_TO_NODES * args.depth().getAsInt();
      return SearchBudget.nodes(Math.max(0, approximatedNodes));
    }

    // 6. Fallback : aucun paramètre fourni.
    return SearchBudget.duration(Duration.ofMillis(FALLBACK_MS));
  }
}
