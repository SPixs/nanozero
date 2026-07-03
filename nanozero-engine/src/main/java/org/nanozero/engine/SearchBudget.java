package org.nanozero.engine;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Limite de la recherche, vérifiée par le moteur à chaque itération de la boucle MCTS (cf. SPEC
 * §3.4).
 *
 * <p>L'interface est composable : on combine plusieurs critères (nœuds, durée, flag externe) via
 * {@link #firstOf(SearchBudget...)} pour répondre aux scénarios UCI typiques (ex. {@code go nodes
 * 10000} + {@code stop} async).
 *
 * <p>Invariants normatifs §3.4 :
 *
 * <ul>
 *   <li><strong>I-Budget-1</strong> : {@link #shouldStop(int, long)} est appelée séquentiellement
 *       par le thread de recherche, jamais concurremment. Une implémentation non-thread-safe
 *       (compteur interne) est acceptable.
 *   <li><strong>I-Budget-2</strong> : une fois {@code shouldStop} retourne {@code true}, l'engine
 *       arrête à la prochaine vérification ; un retour à {@code false} ultérieur est ignoré (pas de
 *       reprise après arrêt).
 * </ul>
 */
@FunctionalInterface
public interface SearchBudget {

  /**
   * Indique si la recherche doit s'arrêter. Appelée après chaque simulation complète (cohérence
   * dans le state).
   *
   * @param simulationsElapsed nombre de simulations terminées depuis le début (≥ 0)
   * @param elapsedNanos durée écoulée depuis le début, en nanosecondes (≥ 0)
   * @return {@code true} si l'engine doit arrêter
   */
  boolean shouldStop(int simulationsElapsed, long elapsedNanos);

  // ------------------------------------------------------------------------------------
  // Constantes / factories
  // ------------------------------------------------------------------------------------

  /**
   * Budget sans limite. Utile pour le mode ponder où l'arrêt est piloté par {@code stop()} ou par
   * {@link #untilStopped(AtomicBoolean)}.
   */
  SearchBudget UNLIMITED = (sims, nanos) -> false;

  /**
   * Limite par nombre de simulations terminées.
   *
   * <p>{@code maxSimulations = 0} est légal : déclenche un stop avant la première simulation (utile
   * pour des tests d'API).
   *
   * @param maxSimulations nombre maximum de simulations (≥ 0)
   * @return budget qui retourne {@code true} dès que {@code simulationsElapsed >= maxSimulations}
   * @throws IllegalArgumentException si {@code maxSimulations < 0}
   */
  static SearchBudget nodes(int maxSimulations) {
    if (maxSimulations < 0) {
      throw new IllegalArgumentException("maxSimulations must be >= 0, got " + maxSimulations);
    }
    return (sims, nanos) -> sims >= maxSimulations;
  }

  /**
   * Limite par deadline absolue en {@code System.nanoTime()}. Ignore {@code elapsedNanos}
   * (comparaison wall-clock pure). Cohérent avec {@code go movetime} d'UCI.
   *
   * @param deadlineNanos instant absolu en {@code System.nanoTime()}
   * @return budget qui retourne {@code true} dès que {@code System.nanoTime() >= deadlineNanos}
   */
  static SearchBudget deadline(long deadlineNanos) {
    return (sims, nanos) -> System.nanoTime() >= deadlineNanos;
  }

  /**
   * Limite par durée relative au début de la recherche. Compare {@code elapsedNanos} fourni par
   * l'engine à {@code duration.toNanos()} ; pas de {@code System.nanoTime()} interne, ce qui rend
   * la factory déterministe et utilisable en test.
   *
   * @param duration durée maximale (non null, non négative)
   * @return budget qui retourne {@code true} dès que {@code elapsedNanos >= duration.toNanos()}
   * @throws NullPointerException si {@code duration} est null
   * @throws IllegalArgumentException si {@code duration} est négative
   */
  static SearchBudget duration(Duration duration) {
    Objects.requireNonNull(duration, "duration must not be null");
    if (duration.isNegative()) {
      throw new IllegalArgumentException("duration must be >= 0, got " + duration);
    }
    long limitNanos = duration.toNanos();
    return (sims, nanos) -> nanos >= limitNanos;
  }

  /**
   * Limite par flag externe. Le stop UCI typique {@code stop} ou un flag de timeout pilotent ce
   * budget.
   *
   * @param stopFlag flag externe (non null)
   * @return budget qui retourne {@code stopFlag.get()}
   * @throws NullPointerException si {@code stopFlag} est null
   */
  static SearchBudget untilStopped(AtomicBoolean stopFlag) {
    Objects.requireNonNull(stopFlag, "stopFlag must not be null");
    return (sims, nanos) -> stopFlag.get();
  }

  /**
   * Composition : le premier des budgets à déclencher gagne. Évalue les budgets dans l'ordre fourni
   * et court-circuite au premier {@code true}.
   *
   * <p>{@code firstOf()} (sans budget) est légal et équivalent à {@link #UNLIMITED}.
   *
   * @param budgets budgets composants (non null, aucun élément null)
   * @return budget composite
   * @throws NullPointerException si {@code budgets} ou un élément est null
   */
  static SearchBudget firstOf(SearchBudget... budgets) {
    Objects.requireNonNull(budgets, "budgets must not be null");
    for (int i = 0; i < budgets.length; i++) {
      if (budgets[i] == null) {
        throw new NullPointerException("budgets[" + i + "] must not be null");
      }
    }
    // Copie défensive pour éviter une mutation externe ultérieure de l'array.
    SearchBudget[] copy = budgets.clone();
    return (sims, nanos) -> {
      for (SearchBudget b : copy) {
        if (b.shouldStop(sims, nanos)) {
          return true;
        }
      }
      return false;
    };
  }
}
