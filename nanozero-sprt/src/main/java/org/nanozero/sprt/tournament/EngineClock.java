package org.nanozero.sprt.tournament;

import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * Horloge per-engine pour une game SPRT. Track le temps restant, applique les increments Fischer
 * après chaque coup, détecte les time forfeits (temps épuisé pendant un move).
 *
 * <p>Cf. SPEC-sprt §5.2.
 *
 * <p><strong>Pattern d'usage</strong> :
 *
 * <pre>{@code
 * EngineClock clock = new EngineClock(TimeControl.fischer(Duration.ofSeconds(10), Duration.ofMillis(100)));
 * // ... game loop:
 * Duration budget = clock.budgetForNextMove();
 * clock.startMove();
 * SearchResult result = engine.startSearch(state, SearchBudget.deadline(...));
 * boolean forfeit = clock.endMove();  // returns true si timeout
 * if (forfeit) {
 *   // game lost on time
 * }
 * }</pre>
 *
 * <p><strong>Concurrent safety</strong> : non thread-safe. Chaque {@code EngineClock} est dédié à
 * un engine et accédé par un seul thread de {@code GameRunner}.
 *
 * <p>Cf. SPEC-sprt §5.3 pour les edge cases (insufficient time, engine crash, etc.).
 */
public final class EngineClock {

  /**
   * Temps minimum requis pour démarrer un move (heuristique défensive). Si {@code remaining &lt;
   * MIN_BUDGET}, on déclare un forfeit immédiat sans même tenter le move.
   */
  public static final Duration MIN_BUDGET = Duration.ofMillis(50);

  private final TimeControl config;
  private Duration remaining;
  private int movesPlayed;
  private Instant moveStartedAt;
  private boolean forfeitted;

  /**
   * Construit une horloge initialisée au {@code baseTime} du {@link TimeControl}.
   *
   * @param config time control non null
   */
  public EngineClock(TimeControl config) {
    this.config = Objects.requireNonNull(config, "config must not be null");
    this.remaining = config.baseTime();
  }

  /** Temps restant courant. */
  public Duration remaining() {
    return remaining;
  }

  /** Nombre de coups joués. */
  public int movesPlayed() {
    return movesPlayed;
  }

  /** Time control associé. */
  public TimeControl config() {
    return config;
  }

  /** {@code true} si l'engine est déjà en forfeit (time out détecté). */
  public boolean isForfeit() {
    return forfeitted;
  }

  /**
   * Budget de temps à passer à l'engine pour le prochain coup. Pour les modes Fischer/sudden, c'est
   * simplement {@link #remaining()}. Pour le mode {@code unlimited}, retourne {@link Duration#ZERO}
   * (= pas de deadline ; le caller doit utiliser un budget {@code UNLIMITED}).
   *
   * @return budget pour le prochain coup
   * @throws IllegalStateException si en forfeit
   */
  public Duration budgetForNextMove() {
    if (forfeitted) {
      throw new IllegalStateException("clock is already forfeit");
    }
    if (config.unlimited()) {
      return Duration.ZERO;
    }
    return remaining;
  }

  /**
   * Vérifie si l'engine a assez de temps pour démarrer un move (au moins {@link #MIN_BUDGET}).
   *
   * <p>Si {@code false}, le caller doit appeler {@link #forceTimeForfeit()} sans même tenter le
   * move (insufficient time).
   *
   * @return {@code true} si le budget est suffisant (ou si unlimited)
   */
  public boolean hasEnoughTime() {
    if (forfeitted) return false;
    if (config.unlimited()) return true;
    return remaining.compareTo(MIN_BUDGET) >= 0;
  }

  /**
   * Marque le début d'un move (capture {@link Instant#now()}). Doit être appelé juste avant
   * l'invocation de l'engine.
   *
   * @throws IllegalStateException si déjà en forfeit ou si {@code startMove()} a déjà été appelé
   *     sans {@code endMove()} subséquent
   */
  public void startMove() {
    if (forfeitted) {
      throw new IllegalStateException("clock is already forfeit");
    }
    if (moveStartedAt != null) {
      throw new IllegalStateException(
          "startMove() called twice without endMove() — sequencing error");
    }
    moveStartedAt = Instant.now();
  }

  /**
   * Marque la fin d'un move. Consomme le temps écoulé depuis {@link #startMove()}, applique
   * l'increment Fischer, et détecte les time forfeits.
   *
   * @return {@code true} si time forfeit déclenché (temps restant ≤ 0 après consommation)
   * @throws IllegalStateException si {@link #startMove()} n'a pas été appelé
   */
  public boolean endMove() {
    if (moveStartedAt == null) {
      throw new IllegalStateException("endMove() called without prior startMove()");
    }
    Duration elapsed = Duration.between(moveStartedAt, Instant.now());
    moveStartedAt = null;
    movesPlayed++;

    if (config.unlimited()) {
      return false;
    }

    remaining = remaining.minus(elapsed);
    if (remaining.isNegative() || remaining.isZero()) {
      forfeitted = true;
      return true;
    }
    // Apply Fischer increment.
    remaining = remaining.plus(config.increment());

    // Classical TC : every movesPerControl moves, reset baseTime budget.
    if (config.movesPerControl() > 0 && movesPlayed % config.movesPerControl() == 0) {
      remaining = remaining.plus(config.baseTime());
    }
    return false;
  }

  /**
   * Force un time forfeit (utilisé quand {@link #hasEnoughTime()} retourne {@code false} avant même
   * de démarrer un move).
   */
  public void forceTimeForfeit() {
    forfeitted = true;
    moveStartedAt = null;
  }
}
