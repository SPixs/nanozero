package org.nanozero.engine.internal;

import java.util.Objects;
import org.nanozero.engine.SearchBudget;

/**
 * Wrapper interne d'un {@link SearchBudget} dont le délégué peut être remplacé à chaud sans
 * interrompre le worker thread (cf. SPEC §12 phase 11).
 *
 * <p>Cas d'usage : mode ponder. {@code startPonder} installe un délégué {@code
 * SearchBudget.UNLIMITED} qui laisse le worker boucler indéfiniment ; {@code ponderhit} appelle
 * {@link #replace(SearchBudget)} pour basculer vers le budget réel sans signal d'interruption. Le
 * worker, qui itère normalement, observe le nouveau délégué au prochain appel à {@link
 * #shouldStop}, qui peut alors déclencher une transition vers {@code DONE} via le flux usuel
 * (budget exhausted).
 *
 * <p><strong>Concurrence</strong> : {@code delegate} est {@code volatile}. La cohérence
 * happens-before entre le caller appelant {@code replace} et le worker lisant {@code shouldStop}
 * est garantie par les sémantiques volatile. Aucun verrou requis.
 *
 * <p>L'API publique {@link SearchBudget} n'est PAS impactée par ce wrapper : il est consommé
 * exclusivement en interne par {@link ThreadController}, qui wrap le {@code SearchBudget} fourni
 * par le caller au moment de la soumission.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class MutableSearchBudget implements SearchBudget {

  private volatile SearchBudget delegate;

  /**
   * Construit un wrapper sur un délégué initial.
   *
   * @param initial délégué initial, non null
   * @throws NullPointerException si {@code initial} est null
   */
  MutableSearchBudget(SearchBudget initial) {
    this.delegate = Objects.requireNonNull(initial, "initial must not be null");
  }

  /**
   * Remplace le délégué courant. Le prochain appel à {@link #shouldStop} interrogera le nouveau
   * délégué. Visibilité garantie via {@code volatile}.
   *
   * @param newDelegate nouveau délégué, non null
   * @throws NullPointerException si {@code newDelegate} est null
   */
  void replace(SearchBudget newDelegate) {
    this.delegate = Objects.requireNonNull(newDelegate, "newDelegate must not be null");
  }

  @Override
  public boolean shouldStop(int simulationsElapsed, long elapsedNanos) {
    return delegate.shouldStop(simulationsElapsed, elapsedNanos);
  }
}
