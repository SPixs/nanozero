package org.nanozero.engine.internal;

import java.util.Objects;
import org.nanozero.nn.NNOutput;
import org.nanozero.nn.Network;

/**
 * Wrapper trivial implémentant {@link NetworkProvider} en déléguant à un {@link Network} réel (cf.
 * SPEC §12 phase 7). Permet de connecter {@code SearcherCore} au réseau pré-chargé via {@code
 * NetworkLoader} (cf. SPEC-nn §5.2).
 *
 * <p>Le batch est figé à 1 conformément à ADR-011 ({@code Network.forward(planes, 1, output)}).
 *
 * <p>Wrappé par {@code Engine} en phase 9 (constructor public).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class NetworkProviderImpl implements NetworkProvider {

  private final Network network;

  /**
   * Construit un wrapper sur un {@link Network} pré-chargé.
   *
   * @param network réseau, non null
   * @throws NullPointerException si {@code network} est null
   */
  public NetworkProviderImpl(Network network) {
    this.network = Objects.requireNonNull(network, "network must not be null");
  }

  @Override
  public void forward(float[] planes, NNOutput output) {
    network.forward(planes, 1, output);
  }
}
