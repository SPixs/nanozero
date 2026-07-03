package org.nanozero.engine.internal.batched;

import java.util.function.BiConsumer;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;
import org.nanozero.nn.NNOutput;
import org.nanozero.nn.NNSingleResult;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkMetadata;

/**
 * Helpers pour construire des {@link Network} mocks dans les tests batched. {@link Network} ayant 4
 * méthodes abstraites, on évite la verbosité de l'anonymous-class par appel.
 */
final class TestNetworks {

  private TestNetworks() {}

  /**
   * Wrap un {@link BiConsumer}-style forward dans une instance {@link Network} complète. Les 3
   * autres méthodes (forwardSingle, metadata, planeEncoder) ne sont pas appelées par le code
   * batched testé : elles lèvent ou retournent null.
   */
  static Network forwardOnly(ForwardImpl impl) {
    return new Network() {
      @Override
      public void forward(float[] planes, int batchSize, NNOutput output) {
        impl.apply(planes, batchSize, output);
      }

      @Override
      public NNSingleResult forwardSingle(GameState state) {
        throw new UnsupportedOperationException("forwardSingle not used in batched tests");
      }

      @Override
      public NetworkMetadata metadata() {
        throw new UnsupportedOperationException("metadata not used in batched tests");
      }

      @Override
      public BitboardPlaneEncoder planeEncoder() {
        throw new UnsupportedOperationException("planeEncoder not used in batched tests");
      }
    };
  }

  /** Network qui ne fait rien (forward no-op). */
  static Network noop() {
    return forwardOnly((planes, batchSize, output) -> {});
  }

  /** Functional interface pour forward (3 args), non disponible nativement Java. */
  @FunctionalInterface
  interface ForwardImpl {
    void apply(float[] planes, int batchSize, NNOutput output);
  }
}
