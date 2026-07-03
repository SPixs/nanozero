package org.nanozero.worker;

import java.lang.reflect.Field;
import java.util.Arrays;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;
import org.nanozero.nn.BitboardPlaneEncoderVector;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;
import org.nanozero.nn.NNSingleResult;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkMetadata;

/**
 * Deterministic in-memory {@link Network} for worker tests. Drives a real {@code Engine}/{@code
 * GamePlayer} without ONNX Runtime — avoids the {@code NetworkOnnx} native binding entirely.
 *
 * <p>{@link #forward} fills {@link NNOutput} with all-zero logits (≈ uniform policy after the
 * engine's masked softmax over legal moves) and a fixed value. Writes to {@code NNOutput}'s
 * package-private buffers via reflection — the established pattern in this codebase (cf. {@code
 * SearcherCoreTest.MockNetworkProvider}), as {@code NNOutput} exposes no public mutator.
 *
 * <p>Optionally counts {@code AutoCloseable.close()} invocations so {@link ModelCache} swap/close
 * branches can be asserted. The variant returned by {@link #closeableThrowing()} throws on {@code
 * close()} to exercise {@code ModelCache.closeLoaded}'s best-effort exception branch.
 */
public class FakeNetwork implements Network, AutoCloseable {

  private static final Field LOGITS_FIELD;
  private static final Field VALUES_FIELD;

  static {
    try {
      LOGITS_FIELD = NNOutput.class.getDeclaredField("logits");
      VALUES_FIELD = NNOutput.class.getDeclaredField("values");
      LOGITS_FIELD.setAccessible(true);
      VALUES_FIELD.setAccessible(true);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private final float value;
  private int closeCount = 0;

  public FakeNetwork() {
    this(0.0f);
  }

  public FakeNetwork(float value) {
    this.value = value;
  }

  @Override
  public void forward(float[] planes, int batchSize, NNOutput output) {
    try {
      float[] outLogits = (float[]) LOGITS_FIELD.get(output);
      float[] outValues = (float[]) VALUES_FIELD.get(output);
      // Uniform logits (all zeros → flat masked-softmax over legal moves) for each batch slot.
      for (int b = 0; b < batchSize; b++) {
        int base = b * MoveEncoding.POLICY_INDICES;
        Arrays.fill(outLogits, base, base + MoveEncoding.POLICY_INDICES, 0.0f);
        outValues[b] = value;
      }
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("FakeNetwork reflection setup failed", e);
    }
  }

  @Override
  public NNSingleResult forwardSingle(GameState state) {
    return new NNSingleResult(new float[MoveEncoding.POLICY_INDICES], value);
  }

  @Override
  public NetworkMetadata metadata() {
    return new NetworkMetadata(
        "resnet8x96-v1", "fake-hash", 0L, "2026-05-30T00:00:00Z", "alphazero-119");
  }

  @Override
  public BitboardPlaneEncoder planeEncoder() {
    return BitboardPlaneEncoderVector.INSTANCE;
  }

  /** Number of times {@link #close()} was invoked (for ModelCache swap/close assertions). */
  public int closeCount() {
    return closeCount;
  }

  @Override
  public void close() {
    closeCount++;
  }

  /** A FakeNetwork whose {@link #close()} throws — exercises ModelCache's best-effort catch. */
  public static FakeNetwork closeableThrowing() {
    return new FakeNetwork() {
      @Override
      public void close() {
        throw new IllegalStateException("close boom");
      }
    };
  }
}
