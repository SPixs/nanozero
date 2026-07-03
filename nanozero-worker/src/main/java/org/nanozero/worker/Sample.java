package org.nanozero.worker;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;

/**
 * One self-play training sample collected by {@link GamePlayer}.
 *
 * <p>Mutable record-like : {@code valueTarget} is set in a second pass once the game outcome is
 * known (backfill from outcomeWhite POV cf. ADR-001).
 *
 * <p>Wire format: float arrays are little-endian float32 byte buffers, base64-encoded.
 *
 * <ul>
 *   <li>{@code inputPlanes} : 119 × 64 = 7 616 floats → 30 464 bytes
 *   <li>{@code policyTarget} : 4 672 floats → 18 688 bytes
 * </ul>
 */
public final class Sample {

  /** {@code 119 * 64} floats — the 119-plane AlphaZero input encoding. */
  public final float[] inputPlanes;

  /** {@code 4 672} floats — target visit distribution over the policy space. */
  public final float[] policyTarget;

  /** Side-to-move at this sample's ply : {@code 0=WHITE}, {@code 1=BLACK}. */
  public final int turn;

  /** 0-based half-move index from game start. */
  public final int ply;

  /** Outcome from this side-to-move's POV : {@code -1, 0, +1}. Backfilled post-game. */
  public float valueTarget;

  public Sample(float[] inputPlanes, float[] policyTarget, int turn, int ply) {
    this.inputPlanes = inputPlanes;
    this.policyTarget = policyTarget;
    this.turn = turn;
    this.ply = ply;
    this.valueTarget = Float.NaN; // sentinel — set during backfill
  }

  /** Base64-encode the input planes (little-endian float32 bytes). */
  public String inputPlanesBase64() {
    return base64FloatArray(inputPlanes);
  }

  /** Base64-encode the policy target (little-endian float32 bytes). */
  public String policyTargetBase64() {
    return base64FloatArray(policyTarget);
  }

  /**
   * Pack a {@code float[]} as little-endian float32 bytes, then base64-encode.
   *
   * <p>Little-endian matches numpy's default {@code dtype=np.float32} byte order on x86/64 — so the
   * trainer can decode directly with {@code np.frombuffer(buf, dtype=np.float32)}.
   */
  static String base64FloatArray(float[] arr) {
    ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : arr) {
      buf.putFloat(v);
    }
    return Base64.getEncoder().encodeToString(buf.array());
  }
}
