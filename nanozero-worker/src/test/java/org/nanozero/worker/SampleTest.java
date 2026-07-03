package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import org.junit.jupiter.api.Test;

class SampleTest {

  @Test
  void inputPlanesBase64RoundTrip() {
    float[] planes = {1.0f, 2.5f, -3.25f, 0.0f, Float.MAX_VALUE};
    Sample s = new Sample(planes, new float[0], 0, 0);

    byte[] decoded = Base64.getDecoder().decode(s.inputPlanesBase64());
    ByteBuffer bb = ByteBuffer.wrap(decoded).order(ByteOrder.LITTLE_ENDIAN);

    assertThat(decoded).hasSize(planes.length * 4);
    for (float expected : planes) {
      assertThat(bb.getFloat()).isEqualTo(expected);
    }
  }

  @Test
  void policyTargetBase64RoundTrip() {
    float[] policy = new float[4672];
    policy[100] = 0.5f;
    policy[3000] = 0.25f;
    Sample s = new Sample(new float[0], policy, 0, 0);

    byte[] decoded = Base64.getDecoder().decode(s.policyTargetBase64());
    assertThat(decoded).hasSize(4672 * 4);

    ByteBuffer bb = ByteBuffer.wrap(decoded).order(ByteOrder.LITTLE_ENDIAN);
    assertThat(bb.getFloat(100 * 4)).isEqualTo(0.5f);
    assertThat(bb.getFloat(3000 * 4)).isEqualTo(0.25f);
  }

  @Test
  void valueTargetStartsAsNaN() {
    Sample s = new Sample(new float[1], new float[1], 0, 5);
    assertThat(Float.isNaN(s.valueTarget)).isTrue();
  }

  @Test
  void valueTargetIsMutableForBackfill() {
    Sample s = new Sample(new float[1], new float[1], 0, 5);
    s.valueTarget = -1.0f;
    assertThat(s.valueTarget).isEqualTo(-1.0f);
  }

  @Test
  void emptyArrayBase64IsEmpty() {
    Sample s = new Sample(new float[0], new float[0], 0, 0);
    assertThat(s.inputPlanesBase64()).isEmpty();
    assertThat(s.policyTargetBase64()).isEmpty();
  }
}
