import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * Sonde TF32 (Ampere+) : compare CPU EP vs CUDA EP avec use_tf32=1 (défaut) et use_tf32=0, sur le
 * même batch K. Mesure aussi le débit K=256 dans les deux modes. Standalone mode source JDK 25 :
 *
 * <pre>
 *   java -cp onnxruntime_gpu-1.20.0.jar CudaTf32Probe.java model.onnx
 * </pre>
 */
public final class CudaTf32Probe {

  private static final int PLANES = 119;
  private static final int K = 256;
  private static final long MEASURE_NANOS = 4_000_000_000L;

  public static void main(String[] args) throws Exception {
    Path model = Path.of(args[0]);
    OrtEnvironment env = OrtEnvironment.getEnvironment();

    float[] input = new float[K * PLANES * 64];
    Random rnd = new Random(42);
    for (int i = 0; i < input.length; i++) input[i] = rnd.nextInt(4) == 0 ? 1f : 0f;

    try (OrtSession cpu = env.createSession(model.toString(), new OrtSession.SessionOptions())) {
      float[][] ref = run(env, cpu, input, K);

      for (int tf32 = 1; tf32 >= 0; tf32--) {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        OrtCUDAProviderOptions cuda = new OrtCUDAProviderOptions(0);
        cuda.add("use_tf32", Integer.toString(tf32));
        so.addCUDA(cuda);
        try (OrtSession gpu = env.createSession(model.toString(), so)) {
          // warmup
          for (int w = 0; w < 10; w++) run(env, gpu, input, K);
          float[][] out = run(env, gpu, input, K);
          double dPolicy = 0, dValue = 0;
          for (int i = 0; i < out[0].length; i++)
            dPolicy = Math.max(dPolicy, Math.abs(out[0][i] - ref[0][i]));
          for (int i = 0; i < out[1].length; i++)
            dValue = Math.max(dValue, Math.abs(out[1][i] - ref[1][i]));

          long t0 = System.nanoTime();
          int runs = 0;
          while (System.nanoTime() - t0 < MEASURE_NANOS) {
            run(env, gpu, input, K);
            runs++;
          }
          double sec = (System.nanoTime() - t0) / 1e9;
          System.out.printf(
          "use_tf32=%d  max|dPolicyLogit|=%.3e  max|dValueLogit|=%.3e  K=%d: %.0f pos/s%n",
              tf32, dPolicy, dValue, K, runs * (double) K / sec);
        }
      }
    }
  }

  private static float[][] run(OrtEnvironment env, OrtSession s, float[] input, int k)
      throws Exception {
    try (OnnxTensor t =
            OnnxTensor.createTensor(
                env, FloatBuffer.wrap(input), new long[] {k, PLANES, 8, 8});
        OrtSession.Result r = s.run(Map.of("board", t))) {
      float[][] policy = (float[][]) r.get(0).getValue();
      float[][] value = (float[][]) r.get(1).getValue();
      float[] pFlat = new float[k * policy[0].length];
      float[] vFlat = new float[k * value[0].length];
      for (int b = 0; b < k; b++) {
        System.arraycopy(policy[b], 0, pFlat, b * policy[0].length, policy[0].length);
        System.arraycopy(value[b], 0, vFlat, b * value[0].length, value[0].length);
      }
      return new float[][] {pFlat, vFlat};
    }
  }
}
