import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * Bench Phase 0 (ADR-001-worker / ADR-013-nn) : courbe throughput vs taille de batch sur ORT CUDA
 * EP, par paliers fixes avec warmup par palier.
 *
 * <p>Standalone — AUCUNE dépendance projet, lançable en mode source JDK 25 :
 *
 * <pre>
 *   java -cp onnxruntime_gpu-1.20.0.jar OrtCudaBatchBench.java model.onnx [--cpu]
 * </pre>
 *
 * <p>Sur Windows, les DLLs CUDA 12 + cuDNN 9 doivent être sur le PATH. Mesure pour chaque palier K
 * ∈ {1, 4, 8, 16, 32, 64} : latence moyenne d'un forward(K) et débit en positions/s. Le palier K=1
 * sert de baseline « loop forward(1) » (le débit loop = débit K=1).
 */
public final class OrtCudaBatchBench {

  private static final int PLANES = 119;
  private static final int[] BUCKETS = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  private static final int MAX_K = 1024;
  private static final int WARMUP_RUNS = 15;
  private static final long MEASURE_NANOS = 4_000_000_000L; // 4 s par palier

  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      System.err.println("usage: OrtCudaBatchBench <model.onnx> [--cpu]");
      System.exit(2);
    }
    Path model = Path.of(args[0]);
    boolean cpu = args.length > 1 && "--cpu".equals(args[1]);

    OrtEnvironment env = OrtEnvironment.getEnvironment();
    OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    opts.setIntraOpNumThreads(1);
    if (!cpu) {
      opts.addCUDA(0); // throws si le provider CUDA est indisponible — c'est le test sm_61
      System.out.println("# provider: CUDA EP (device 0)");
    } else {
      System.out.println("# provider: CPU EP");
    }

    try (OrtSession session = env.createSession(model.toString(), opts)) {
      String inputName = session.getInputNames().iterator().next();
      NodeInfo info = session.getInputInfo().get(inputName);
      System.out.println("# model: " + model + "  input: " + inputName + " " + info);
      System.out.println("# warmup " + WARMUP_RUNS + " runs/palier, mesure " + (MEASURE_NANOS / 1e9) + " s/palier");
      System.out.println();
      System.out.println("batch_K\truns\tmean_ms\tpos_per_s\tspeedup_vs_K1");

      // Données d'entrée pseudo-aléatoires 0/1 (le coût d'un ResNet est indépendant des valeurs)
      Random rng = new Random(42);
      float[] planes = new float[MAX_K * PLANES * 8 * 8];
      for (int i = 0; i < planes.length; i++) planes[i] = rng.nextInt(4) == 0 ? 1.0f : 0.0f;

      double baselinePosPerS = -1;
      for (int k : BUCKETS) {
        long[] shape = {k, PLANES, 8, 8};
        FloatBuffer buf = FloatBuffer.wrap(planes, 0, k * PLANES * 8 * 8);

        // Warmup du palier : force la compilation/sélection des kernels pour CETTE shape
        for (int w = 0; w < WARMUP_RUNS; w++) {
          runOnce(env, session, inputName, buf, shape);
        }

        // Mesure
        int runs = 0;
        long t0 = System.nanoTime();
        long elapsed;
        do {
          runOnce(env, session, inputName, buf, shape);
          runs++;
          elapsed = System.nanoTime() - t0;
        } while (elapsed < MEASURE_NANOS);

        double meanMs = elapsed / 1e6 / runs;
        double posPerS = k * runs / (elapsed / 1e9);
        if (baselinePosPerS < 0) baselinePosPerS = posPerS;
        System.out.printf(
            "%d\t%d\t%.2f\t%.0f\t%.1fx%n", k, runs, meanMs, posPerS, posPerS / baselinePosPerS);
      }
    }
  }

  private static void runOnce(
      OrtEnvironment env, OrtSession session, String inputName, FloatBuffer buf, long[] shape)
      throws Exception {
    buf.rewind();
    try (OnnxTensor t = OnnxTensor.createTensor(env, buf, shape);
        OrtSession.Result r = session.run(Map.of(inputName, t))) {
      // touche la sortie pour empêcher toute élision (shape [K] ou [K,1] selon export)
      Object v = r.get(r.size() - 1).getValue();
      float probe =
          v instanceof float[][] m ? m[0][0] : v instanceof float[] a ? a[0] : 0.0f;
      if (Float.isNaN(probe)) throw new IllegalStateException("NaN output");
    }
  }
}
