import org.nanozero.nn.NNOutput;
import org.nanozero.nn.NetworkOnnx;
import java.nio.file.Path;
import java.util.Random;

/**
 * Validation Phase 1 (ADR-013) du chemin batché CUDA de {@link NetworkOnnx} sur machine GPU — hors
 * CI. Complète {@code NetworkOnnxBatchedTest} (CI, seam CPU) en exerçant le VRAI chemin de
 * production : {@code loadCuda} → warmup paliers → forward(K) → extraction WDL.
 *
 * <p>Lancement (mode source JDK 25, jar UCI shadé -Pgpu sur le classpath, DLLs CUDA dans PATH) :
 *
 * <pre>
 *   java --add-modules jdk.incubator.vector -cp nanozero-uci-X.Y.Z.jar \
 *       NetworkOnnxCudaValidation.java model.onnx
 * </pre>
 *
 * <p>Deux volets : (1) parité numérique CUDA batché vs CPU loop (mêmes positions), (2) débit par
 * palier à travers l'API complète (overhead extraction inclus — chiffre « monde réel » à comparer
 * au harness ORT brut {@code OrtCudaBatchBench}).
 */
public final class NetworkOnnxCudaValidation {

  private static final int FLOATS = NetworkOnnx.INPUT_FLOATS_PER_POS;
  private static final int LOGITS = NetworkOnnx.POLICY_LOGITS_PER_POS;
  private static final long MEASURE_NANOS = 4_000_000_000L;

  public static void main(String[] args) throws Exception {
    Path model = Path.of(args[0]);
    System.out.println("# loading CPU EP (loop reference)...");
    NetworkOnnx cpu = NetworkOnnx.load(model);
    System.out.println("# loading CUDA EP (batched, warmup paliers)...");
    long t0 = System.nanoTime();
    NetworkOnnx cuda = NetworkOnnx.loadCuda(model);
    System.out.printf("# loadCuda + warmup : %.1f s%n", (System.nanoTime() - t0) / 1e9);
    System.out.println("# maxBatch : cpu=" + cpu.maxBatch() + " cuda=" + cuda.maxBatch());

    Random rng = new Random(42);
    float[] planes = new float[cuda.maxBatch() * FLOATS];
    for (int i = 0; i < planes.length; i++) planes[i] = rng.nextInt(4) == 0 ? 1.0f : 0.0f;

    // ---- Volet 1 : parité CUDA batché vs CPU loop ----
    // Seuils : |dV| < 1e-4 (value scalaire) et |dPrior| < 1e-3 (policy POST-softmax, la
    // grandeur consommée par le PUCT). Les logits BRUTS divergent de ~1e-3..1e-2 entre EPs et
    // entre tailles de batch — cuDNN sélectionne des engines différents par shape (ordre des
    // réductions float32 différent) ; c'est attendu et sans effet opérationnel (rapporté à
    // titre informatif).
    System.out.println();
    System.out.println("== parité CUDA batché vs CPU loop ==");
    System.out.println("batch_K\tmax|dV|\tmax|dPrior|\tmax|dLogit| (info)");
    NNOutput cpuOut = new NNOutput();
    NNOutput cudaOut = new NNOutput(cuda.maxBatch());
    boolean ok = true;
    for (int k : new int[] {1, 3, 17, 64}) {
      cpu.forward(planes, k, cpuOut);
      cuda.forward(planes, k, cudaOut);
      double dv = 0;
      double dp = 0;
      double dl = 0;
      for (int b = 0; b < k; b++) {
        dv = Math.max(dv, Math.abs(cpuOut.valueOf(b) - cudaOut.valueOf(b)));
        float[] le = cpuOut.logitsOf(b);
        float[] la = cudaOut.logitsOf(b);
        dp = Math.max(dp, maxSoftmaxDiff(le, la));
        for (int i = 0; i < LOGITS; i++) dl = Math.max(dl, Math.abs(le[i] - la[i]));
      }
      ok &= dv < 1e-4 && dp < 1e-3;
      System.out.printf("%d\t%.2e\t%.2e\t%.2e%n", k, dv, dp, dl);
    }
    // >64 : référence = CUDA forward(1) position par position (échantillonné)
    for (int k : new int[] {100, 256}) {
      cuda.forward(planes, k, cudaOut);
      float[] single = new float[FLOATS];
      NNOutput ref = new NNOutput(1);
      double dv = 0;
      double dp = 0;
      double dl = 0;
      for (int b = 0; b < k; b += 41) {
        System.arraycopy(planes, b * FLOATS, single, 0, FLOATS);
        cuda.forward(single, 1, ref);
        dv = Math.max(dv, Math.abs(ref.valueOf(0) - cudaOut.valueOf(b)));
        float[] le = ref.logitsOf(0);
        float[] la = cudaOut.logitsOf(b);
        dp = Math.max(dp, maxSoftmaxDiff(le, la));
        for (int i = 0; i < LOGITS; i++) dl = Math.max(dl, Math.abs(le[i] - la[i]));
      }
      ok &= dv < 1e-4 && dp < 1e-3;
      System.out.printf("%d*\t%.2e\t%.2e\t%.2e\t(* vs CUDA unitaire, échantillonné)%n", k, dv, dp, dl);
    }
    System.out.println(ok ? "PARITY OK" : "PARITY FAILED");

    // ---- Volet 2 : débit par palier via l'API complète ----
    System.out.println();
    System.out.println("== débit NetworkOnnx.forward (API complète, extraction incluse) ==");
    System.out.println("batch_K\truns\tmean_ms\tpos_per_s");
    for (int k : new int[] {1, 8, 32, 64, 128, 256}) {
      int runs = 0;
      long start = System.nanoTime();
      long elapsed;
      do {
        cuda.forward(planes, k, cudaOut);
        runs++;
        elapsed = System.nanoTime() - start;
      } while (elapsed < MEASURE_NANOS);
      System.out.printf(
          "%d\t%d\t%.2f\t%.0f%n", k, runs, elapsed / 1e6 / runs, k * runs / (elapsed / 1e9));
    }
    if (!ok) System.exit(1);
  }

  /** Max |Δp| entre les deux distributions softmax (sur tous les 4672 logits). */
  private static double maxSoftmaxDiff(float[] logitsA, float[] logitsB) {
    double[] pa = softmax(logitsA);
    double[] pb = softmax(logitsB);
    double max = 0;
    for (int i = 0; i < pa.length; i++) max = Math.max(max, Math.abs(pa[i] - pb[i]));
    return max;
  }

  private static double[] softmax(float[] logits) {
    double maxLogit = Double.NEGATIVE_INFINITY;
    for (float l : logits) maxLogit = Math.max(maxLogit, l);
    double[] p = new double[logits.length];
    double sum = 0;
    for (int i = 0; i < logits.length; i++) {
      p[i] = Math.exp(logits[i] - maxLogit);
      sum += p[i];
    }
    for (int i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }
}
