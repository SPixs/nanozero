package org.nanozero.nn;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;

/**
 * Implémentation {@link Network} via ONNX Runtime Java (Phase 12 PoC, v1.1.0+).
 *
 * <p>Format input attendu : modèle ONNX BN-folded équivalent au {@code .npz} consumé par {@link
 * NetworkVectorApi}. Exporté par {@code nanozero-nn-onnx/scripts/ phase2a_export_bn_folded_onnx.py}
 * ou par le pipeline training Python (post-Phase 12).
 *
 * <p>Speedup observé phase 12-prod-launch (W3090 Ryzen 5 3600X) :
 *
 * <ul>
 *   <li>NetworkVectorApi (Vector API SIMD) : ~49 inf/sec
 *   <li>NetworkOnnx CPU EP : ~844 inf/sec (17×)
 *   <li>NetworkOnnx CUDA EP : ~1103 inf/sec (22×)
 * </ul>
 *
 * <p>CPU EP par défaut. CUDA EP via {@link #loadCuda(Path)} (requiert build {@code
 * onnxruntime_gpu}).
 *
 * <p>Deux chemins {@link #forward} (amendement v1.6.0, ADR-013) :
 *
 * <ul>
 *   <li><b>CPU EP</b> : loop K×forward(1). Le batching ORT ne paie pas sur CPU (courbe plate
 *       mesurée, cf. PERF-GPU-BATCH-phase0.md) ; la boucle est stable et thread-safe.
 *   <li><b>CUDA EP</b> : vrai forward(K) en un seul {@code session.run}, par paliers fixes {@link
 *       #CUDA_BATCH_BUCKETS} avec padding au palier supérieur et warmup de chaque palier au
 *       chargement. L'ensemble des shapes vues par CUDA EP est fini et pré-compilé — élimine les
 *       recompiles de kernels qui effondraient le forward(K) à shape libre (bench 2026-05-20). 1080
 *       Ti : 1174 pos/s @K=1 → 18457 @K=256.
 * </ul>
 *
 * <p><b>IO zéro-alloc du chemin batché (v1.6.0)</b> : le profil JFR du worker GPU en production
 * (2026-06-11) a montré que l'ancienne implémentation passait ~19 % du CPU dans {@code
 * Bits.setMemory} (allocation + zeroing d'un direct buffer PAR batch dans {@code createTensor} sur
 * buffer heap) et ~11 % dans le boxing {@code float[][]} de l'extraction. Le chemin batché utilise
 * désormais des buffers directs persistants (ordre natif) et des tenseurs d'entrée/sortie pré-créés
 * par palier — l'entrée est zéro-copie côté ORT, les sorties sont écrites par ORT directement dans
 * nos buffers (pinned outputs). Plus aucune allocation native par batch.
 */
public final class NetworkOnnx implements Network {

  /** Input layout AlphaZero : 119 plans 8×8 par position = 7616 floats. */
  public static final int INPUT_FLOATS_PER_POS = 119 * 8 * 8;

  /**
   * Output layout policy : 73 plans 8×8 par position = 4672 logits (idem
   * MoveEncoding.POLICY_INDICES).
   */
  public static final int POLICY_LOGITS_PER_POS = 4672;

  /** Output layout value : 3 logits WDL (Win/Draw/Loss) par position depuis v1.5.0 (ADR-012). */
  private static final int VALUE_LOGITS_PER_POS = 3;

  /**
   * Paliers de batch du chemin batché CUDA (ADR-013). Un batch de taille k est paddé (zéros) au
   * palier supérieur ; chaque palier est warmé au chargement. Croissant, dernier = capacité max.
   */
  static final int[] CUDA_BATCH_BUCKETS = {1, 4, 8, 16, 32, 64, 128, 256};

  /** Capacité maximale du chemin batché CUDA ({@code maxBatch()} en mode CUDA). */
  public static final int CUDA_MAX_BATCH = 256;

  /** Une position de zéros, source du re-zeroing de la zone de padding du buffer persistant. */
  private static final float[] ZERO_POSITION = new float[INPUT_FLOATS_PER_POS];

  private final OrtEnvironment env;
  private final OrtSession session;
  private final String inputName;
  private final String policyOutputName;
  private final String valueOutputName;
  private final NetworkMetadata metadata;
  private final BitboardPlaneEncoder encoder = BitboardPlaneEncoderVector.INSTANCE;

  /**
   * (v1.3.0 garde-fou) Sentinelle "forward en cours" pour les chemins non thread-safe : CUDA EP
   * (ORT CUDA EP n'est pas thread-safe pour {@code session.run()} concurrent — invocations
   * parallèles = comportement indéfini) et, depuis v1.6.0, tout chemin batché (les buffers IO
   * persistants sont mono-thread par construction). Détecte les appels concurrents et lève {@link
   * IllegalStateException} avec un message explicite (préférable à un crash silencieux). {@code
   * null} en mode CPU EP loop (thread-safe). Cf. SPEC-engine §16.4.1, ADR-017.
   */
  private final AtomicBoolean cudaForwardInFlight;

  /**
   * {@code true} si {@link #forward} route vers le chemin batché par paliers (CUDA EP en production
   * ; CPU EP possible via le seam de test {@link #loadCpuBatchedForTests}).
   */
  private final boolean batchedForward;

  /**
   * IO persistant d'un palier : tenseurs pré-créés wrappant des slices des buffers directs hôtes
   * (entrée zéro-copie, sorties pinnées). Les maps sont pré-construites pour un {@code session.run}
   * sans aucune allocation par batch.
   */
  private static final class BucketIo {
    final Map<String, OnnxTensor> inputs;
    final Map<String, OnnxTensor> pinnedOutputs;

    BucketIo(Map<String, OnnxTensor> inputs, Map<String, OnnxTensor> pinnedOutputs) {
      this.inputs = inputs;
      this.pinnedOutputs = pinnedOutputs;
    }
  }

  /** Buffer hôte direct (ordre natif) des planes d'entrée, capacité {@code CUDA_MAX_BATCH}. */
  private final FloatBuffer inputHost;

  /** Buffer hôte direct des logits policy, écrit par ORT (pinned output). */
  private final FloatBuffer policyHost;

  /** Buffer hôte direct des logits WDL, écrit par ORT (pinned output). */
  private final FloatBuffer valueHost;

  /** IO pré-créé par palier, indexé comme {@link #CUDA_BATCH_BUCKETS}. */
  private final BucketIo[] bucketIo;

  /** Scratch WDL (3 floats) du chemin batché — mono-thread (cf. sentinelle). */
  private final float[] wdlScratch = new float[VALUE_LOGITS_PER_POS];

  /**
   * High-water mark des positions écrites dans {@link #inputHost} : seules les positions de padding
   * encore sales ({@code [batchSize, min(dirty, bucket))}) sont re-zéroées à chaque batch — en
   * régime établi (K stable) le re-zeroing est quasi nul.
   */
  private int dirtyPositions;

  // Instrumentation fine du chemin batché (2026-06-11, enquête "run 32 ms back-to-back") : tout
  // est écrit/logué par l'unique thread appelant (cf. sentinelle) — pas de synchronisation.
  private static final java.util.logging.Logger BATCH_LOG =
      java.util.logging.Logger.getLogger(NetworkOnnx.class.getName());
  private static final int BATCH_LOG_EVERY = 512;
  private long statPutNanos;
  private long statRunNanos;
  private long statExtractNanos;
  private long statRunMinNanos = Long.MAX_VALUE;
  private long statRunMaxNanos;
  private int statCount;

  private NetworkOnnx(Path onnxPath, boolean useCuda, boolean batchedForward) {
    try {
      this.env = OrtEnvironment.getEnvironment();
      OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
      if (useCuda) {
        // use_tf32=0 — sur Ampere+ (RTX 3090), cuDNN/cuBLAS routent les convs FP32 vers les
        // tensor cores TF32 (mantisse 10 bits) par défaut : divergence logits jusqu'à 2,9 vs
        // CPU EP (mesure 2026-06-10, sonde CudaTf32Probe sur 3090). Sur le 8×96 le TF32
        // n'apporte AUCUN débit (22177 pos/s OFF vs 20791 ON @K=256) : le désactiver est un
        // gain pur (parité FP32 ≤3e-3 logits + ~7 %). Sans effet sur Pascal (pas de TF32).
        var cudaOpts = new ai.onnxruntime.providers.OrtCUDAProviderOptions(0);
        cudaOpts.add("use_tf32", "0");
        opts.addCUDA(cudaOpts);
        // (2026-06-11) Pool intra-op à 1 AUSSI en CUDA : le graphe 8×96 est 100 % GPU, mais le
        // pool par défaut (≥ #cores threads) parallélise les copies host-side ; sous charge
        // (worker multi-parties : storm de threads de jeu à chaque batch complété), UN retardataire
        // du fork-join préempté ~1-2 quantums Windows bloque tout le run() — mesuré : run 33 ms au
        // lieu de ~6-10 ms, indépendant de K/clocks/priorité du caller, sur w1080 ET w3090.
        opts.setIntraOpNumThreads(1);
        opts.setInterOpNumThreads(1);
      } else {
        // Phase 12 hotfix-011 — Limit ORT CPU EP threads to 1 per session.
        // Default ORT uses min(N_cores, 16) intra-op threads per session. With fastchess
        // SPRT -concurrency=4 → 8 ORT sessions concurrent × N_cores threads = severe
        // oversubscription. Symptom : 14-23% game time forfeits as MCTS first-sim per
        // ucinewgame takes 5-10× its 100ms budget under CPU contention.
        // Diagnostic 2026-05-16 : concurrency=1 → 1% timeouts ; concurrency=4 default
        // → 23% timeouts. With (1,1) per session, OS scheduler load-balances JVMs naturally.
        opts.setIntraOpNumThreads(1);
        opts.setInterOpNumThreads(1);
      }
      this.session = env.createSession(onnxPath.toString(), opts);
      this.inputName = session.getInputNames().iterator().next();
      var outputIter = session.getOutputNames().iterator();
      this.policyOutputName = outputIter.next();
      this.valueOutputName = outputIter.next();
    } catch (OrtException e) {
      throw new IllegalStateException("Failed to load ONNX model: " + onnxPath, e);
    }
    // Metadata "stub" pour Network interface — ONNX file ne porte pas les mêmes meta.
    // Si besoin, à étendre via parsing custom metadata côté ONNX export.
    this.metadata = new NetworkMetadata("onnx-runtime", "n/a", 0, "n/a", "alphazero-119");
    // (v1.3.0 garde-fou) Sentinelle pour CUDA EP et pour tout chemin batché (buffers IO
    // persistants mono-thread). CPU EP loop reste sans sentinelle (thread-safe).
    this.cudaForwardInFlight = (useCuda || batchedForward) ? new AtomicBoolean(false) : null;
    this.batchedForward = batchedForward;
    if (batchedForward) {
      this.inputHost = allocateHost(CUDA_MAX_BATCH * INPUT_FLOATS_PER_POS);
      this.policyHost = allocateHost(CUDA_MAX_BATCH * POLICY_LOGITS_PER_POS);
      this.valueHost = allocateHost(CUDA_MAX_BATCH * VALUE_LOGITS_PER_POS);
      this.bucketIo = new BucketIo[CUDA_BATCH_BUCKETS.length];
      try {
        for (int i = 0; i < CUDA_BATCH_BUCKETS.length; i++) {
          int bucket = CUDA_BATCH_BUCKETS[i];
          OnnxTensor input =
              OnnxTensor.createTensor(
                  env,
                  sliceOf(inputHost, bucket * INPUT_FLOATS_PER_POS),
                  new long[] {bucket, 119, 8, 8});
          OnnxTensor policy =
              OnnxTensor.createTensor(
                  env,
                  sliceOf(policyHost, bucket * POLICY_LOGITS_PER_POS),
                  new long[] {bucket, POLICY_LOGITS_PER_POS});
          OnnxTensor value =
              OnnxTensor.createTensor(
                  env,
                  sliceOf(valueHost, bucket * VALUE_LOGITS_PER_POS),
                  new long[] {bucket, VALUE_LOGITS_PER_POS});
          bucketIo[i] =
              new BucketIo(
                  Map.of(inputName, input),
                  Map.of(policyOutputName, policy, valueOutputName, value));
        }
      } catch (OrtException e) {
        throw new IllegalStateException("Failed to allocate batched IO tensors", e);
      }
      warmupBuckets();
    } else {
      this.inputHost = null;
      this.policyHost = null;
      this.valueHost = null;
      this.bucketIo = null;
    }
  }

  /** Alloue un buffer hôte direct en ordre natif (pré-zéroé par {@code allocateDirect}). */
  private static FloatBuffer allocateHost(int floats) {
    return ByteBuffer.allocateDirect(floats * Float.BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  /** Slice direct {@code [0, floats)} d'un buffer hôte (mémoire partagée, position 0). */
  private static FloatBuffer sliceOf(FloatBuffer host, int floats) {
    return host.duplicate().limit(floats).slice();
  }

  /**
   * Warmup du chemin batché : une inférence dummy (zéros) par palier de {@link
   * #CUDA_BATCH_BUCKETS}, deux passes. Force CUDA EP à compiler/sélectionner ses kernels pour
   * chaque shape AVANT le premier batch réel — l'ensemble des shapes est fini, plus aucun recompile
   * en régime établi (ADR-013). Coût one-shot au chargement (~quelques centaines de ms). Exerce
   * aussi le chemin pinned-outputs de chaque palier.
   */
  private void warmupBuckets() {
    try {
      for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < CUDA_BATCH_BUCKETS.length; i++) {
          session.run(bucketIo[i].inputs, bucketIo[i].pinnedOutputs).close();
        }
      }
    } catch (OrtException e) {
      throw new IllegalStateException("ONNX bucket warmup failed", e);
    }
  }

  /**
   * (v1.3.0) Garde-fou contre les appels concurrents à {@code forward} en mode CUDA EP ou batché.
   * ORT CUDA EP ne supporte pas le {@code session.run()} concurrent inter-thread, et les buffers IO
   * persistants du chemin batché sont mono-thread. En mode CPU EP loop, no-op (thread-safe).
   *
   * <p>Pattern recommandé pour utiliser CUDA EP en multi-thread MCTS : router tous les forwards via
   * un thread unique dédié (cf. {@code engine.internal.batched.NNEvalThread}, ou le thread runner
   * du {@link BatchingNetwork} côté worker GPU) en configurant {@code EngineConfig.batchSize > 1}.
   * Le crash explicite ici remplace le comportement indéfini de l'API ORT pour signaler une
   * mauvaise configuration à l'utilisateur.
   *
   * @throws IllegalStateException si un autre thread est déjà en train d'exécuter {@code forward()}
   *     sur cette instance CUDA.
   */
  private void acquireSingleThreadedForward() {
    if (cudaForwardInFlight != null && !cudaForwardInFlight.compareAndSet(false, true)) {
      throw new IllegalStateException(
          "NetworkOnnx CUDA EP requires single-threaded forward(). Concurrent invocation detected."
              + " If you need multi-threaded MCTS with CUDA, configure EngineConfig.batchSize > 1"
              + " to route all forwards through the dedicated NNEvalThread (cf. SPEC-engine"
              + " §16.4.1, ADR-017).");
    }
  }

  private void releaseSingleThreadedForward() {
    if (cudaForwardInFlight != null) {
      cudaForwardInFlight.set(false);
    }
  }

  /**
   * Charge un model ONNX avec CPU EP (default).
   *
   * @param onnxPath chemin vers le fichier .onnx
   * @return NetworkOnnx prêt à inférer
   * @throws IOException si le chargement échoue (wrapping IllegalStateException)
   */
  public static NetworkOnnx load(Path onnxPath) throws IOException {
    try {
      return new NetworkOnnx(onnxPath, false, false);
    } catch (IllegalStateException e) {
      throw new IOException("Failed to load ONNX model", e);
    }
  }

  /**
   * Charge un model ONNX avec CUDA EP (GPU device 0) et chemin batché par paliers (ADR-013).
   * Requiert build avec onnxruntime_gpu artifact + driver NVIDIA + DLLs CUDA 12.x + cuDNN 9.x dans
   * PATH. Le chargement inclut le warmup des paliers (~quelques centaines de ms).
   */
  public static NetworkOnnx loadCuda(Path onnxPath) throws IOException {
    try {
      return new NetworkOnnx(onnxPath, true, true);
    } catch (IllegalStateException e) {
      throw new IOException("Failed to load ONNX model with CUDA EP", e);
    }
  }

  /**
   * Seam de test : CPU EP avec le chemin batché par paliers activé. Permet de tester en CI (sans
   * GPU) la logique paliers/padding/extraction WDL du chemin batché — identique au mode CUDA, seul
   * l'execution provider diffère. NE PAS utiliser en production (le batching ORT ne paie pas sur
   * CPU, cf. PERF-GPU-BATCH-phase0.md).
   */
  static NetworkOnnx loadCpuBatchedForTests(Path onnxPath) throws IOException {
    try {
      return new NetworkOnnx(onnxPath, false, true);
    } catch (IllegalStateException e) {
      throw new IOException("Failed to load ONNX model (CPU batched test seam)", e);
    }
  }

  @Override
  public int maxBatch() {
    return batchedForward ? CUDA_MAX_BATCH : MAX_BATCH;
  }

  @Override
  public void forward(float[] planes, int batchSize, NNOutput output) {
    if (batchSize < MIN_BATCH || batchSize > maxBatch()) {
      throw new IllegalArgumentException(
          "batchSize " + batchSize + " out of [" + MIN_BATCH + ", " + maxBatch() + "]");
    }
    if (planes.length < batchSize * INPUT_FLOATS_PER_POS) {
      throw new IllegalArgumentException(
          "planes length " + planes.length + " < batchSize × " + INPUT_FLOATS_PER_POS);
    }
    if (output.capacity() < batchSize) {
      throw new IllegalArgumentException(
          "output capacity " + output.capacity() + " < batchSize " + batchSize);
    }
    acquireSingleThreadedForward();
    try {
      if (batchedForward) {
        forwardBatchedInternal(planes, batchSize, output);
      } else {
        forwardInternal(planes, batchSize, output);
      }
    } finally {
      releaseSingleThreadedForward();
    }
  }

  /**
   * Chemin batché par paliers (ADR-013) : UN {@code session.run} à shape {@code [bucket, 119, 8,
   * 8]} où {@code bucket} est le plus petit palier ≥ {@code batchSize}.
   *
   * <p>IO zéro-alloc (v1.6.0) : les positions réelles sont recopiées dans {@link #inputHost}
   * (persistant, wrappé zéro-copie par le tenseur d'entrée du palier) ; seule la zone de padding
   * encore sale est re-zéroée (high-water mark {@link #dirtyPositions}) — le buffer du caller n'est
   * jamais muté ni lu au-delà de {@code batchSize} positions. ORT écrit policy et value directement
   * dans {@link #policyHost} / {@link #valueHost} (pinned outputs) ; les sorties fantômes des
   * positions de padding sont ignorées.
   */
  private void forwardBatchedInternal(float[] planes, int batchSize, NNOutput output) {
    long t0 = System.nanoTime();
    int bucketIndex = bucketIndexFor(batchSize);
    int bucket = CUDA_BATCH_BUCKETS[bucketIndex];
    inputHost.clear();
    inputHost.put(planes, 0, batchSize * INPUT_FLOATS_PER_POS);
    if (dirtyPositions > batchSize) {
      int zeroTo = Math.min(dirtyPositions, bucket);
      for (int p = batchSize; p < zeroTo; p++) {
        inputHost.put(ZERO_POSITION, 0, INPUT_FLOATS_PER_POS);
      }
      // Si ce palier couvrait toute la zone sale, tout au-delà de batchSize est redevenu propre ;
      // sinon la zone [bucket, dirty) reste sale (hors de portée de CE run) et le mark est gardé.
      dirtyPositions = dirtyPositions > bucket ? dirtyPositions : batchSize;
    } else {
      dirtyPositions = batchSize;
    }
    long t1 = System.nanoTime();
    try {
      session.run(bucketIo[bucketIndex].inputs, bucketIo[bucketIndex].pinnedOutputs).close();
    } catch (OrtException e) {
      throw new IllegalStateException("ONNX batched forward failed", e);
    }
    long t2 = System.nanoTime();
    policyHost.clear();
    policyHost.get(output.logits, 0, batchSize * POLICY_LOGITS_PER_POS);
    valueHost.clear();
    for (int b = 0; b < batchSize; b++) {
      valueHost.get(wdlScratch, 0, VALUE_LOGITS_PER_POS);
      output.values[b] = wdlToValue(wdlScratch);
    }
    long t3 = System.nanoTime();
    statPutNanos += t1 - t0;
    long run = t2 - t1;
    statRunNanos += run;
    statExtractNanos += t3 - t2;
    if (run < statRunMinNanos) {
      statRunMinNanos = run;
    }
    if (run > statRunMaxNanos) {
      statRunMaxNanos = run;
    }
    if (++statCount >= BATCH_LOG_EVERY) {
      final int n = statCount;
      final double putUs = statPutNanos / 1e3 / n;
      final double runMs = statRunNanos / 1e6 / n;
      final double extractUs = statExtractNanos / 1e3 / n;
      final double runMinMs = statRunMinNanos / 1e6;
      final double runMaxMs = statRunMaxNanos / 1e6;
      BATCH_LOG.info(
          () ->
              String.format(
                  "onnx-batched: %d runs | put %.0f µs | session.run %.2f ms (min %.2f, max %.2f)"
                      + " | extract %.0f µs",
                  n, putUs, runMs, runMinMs, runMaxMs, extractUs));
      statPutNanos = 0;
      statRunNanos = 0;
      statExtractNanos = 0;
      statRunMinNanos = Long.MAX_VALUE;
      statRunMaxNanos = 0;
      statCount = 0;
    }
  }

  /**
   * Retourne le plus petit palier de {@link #CUDA_BATCH_BUCKETS} ≥ {@code batchSize}.
   *
   * @throws IllegalArgumentException si {@code batchSize} hors {@code [1, CUDA_MAX_BATCH]}
   */
  static int bucketFor(int batchSize) {
    return CUDA_BATCH_BUCKETS[bucketIndexFor(batchSize)];
  }

  /** Index dans {@link #CUDA_BATCH_BUCKETS} du plus petit palier ≥ {@code batchSize}. */
  static int bucketIndexFor(int batchSize) {
    if (batchSize < 1 || batchSize > CUDA_MAX_BATCH) {
      throw new IllegalArgumentException(
          "batchSize " + batchSize + " out of [1, " + CUDA_MAX_BATCH + "]");
    }
    for (int i = 0; i < CUDA_BATCH_BUCKETS.length; i++) {
      if (CUDA_BATCH_BUCKETS[i] >= batchSize) {
        return i;
      }
    }
    throw new AssertionError("unreachable: CUDA_BATCH_BUCKETS must end at CUDA_MAX_BATCH");
  }

  private void forwardInternal(float[] planes, int batchSize, NNOutput output) {
    // Chemin CPU EP : loop K×forward(1). Le batching ORT ne paie pas sur CPU (courbe plate
    // ~350-430 pos/s quel que soit K, cf. PERF-GPU-BATCH-phase0.md) ; la boucle est stable et
    // thread-safe. Le chemin batché par paliers (forwardBatchedInternal, ADR-013) est réservé
    // au CUDA EP, où le forward(K) à shape libre s'effondrait par recompile de kernels
    // (bench 2026-05-20 : 62-408 nps vs 845 en loop — résolu par les paliers fixes + warmup).
    try {
      for (int b = 0; b < batchSize; b++) {
        long[] shape = {1, 119, 8, 8};
        float[] singlePlanes = new float[INPUT_FLOATS_PER_POS];
        System.arraycopy(planes, b * INPUT_FLOATS_PER_POS, singlePlanes, 0, INPUT_FLOATS_PER_POS);
        try (OnnxTensor input =
            OnnxTensor.createTensor(env, FloatBuffer.wrap(singlePlanes), shape)) {
          Map<String, OnnxTensor> inputs = new HashMap<>();
          inputs.put(inputName, input);
          try (OrtSession.Result results = session.run(inputs)) {
            float[][] policyOut =
                (float[][]) results.get(policyOutputName).orElseThrow().getValue();
            // Value head WDL v1.5.0 : sortie [1, 3] (Win/Draw/Loss logits).
            float[][] valueOut = (float[][]) results.get(valueOutputName).orElseThrow().getValue();
            System.arraycopy(
                policyOut[0], 0, output.logits, b * POLICY_LOGITS_PER_POS, POLICY_LOGITS_PER_POS);
            output.values[b] = wdlToValue(valueOut[0]);
          }
        }
      }
    } catch (OrtException e) {
      throw new IllegalStateException("ONNX forward failed", e);
    }
  }

  @Override
  public NNSingleResult forwardSingle(GameState state) {
    acquireSingleThreadedForward();
    try {
      return forwardSingleInternal(state);
    } finally {
      releaseSingleThreadedForward();
    }
  }

  private NNSingleResult forwardSingleInternal(GameState state) {
    float[] planes = new float[INPUT_FLOATS_PER_POS];
    state.toPlanes(planes, 0, encoder);
    try {
      long[] shape = {1, 119, 8, 8};
      try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(planes), shape)) {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, input);
        try (OrtSession.Result results = session.run(inputs)) {
          float[][] policyOut = (float[][]) results.get(policyOutputName).orElseThrow().getValue();
          float[][] valueOut = (float[][]) results.get(valueOutputName).orElseThrow().getValue();
          return new NNSingleResult(policyOut[0].clone(), wdlToValue(valueOut[0]));
        }
      }
    } catch (OrtException e) {
      throw new IllegalStateException("ONNX forwardSingle failed", e);
    }
  }

  /**
   * Convertit les 3 logits WDL (Win/Draw/Loss) en value scalaire {@code P(W) - P(L)} ∈ [-1, 1] via
   * un softmax numériquement stable. Identique au calcul du chemin SIMD {@link NetworkVectorApi}
   * (v1.5.0).
   *
   * @param wdl logits bruts {@code [W, D, L]}
   * @return value scalaire pour le PUCT
   */
  private static float wdlToValue(float[] wdl) {
    float w = wdl[0];
    float d = wdl[1];
    float l = wdl[2];
    float max = Math.max(w, Math.max(d, l));
    float ew = (float) Math.exp(w - max);
    float ed = (float) Math.exp(d - max);
    float el = (float) Math.exp(l - max);
    return (ew - el) / (ew + ed + el);
  }

  @Override
  public NetworkMetadata metadata() {
    return metadata;
  }

  @Override
  public BitboardPlaneEncoder planeEncoder() {
    return encoder;
  }
}
