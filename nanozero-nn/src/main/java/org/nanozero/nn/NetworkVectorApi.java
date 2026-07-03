package org.nanozero.nn;

import java.util.Arrays;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;
import org.nanozero.nn.kernels.Activations;
import org.nanozero.nn.kernels.Conv2D1x1;
import org.nanozero.nn.kernels.Conv2D3x3;
import org.nanozero.nn.kernels.Linear;
import org.nanozero.nn.kernels.Skip;

/**
 * Réseau de neurones AlphaZero ResNet 8×96 chargé en mémoire (cf. SPEC §4.2.1, §5.1).
 *
 * <p>Architecture figée (§3.3) : input conv 119→96 + 8 blocs résiduels (96→96→96 chacun avec
 * skip-add) + policy head (conv 1×1 96→73) + value head WDL v1.5.0 (conv 1×1 96→1, FC 64→64, FC
 * 64→3 Win/Draw/Loss → softmax → value scalaire = P(W)−P(L)). Détails algorithmiques de {@link
 * #forward(float[], int, NNOutput)} en §5.1.2.
 *
 * <p><strong>Invariants</strong> :
 *
 * <ul>
 *   <li>{@code I-Net-1} : immutable une fois construit. Les poids et le foncteur d'encoder sont
 *       partagés en lecture seule entre tous les threads.
 *   <li>{@code I-Net-2} : {@link #forward} est thread-confiné côté activations via {@link
 *       ThreadLocal} interne. Chaque thread a son propre buffer scratch (~4.7 MB par thread, lazy
 *       au premier appel).
 * </ul>
 *
 * <p>Le constructeur public n'est PAS exposé : utiliser {@code NetworkLoader.load} (phase 8). Un
 * constructeur package-private est fourni pour les tests phase 7 (poids déjà reorderés).
 */
public final class NetworkVectorApi implements Network {

  // Constants déplacés vers interface Network (MAX_BATCH, MIN_BATCH).
  // Re-référencés ici via Network.* dans le code below pour compat.

  /** Nombre de channels d'entrée du réseau (format AlphaZero 119 plans). */
  static final int INPUT_CHANNELS = 119;

  /** Nombre de channels dans la tour résiduelle (cf. NetworkConfig.CHANNELS = 96). */
  static final int CHANNELS = 96;

  /** Hauteur × largeur des plans (8 × 8 = 64). */
  static final int HW = 64;

  /** Nombre de plans dans la sortie policy (cf. NetworkConfig.POLICY_PLANES = 73). */
  static final int POLICY_PLANES = 73;

  /** Channels intermédiaires du value head après conv 1×1 (= 1). */
  static final int VALUE_HEAD_CHANNELS = 1;

  /** Taille de la couche cachée FC du value head (= 64). */
  static final int VALUE_HEAD_HIDDEN = 64;

  /** Nombre de classes WDL du value head (Win/Draw/Loss, = 3) — v1.5.0. */
  static final int VALUE_WDL_CLASSES = 3;

  /** Nombre de blocs résiduels (= 8). */
  static final int NB_RESIDUAL_BLOCKS = 8;

  // ---------------------------------------------------------------------------------------------
  // Champs immutables (poids reorderés + bias par couche)
  // ---------------------------------------------------------------------------------------------

  private final NetworkMetadata metadata;

  /** Poids reorderés via {@code WeightsLayout.reorderConv3x3}. */
  final float[] inputConvWeights;

  final float[] inputConvBias;

  /** Poids des conv1 de chaque bloc résiduel, reorderés. */
  final float[][] blockConv1Weights;

  final float[][] blockConv1Bias;

  /** Poids des conv2 de chaque bloc résiduel, reorderés. */
  final float[][] blockConv2Weights;

  final float[][] blockConv2Bias;

  /** Poids du policy head conv 1×1 (96 → 73), layout row-major NON reorderé. */
  final float[] policyConvWeights;

  final float[] policyConvBias;

  /** Poids du value head conv 1×1 (96 → 1), layout row-major NON reorderé. */
  final float[] valueConvWeights;

  final float[] valueConvBias;

  /** Poids de value_fc1 (64 → 64), layout row-major. */
  final float[] valueFc1Weights;

  final float[] valueFc1Bias;

  /** Poids de value_fc2 (64 → 3 WDL v1.5.0), layout row-major [3, 64]. */
  final float[] valueFc2Weights;

  final float[] valueFc2Bias;

  /** Encoder Vector API singleton. */
  private final BitboardPlaneEncoder encoder = BitboardPlaneEncoderVector.INSTANCE;

  /**
   * Buffer scratch par thread pour le double buffering ping-pong (§5.1.3) : 3 buffers de {@code
   * MAX_BATCH × CHANNELS × HW = 64 × 96 × 64 = 393 216 floats ≈ 1.57 MB} chacun, total ~4.7 MB par
   * thread.
   */
  private static final class Scratch {
    final float[] bufferA = new float[MAX_BATCH * CHANNELS * HW];
    final float[] bufferB = new float[MAX_BATCH * CHANNELS * HW];
    final float[] tmp = new float[MAX_BATCH * CHANNELS * HW];
  }

  private final ThreadLocal<Scratch> scratch = ThreadLocal.withInitial(Scratch::new);

  /** Buffer planes thread-local pour {@link #forwardSingle}. ~1.86 MB par thread. */
  private final ThreadLocal<float[]> singlePlanes =
      ThreadLocal.withInitial(() -> new float[MAX_BATCH * INPUT_CHANNELS * HW]);

  /** {@link NNOutput} thread-local pour {@link #forwardSingle}. ~1.2 MB par thread. */
  private final ThreadLocal<NNOutput> singleOutput = ThreadLocal.withInitial(NNOutput::new);

  // ---------------------------------------------------------------------------------------------
  // Constructeur package-private (phase 7 : utilisé par NetworkTest ; phase 8 : NetworkLoader)
  // ---------------------------------------------------------------------------------------------

  NetworkVectorApi(
      NetworkMetadata metadata,
      float[] inputConvWeights,
      float[] inputConvBias,
      float[][] blockConv1Weights,
      float[][] blockConv1Bias,
      float[][] blockConv2Weights,
      float[][] blockConv2Bias,
      float[] policyConvWeights,
      float[] policyConvBias,
      float[] valueConvWeights,
      float[] valueConvBias,
      float[] valueFc1Weights,
      float[] valueFc1Bias,
      float[] valueFc2Weights,
      float[] valueFc2Bias) {
    this.metadata = metadata;
    this.inputConvWeights = inputConvWeights;
    this.inputConvBias = inputConvBias;
    this.blockConv1Weights = blockConv1Weights;
    this.blockConv1Bias = blockConv1Bias;
    this.blockConv2Weights = blockConv2Weights;
    this.blockConv2Bias = blockConv2Bias;
    this.policyConvWeights = policyConvWeights;
    this.policyConvBias = policyConvBias;
    this.valueConvWeights = valueConvWeights;
    this.valueConvBias = valueConvBias;
    this.valueFc1Weights = valueFc1Weights;
    this.valueFc1Bias = valueFc1Bias;
    this.valueFc2Weights = valueFc2Weights;
    this.valueFc2Bias = valueFc2Bias;
  }

  // ---------------------------------------------------------------------------------------------
  // API publique
  // ---------------------------------------------------------------------------------------------

  /** Retourne l'encoder Vector API (cf. §4.2.1). */
  public BitboardPlaneEncoder planeEncoder() {
    return encoder;
  }

  /** Retourne les métadonnées du modèle chargé (cf. §4.2.1). */
  public NetworkMetadata metadata() {
    return metadata;
  }

  /**
   * Exécute le forward pass sur un batch de positions (cf. SPEC §5.1.2, §5.1.3). Zéro alloc en hot
   * path après l'init lazy du {@link ThreadLocal} scratch au premier appel par thread.
   *
   * @param planes buffer de planes d'entrée, longueur exacte {@code MAX_BATCH × 119 × 64}. Seuls
   *     les {@code batchSize} premiers échantillons sont lus.
   * @param batchSize nombre de positions effectivement à inférer ({@code 1..MAX_BATCH})
   * @param output buffer de sortie pré-alloué, rempli en place ; les indices au-delà de {@code
   *     batchSize} ne sont pas garantis touchés
   * @throws IllegalArgumentException si dimensions ou {@code batchSize} hors plage
   */
  public void forward(float[] planes, int batchSize, NNOutput output) {
    validateForwardArgs(planes, batchSize, output);

    Scratch s = scratch.get();
    float[] bufferA = s.bufferA;
    float[] bufferB = s.bufferB;
    float[] tmp = s.tmp;

    int featureSize = batchSize * CHANNELS * HW;

    // 1. Input conv 119 → 96 + ReLU. Sortie dans bufferA.
    Conv2D3x3.applyBatch(
        planes, inputConvWeights, inputConvBias, bufferA, INPUT_CHANNELS, CHANNELS, batchSize);
    Activations.reluInPlace(bufferA, featureSize);

    // 2. Tour résiduelle (8 blocs ping-pong A↔B, scratch dans tmp).
    for (int i = 0; i < NB_RESIDUAL_BLOCKS; i++) {
      float[] src = (i & 1) == 0 ? bufferA : bufferB;
      float[] dst = (i & 1) == 0 ? bufferB : bufferA;

      // tmp = ReLU(conv3x3(src, conv1.W, conv1.b))
      Conv2D3x3.applyBatch(
          src, blockConv1Weights[i], blockConv1Bias[i], tmp, CHANNELS, CHANNELS, batchSize);
      Activations.reluInPlace(tmp, featureSize);

      // dst = conv3x3(tmp, conv2.W, conv2.b)
      Conv2D3x3.applyBatch(
          tmp, blockConv2Weights[i], blockConv2Bias[i], dst, CHANNELS, CHANNELS, batchSize);

      // dst = ReLU(dst + src)
      Skip.addInPlace(dst, src, featureSize);
      Activations.reluInPlace(dst, featureSize);
    }
    // Après 8 itérations (8 pair → dst final = bufferA), activations finales dans bufferA.
    float[] activations = bufferA;

    // 3. Policy head : conv 1×1 (96 → 73). Sortie NCHW [N, 73, 64] dans tmp.
    Conv2D1x1.applyBatch(
        activations, policyConvWeights, policyConvBias, tmp, CHANNELS, POLICY_PLANES, batchSize);

    // 3bis. Transposition NCHW [N, 73, 64] → format §3.5.1 [N, fromSquare * 73 + plane] dans
    //       output.logits. Boucles scalaires, ~300k éléments par batch — 0.1% du forward.
    transposeNCHWtoLogits(tmp, output.logits, batchSize);

    // 4. Value head : conv 1×1 (96 → 1) → bufferB (vu comme [N, 64]).
    Conv2D1x1.applyBatch(
        activations,
        valueConvWeights,
        valueConvBias,
        bufferB,
        CHANNELS,
        VALUE_HEAD_CHANNELS,
        batchSize);
    // valueFlat[N × 64] = bufferB (1 channel × 8 × 8 = 64).

    // valueHidden[N × 64] = ReLU(linear(valueFlat, fc1)) → tmp.
    Linear.applyBatch(
        bufferB,
        valueFc1Weights,
        valueFc1Bias,
        tmp,
        VALUE_HEAD_HIDDEN,
        VALUE_HEAD_HIDDEN,
        batchSize);
    Activations.reluInPlace(tmp, batchSize * VALUE_HEAD_HIDDEN);

    // valueLogits[N × 3] = linear(valueHidden, fc2) → bufferB (WDL v1.5.0).
    // bufferB est libre ici (valueFlat consommé par fc1) et largement assez grand.
    Linear.applyBatch(
        tmp,
        valueFc2Weights,
        valueFc2Bias,
        bufferB,
        VALUE_HEAD_HIDDEN,
        VALUE_WDL_CLASSES,
        batchSize);

    // values = P(Win) - P(Loss) via softmax(W,D,L) numériquement stable.
    // On expose un scalaire ∈ [-1, 1] au PUCT (engine inchangé) ; P(Draw) reste
    // récupérable via les logits pour un éventuel contempt UCI ultérieur.
    for (int n = 0; n < batchSize; n++) {
      int o = n * VALUE_WDL_CLASSES;
      float w = bufferB[o];
      float d = bufferB[o + 1];
      float l = bufferB[o + 2];
      float max = Math.max(w, Math.max(d, l));
      float ew = (float) Math.exp(w - max);
      float ed = (float) Math.exp(d - max);
      float el = (float) Math.exp(l - max);
      output.values[n] = (ew - el) / (ew + ed + el);
    }
  }

  /**
   * Commodité pour inférer une seule position. Allouante par appel : NE PAS UTILISER en hot path
   * MCTS (cf. SPEC §4.2.1). Buffers internes thread-local lazy.
   *
   * @param state position à évaluer
   * @return {@link NNSingleResult} avec logits (copie indépendante) et value
   */
  public NNSingleResult forwardSingle(GameState state) {
    float[] planes = singlePlanes.get();
    NNOutput out = singleOutput.get();
    // Remplit les 119 × 64 = 7616 premiers floats de planes ; le reste n'est pas lu (batchSize=1).
    state.toPlanes(planes, 0, encoder);
    forward(planes, 1, out);
    float[] logitsCopy = Arrays.copyOfRange(out.logits, 0, MoveEncoding.POLICY_INDICES);
    return new NNSingleResult(logitsCopy, out.values[0]);
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers package-private (testables en isolation)
  // ---------------------------------------------------------------------------------------------

  /**
   * Transpose NCHW {@code [batchSize, POLICY_PLANES=73, HW=64]} (sortie de Conv2D1x1) vers le
   * format §3.5.1 {@code [batchSize, fromSquare * POLICY_PLANES + planeIndex]} (= 4672 logits par
   * sample).
   *
   * <p>{@code dst[n*4672 + square*73 + plane] = src[n*73*64 + plane*64 + square]}.
   *
   * <p>Méthode package-private isolée pour permettre un test paramétrique direct sur la signature
   * de transposition (point chaud silencieux : une inversion d'axes corrompt l'apprentissage en
   * phase 9 sans erreur visible avant).
   *
   * @param src buffer NCHW {@code [batchSize × POLICY_PLANES × HW]}
   * @param dst buffer plat {@code [batchSize × POLICY_INDICES]} (rempli en place)
   * @param batchSize nombre de samples
   */
  static void transposeNCHWtoLogits(float[] src, float[] dst, int batchSize) {
    for (int n = 0; n < batchSize; n++) {
      int srcBase = n * POLICY_PLANES * HW;
      int dstBase = n * MoveEncoding.POLICY_INDICES;
      for (int square = 0; square < HW; square++) {
        for (int plane = 0; plane < POLICY_PLANES; plane++) {
          dst[dstBase + square * POLICY_PLANES + plane] = src[srcBase + plane * HW + square];
        }
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------------------------

  private static void validateForwardArgs(float[] planes, int batchSize, NNOutput output) {
    if (planes == null) {
      throw new IllegalArgumentException("planes must not be null");
    }
    int expectedPlanesLength = MAX_BATCH * INPUT_CHANNELS * HW;
    if (planes.length != expectedPlanesLength) {
      throw new IllegalArgumentException(
          "planes length must be " + expectedPlanesLength + ", got " + planes.length);
    }
    if (batchSize < MIN_BATCH || batchSize > MAX_BATCH) {
      throw new IllegalArgumentException(
          "batchSize " + batchSize + " out of [" + MIN_BATCH + ", " + MAX_BATCH + "]");
    }
    if (output == null) {
      throw new IllegalArgumentException("output must not be null");
    }
    int expectedLogitsLength = MAX_BATCH * MoveEncoding.POLICY_INDICES;
    if (output.logits.length != expectedLogitsLength) {
      throw new IllegalArgumentException(
          "output.logits length must be " + expectedLogitsLength + ", got " + output.logits.length);
    }
    if (output.values.length != MAX_BATCH) {
      throw new IllegalArgumentException(
          "output.values length must be " + MAX_BATCH + ", got " + output.values.length);
    }
  }
}
