package org.nanozero.nn;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;
import org.nanozero.nn.internal.NpzReader;
import org.nanozero.nn.internal.NpzReader.NpzData;
import org.nanozero.nn.internal.WeightsLayout;

/**
 * Charge un {@link Network} depuis un fichier {@code .npz} produit par {@code
 * numpy.savez_compressed} (cf. SPEC §4.2.2, §5.2, §6).
 *
 * <p>Effectue dans l'ordre les 8 étapes de validation §6.6 :
 *
 * <ol>
 *   <li>Ouverture du ZIP {@code .npz} ({@link IOException} si fail)
 *   <li>Présence et valeur stricte de {@code _meta_architecture_version = "resnet8x96-v1"}
 *   <li>Présence et valeur stricte de {@code _meta_input_plane_format = "alphazero-119"}
 *   <li>Présence des trois autres méta {@code _meta_model_hash}, {@code _meta_training_step},
 *       {@code _meta_export_date}
 *   <li>Présence + shape exact + dtype float32 des 42 tenseurs §6.3
 *   <li>Sanity values §5.2.2 : aucun NaN, aucun Inf, magnitude {@code |w| < 100} sur tout poids
 *   <li>Vérification du hash SHA-256 (cf. §6.5) si {@code options.verifyHash() == true} (défaut)
 *   <li>Reorder des 17 tenseurs Conv 3×3 via {@link WeightsLayout} et construction de l'instance
 *       {@link Network}
 * </ol>
 *
 * <p>Toute déviation lève {@link IllegalArgumentException} avec localisation précise du tenseur
 * fautif et message explicite. L'ordre garantit que les erreurs les plus rapidement détectables
 * (mauvaise architecture) soient remontées en premier.
 *
 * <p><strong>Hash convention</strong> (cf. §6.5) : SHA-256 sur la concaténation des bytes float32
 * little-endian des <strong>42 tenseurs de poids exclusivement</strong> (les noms commençant par
 * {@code _meta_} sont exclus du hash), triés alphabétiquement par nom. Format hex lowercase.
 * Reproduit bit-pour-bit le calcul Python {@code numpy.tobytes()} sur descripteur {@code <f4} sur
 * x86.
 *
 * <p>Exemple d'usage :
 *
 * <pre>{@code
 * Network net = NetworkLoader.load(Path.of("model.npz"));
 * Network netFastBoot = NetworkLoader.load(Path.of("model.npz"), LoadOptions.skipHashCheck());
 * }</pre>
 */
public final class NetworkLoader {

  // ---------------------------------------------------------------------------------------------
  // Constantes du contrat §6.3 (à maintenir alignées si l'architecture évolue ; figée §3.3 v1).
  // ---------------------------------------------------------------------------------------------

  static final String EXPECTED_ARCH_VERSION = "resnet8x96-v1";
  static final String EXPECTED_INPUT_PLANE_FORMAT = "alphazero-119";

  static final String META_ARCH_VERSION = "_meta_architecture_version";
  static final String META_INPUT_PLANE_FORMAT = "_meta_input_plane_format";
  static final String META_MODEL_HASH = "_meta_model_hash";
  static final String META_TRAINING_STEP = "_meta_training_step";
  static final String META_EXPORT_DATE = "_meta_export_date";

  /** Magnitude maximale absolue tolérée pour un poids (cf. §5.2.2). */
  static final float MAX_ABS_WEIGHT = 100.0f;

  /**
   * Liste figée des 42 tenseurs de poids attendus (§6.3) avec leur shape canonique. Ordre
   * d'insertion {@link LinkedHashMap} = ordre §6.3.1 → §6.3.4 pour faciliter le diagnostic. Ce
   * n'est PAS l'ordre de hash (qui utilise un tri alphabétique).
   */
  static final Map<String, int[]> EXPECTED_TENSORS = buildExpectedTensors();

  private static Map<String, int[]> buildExpectedTensors() {
    Map<String, int[]> m = new LinkedHashMap<>();
    // §6.3.1 input conv.
    m.put("input_conv.weight", new int[] {96, 119, 3, 3});
    m.put("input_conv.bias", new int[] {96});
    // §6.3.2 tour résiduelle (8 blocs × 4 tenseurs).
    for (int i = 0; i < 8; i++) {
      m.put("block_" + i + ".conv1.weight", new int[] {96, 96, 3, 3});
      m.put("block_" + i + ".conv1.bias", new int[] {96});
      m.put("block_" + i + ".conv2.weight", new int[] {96, 96, 3, 3});
      m.put("block_" + i + ".conv2.bias", new int[] {96});
    }
    // §6.3.3 policy head.
    m.put("policy_head.conv.weight", new int[] {73, 96, 1, 1});
    m.put("policy_head.conv.bias", new int[] {73});
    // §6.3.4 value head.
    m.put("value_head.conv.weight", new int[] {1, 96, 1, 1});
    m.put("value_head.conv.bias", new int[] {1});
    m.put("value_head.fc1.weight", new int[] {64, 64});
    m.put("value_head.fc1.bias", new int[] {64});
    m.put("value_head.fc2.weight", new int[] {3, 64}); // WDL v1.5.0 (Win/Draw/Loss)
    m.put("value_head.fc2.bias", new int[] {3});
    if (m.size() != 42) {
      throw new AssertionError("EXPECTED_TENSORS doit contenir 42 entrées, en a " + m.size());
    }
    return Map.copyOf(m);
  }

  private NetworkLoader() {
    throw new AssertionError("Non-instantiable");
  }

  // ---------------------------------------------------------------------------------------------
  // API publique
  // ---------------------------------------------------------------------------------------------

  /**
   * Charge un {@link Network} avec les options par défaut (hash check activé, cf. §6.6 étape 8).
   *
   * @param path chemin vers le fichier {@code .npz}
   * @return réseau chargé, prêt à inférer
   * @throws IOException si la lecture du ZIP échoue (fichier absent, ZIP corrompu, etc.)
   * @throws IllegalArgumentException si une étape §6.6 échoue (header invalide, tenseur manquant ou
   *     mal dimensionné, valeur NaN/Inf/hors magnitude, hash mismatch)
   */
  public static Network load(Path path) throws IOException {
    return load(path, LoadOptions.defaults());
  }

  /**
   * Charge un {@link Network} en dispatchant sur l'extension du fichier.
   *
   * <ul>
   *   <li>{@code .npz} : {@link NetworkVectorApi} via Vector API SIMD CPU (legacy v1.0.0).
   *   <li>{@code .onnx} : {@link NetworkOnnx} via ONNX Runtime Java (v1.1.0+ Phase 12 PoC).
   * </ul>
   *
   * @param path chemin vers le fichier model
   * @return réseau chargé (interface polymorphe)
   * @throws IOException si la lecture échoue
   */
  public static Network loadAuto(Path path) throws IOException {
    String name = path.getFileName().toString().toLowerCase();
    if (name.endsWith(".onnx")) {
      return NetworkOnnx.load(path);
    }
    return load(path);
  }

  /**
   * Charge un {@link Network} avec options explicites (cf. {@link LoadOptions}).
   *
   * @param path chemin vers le fichier {@code .npz}
   * @param options options de chargement
   * @return réseau chargé
   * @throws IOException si la lecture du ZIP échoue
   * @throws IllegalArgumentException si une étape §6.6 échoue
   */
  public static Network load(Path path, LoadOptions options) throws IOException {
    // Étape 1 : ouverture ZIP. NpzReader lève IOException si fichier absent / ZIP corrompu /
    // format .npy invalide ; aucune validation sémantique faite ici.
    NpzData data = NpzReader.read(path);

    // Étapes 2-5 : header _meta_*.
    validateMetadata(data);

    // Étape 6 : présence + shape + dtype des 42 tenseurs §6.3.
    Map<String, float[]> floats = data.floatTensors();
    Map<String, int[]> shapes = data.tensorShapes();
    for (var entry : EXPECTED_TENSORS.entrySet()) {
      validateTensorPresenceAndShape(floats, shapes, entry.getKey(), entry.getValue());
    }

    // Étape 7 : sanity NaN / Inf / |w| < 100 (fail-fast par tenseur).
    for (String name : EXPECTED_TENSORS.keySet()) {
      validateValues(name, floats.get(name));
    }

    // Étape 8 : hash check si activé.
    if (options.verifyHash()) {
      String expected = data.stringScalars().get(META_MODEL_HASH);
      verifyHash(floats, expected);
    }

    // Reorder Conv 3×3 (input_conv + 16 conv résiduelles) ; Conv 1×1 et FC restent row-major.
    NetworkMetadata metadata =
        new NetworkMetadata(
            data.stringScalars().get(META_ARCH_VERSION),
            data.stringScalars().get(META_MODEL_HASH),
            data.int64Tensors().get(META_TRAINING_STEP)[0],
            data.stringScalars().get(META_EXPORT_DATE),
            data.stringScalars().get(META_INPUT_PLANE_FORMAT));

    float[] inputConvW = WeightsLayout.reorderConv3x3(floats.get("input_conv.weight"), 119, 96);
    float[][] block1W = new float[8][];
    float[][] block1B = new float[8][];
    float[][] block2W = new float[8][];
    float[][] block2B = new float[8][];
    for (int i = 0; i < 8; i++) {
      block1W[i] = WeightsLayout.reorderConv3x3(floats.get("block_" + i + ".conv1.weight"), 96, 96);
      block1B[i] = floats.get("block_" + i + ".conv1.bias");
      block2W[i] = WeightsLayout.reorderConv3x3(floats.get("block_" + i + ".conv2.weight"), 96, 96);
      block2B[i] = floats.get("block_" + i + ".conv2.bias");
    }

    return new NetworkVectorApi(
        metadata,
        inputConvW,
        floats.get("input_conv.bias"),
        block1W,
        block1B,
        block2W,
        block2B,
        floats.get("policy_head.conv.weight"),
        floats.get("policy_head.conv.bias"),
        floats.get("value_head.conv.weight"),
        floats.get("value_head.conv.bias"),
        floats.get("value_head.fc1.weight"),
        floats.get("value_head.fc1.bias"),
        floats.get("value_head.fc2.weight"),
        floats.get("value_head.fc2.bias"));
  }

  // ---------------------------------------------------------------------------------------------
  // Étapes de validation, package-private pour testabilité directe
  // ---------------------------------------------------------------------------------------------

  /**
   * Étapes 2-5 §6.6. Vérifie présence et valeur stricte des deux méta figées + présence (sans
   * validation de valeur) des trois autres.
   *
   * @apiNote Internal — exposed for testing.
   */
  static void validateMetadata(NpzData data) {
    Map<String, String> strings = data.stringScalars();
    Map<String, long[]> ints = data.int64Tensors();

    // Étape 2.
    String arch = strings.get(META_ARCH_VERSION);
    if (arch == null) {
      throw new IllegalArgumentException("Missing " + META_ARCH_VERSION);
    }
    if (!EXPECTED_ARCH_VERSION.equals(arch)) {
      throw new IllegalArgumentException("Incompatible architecture: " + arch);
    }

    // Étape 3.
    String inputFormat = strings.get(META_INPUT_PLANE_FORMAT);
    if (inputFormat == null) {
      throw new IllegalArgumentException("Missing " + META_INPUT_PLANE_FORMAT);
    }
    if (!EXPECTED_INPUT_PLANE_FORMAT.equals(inputFormat)) {
      throw new IllegalArgumentException("Incompatible input plane format: " + inputFormat);
    }

    // Étape 4 : présence des 3 autres méta (valeurs non validées).
    if (strings.get(META_MODEL_HASH) == null) {
      throw new IllegalArgumentException("Missing " + META_MODEL_HASH);
    }
    long[] step = ints.get(META_TRAINING_STEP);
    if (step == null || step.length != 1) {
      throw new IllegalArgumentException(
          "Missing or malformed " + META_TRAINING_STEP + " (expected int64 scalar)");
    }
    if (strings.get(META_EXPORT_DATE) == null) {
      throw new IllegalArgumentException("Missing " + META_EXPORT_DATE);
    }
  }

  /**
   * Étape 6 §6.6 pour un tenseur : présence dans la map float, shape exact, dtype float32 garanti
   * par {@code NpzReader.floatTensors()}.
   *
   * @apiNote Internal — exposed for testing.
   */
  static void validateTensorPresenceAndShape(
      Map<String, float[]> floats, Map<String, int[]> shapes, String name, int[] expectedShape) {
    int[] shape = shapes.get(name);
    if (shape == null) {
      throw new IllegalArgumentException("Missing tensor: " + name);
    }
    float[] data = floats.get(name);
    if (data == null) {
      // Le tenseur existe (shape présent dans tensorShapes) mais pas dans floatTensors → mauvais
      // dtype (int64, string, etc.). NpzReader expose le dtype implicitement par la map cible.
      throw new IllegalArgumentException("Wrong dtype for " + name + " (expected float32 / '<f4')");
    }
    if (!Arrays.equals(shape, expectedShape)) {
      throw new IllegalArgumentException(
          "Wrong shape for "
              + name
              + ": expected "
              + Arrays.toString(expectedShape)
              + ", got "
              + Arrays.toString(shape));
    }
  }

  /**
   * Étape 7 §6.6 / §5.2.2 : aucun NaN, aucun Inf, |w| {@code <} {@link #MAX_ABS_WEIGHT}. Fail-fast
   * au premier index fautif.
   *
   * @apiNote Internal — exposed for testing.
   */
  static void validateValues(String tensorName, float[] data) {
    for (int i = 0; i < data.length; i++) {
      float v = data[i];
      if (Float.isNaN(v)) {
        throw new IllegalArgumentException("NaN value in tensor=" + tensorName + " at index=" + i);
      }
      if (Float.isInfinite(v)) {
        throw new IllegalArgumentException(
            "Infinite value in tensor=" + tensorName + " at index=" + i + ", value=" + v);
      }
      if (Math.abs(v) >= MAX_ABS_WEIGHT) {
        throw new IllegalArgumentException(
            "Magnitude exceeds "
                + MAX_ABS_WEIGHT
                + " in tensor="
                + tensorName
                + " at index="
                + i
                + ", value="
                + v);
      }
    }
  }

  /**
   * Étape 8 §6.6 : recalcul SHA-256 sur les 42 tenseurs de poids triés alphabétiquement et
   * comparaison avec le hash attendu (typiquement {@code _meta_model_hash}).
   *
   * @throws IllegalArgumentException si le hash recalculé diffère de {@code expectedHash}
   * @apiNote Internal — exposed for testing.
   */
  static void verifyHash(Map<String, float[]> floats, String expectedHash) {
    String actual = computeWeightsHash(floats);
    if (!actual.equalsIgnoreCase(expectedHash)) {
      throw new IllegalArgumentException(
          "Hash mismatch: expected " + expectedHash + ", computed " + actual);
    }
  }

  /**
   * Calcule le hash SHA-256 hex (lowercase) des 42 tenseurs de poids §6.3. Convention §6.5 :
   * concaténation des bytes float32 little-endian dans l'ordre alphabétique des noms, en
   * <strong>excluant</strong> tout nom commençant par {@code _meta_}. Reproduit bit-pour-bit {@code
   * numpy.tobytes()} sur descripteur {@code <f4} sur x86.
   *
   * @apiNote Internal — exposed for testing.
   */
  static String computeWeightsHash(Map<String, float[]> floats) {
    MessageDigest sha;
    try {
      sha = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 not available in JDK", e);
    }
    // Tri alphabétique stable des noms hors _meta_*. Les float tensors n'incluent pas les méta
    // (ce sont des string ou int64 scalars), mais on filtre explicitement par sécurité.
    var names = new TreeSet<String>();
    for (String key : floats.keySet()) {
      if (!key.startsWith("_meta_")) {
        names.add(key);
      }
    }
    for (String name : names) {
      float[] arr = floats.get(name);
      // ByteBuffer LE pour reproduire numpy.tobytes() avec descripteur '<f4'.
      ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (float v : arr) {
        buf.putFloat(v);
      }
      sha.update(buf.array());
    }
    byte[] digest = sha.digest();
    StringBuilder hex = new StringBuilder(digest.length * 2);
    for (byte b : digest) {
      hex.append(String.format("%02x", b & 0xFF));
    }
    return hex.toString();
  }
}
