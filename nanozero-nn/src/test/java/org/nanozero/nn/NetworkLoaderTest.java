package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.nn.internal.NpzReader;
import org.nanozero.nn.internal.NpzReader.NpzData;

/**
 * Tests {@link NetworkLoader} (cf. SPEC §13 phase 8).
 *
 * <p>Trois axes :
 *
 * <ul>
 *   <li>Happy path sur le fixture {@code test-realistic.npz} (chargement, hash check, forward sans
 *       crash, intégration {@code forwardSingle}).
 *   <li>Rejets §6.6 sur méthodes package-private alimentées par des Maps synthétiques (zéro fixture
 *       binaire ; un fichier .npz corrompu serait équivalent mais coûteux à maintenir).
 *   <li>Performance indicative {@code @Tag("perf")} : boot &lt; 500 ms hors hash check, &lt; 1 s
 *       avec.
 * </ul>
 */
class NetworkLoaderTest {

  /** Hash attendu pour {@code test-realistic.npz} (régénéré pour le value head WDL v1.5.0). */
  private static final String FIXTURE_HASH =
      "1a5ee35245d96833f450ca486fe49cbd5f1f84936ddc2e69182c6ccf55d0d2bb";

  // ---------------------------------------------------------------------------------------------
  // Helpers : path resolution + Maps synthétiques pour tests de rejet
  // ---------------------------------------------------------------------------------------------

  private static Path fixturePath() {
    var url = NetworkLoaderTest.class.getResource("/npz/test-realistic.npz");
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/test-realistic.npz");
    }
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/test-realistic.npz", e);
    }
  }

  /** NpzData synthétique avec les 5 méta valides + 42 tenseurs zero-init (sanity-compliant). */
  private static NpzData syntheticValidData() {
    Map<String, float[]> floats = new HashMap<>();
    Map<String, long[]> int64s = new HashMap<>();
    Map<String, String> strings = new HashMap<>();
    Map<String, int[]> shapes = new HashMap<>();

    for (var entry : NetworkLoader.EXPECTED_TENSORS.entrySet()) {
      int total = 1;
      for (int d : entry.getValue()) {
        total *= d;
      }
      floats.put(entry.getKey(), new float[total]); // zero-filled
      shapes.put(entry.getKey(), entry.getValue());
    }

    strings.put(NetworkLoader.META_ARCH_VERSION, "resnet8x96-v1");
    strings.put(NetworkLoader.META_INPUT_PLANE_FORMAT, "alphazero-119");
    strings.put(NetworkLoader.META_MODEL_HASH, "deadbeef");
    strings.put(NetworkLoader.META_EXPORT_DATE, "2026-05-08T12:00:00Z");
    int64s.put(NetworkLoader.META_TRAINING_STEP, new long[] {42L});

    shapes.put(NetworkLoader.META_ARCH_VERSION, new int[0]);
    shapes.put(NetworkLoader.META_INPUT_PLANE_FORMAT, new int[0]);
    shapes.put(NetworkLoader.META_MODEL_HASH, new int[0]);
    shapes.put(NetworkLoader.META_EXPORT_DATE, new int[0]);
    shapes.put(NetworkLoader.META_TRAINING_STEP, new int[0]);

    return new NpzData(floats, int64s, strings, shapes);
  }

  // ---------------------------------------------------------------------------------------------
  // Happy path : chargement de test-realistic.npz
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Happy path : test-realistic.npz")
  class HappyPath {

    @Test
    @DisplayName("load(path) : Network non-null, metadata cohérente avec le fixture")
    void testLoadValidNpz() throws IOException {
      Network net = NetworkLoader.load(fixturePath());
      assertThat(net).isNotNull();
      NetworkMetadata md = net.metadata();
      assertThat(md.architectureVersion()).isEqualTo("resnet8x96-v1");
      assertThat(md.inputPlaneFormat()).isEqualTo("alphazero-119");
      assertThat(md.modelHash()).isEqualTo(FIXTURE_HASH);
      assertThat(md.trainingStep()).isEqualTo(1_000_000L);
      assertThat(md.exportDate()).isEqualTo("2026-05-08T12:00:00Z");
    }

    @Test
    @DisplayName("forward(planes, batch=8, output) : pas de crash, pas de NaN dans logits/values")
    void testLoadedNetworkForwardsWithoutCrash() throws IOException {
      Network net = NetworkLoader.load(fixturePath());
      float[] planes = new float[Network.MAX_BATCH * 119 * 64];
      java.util.Random rng = new java.util.Random(0xCAFEL);
      for (int i = 0; i < 8 * 119 * 64; i++) {
        planes[i] = rng.nextFloat() < 0.3f ? 1f : 0f;
      }
      NNOutput out = new NNOutput();
      net.forward(planes, 8, out);
      for (int n = 0; n < 8; n++) {
        for (int i = 0; i < MoveEncoding.POLICY_INDICES; i++) {
          float v = out.logits[n * MoveEncoding.POLICY_INDICES + i];
          assertThat(Float.isNaN(v)).as("logit NaN at n=%d i=%d", n, i).isFalse();
        }
        assertThat(Float.isNaN(out.values[n])).as("value NaN at n=%d", n).isFalse();
        assertThat(out.values[n]).isBetween(-1.0f, 1.0f);
      }
    }

    @Test
    @DisplayName("load(path, skipHashCheck()) : OK même sans validation hash")
    void testSkipHashCheck() throws IOException {
      Network net = NetworkLoader.load(fixturePath(), LoadOptions.skipHashCheck());
      assertThat(net).isNotNull();
      // metadata.modelHash() reste celui du fichier (non recalculé), valeur informationnelle.
      assertThat(net.metadata().modelHash()).isEqualTo(FIXTURE_HASH);
    }

    @Test
    @DisplayName("forwardSingle(starting position) : value ∈ [-1, 1], pas de crash")
    void testLoadAndForwardSingle() throws IOException {
      Network net = NetworkLoader.load(fixturePath());
      NNSingleResult result = net.forwardSingle(new GameState());
      assertThat(result.value()).isBetween(-1.0f, 1.0f);
      assertThat(result.logits()).hasSize(MoveEncoding.POLICY_INDICES);
      // Tous les logits finis.
      for (int i = 0; i < result.logits().length; i++) {
        assertThat(Float.isFinite(result.logits()[i])).as("logit[%d] finite", i).isTrue();
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Rejets §6.6 sur Maps synthétiques (validateMetadata)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Rejets §6.6 : validateMetadata")
  class MetadataRejection {

    @Test
    @DisplayName("Manque _meta_architecture_version → IAE explicite")
    void testValidateMetadata_missingArchVersion() {
      NpzData data = syntheticValidData();
      data.stringScalars().remove(NetworkLoader.META_ARCH_VERSION);
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("Missing _meta_architecture_version");
    }

    @Test
    @DisplayName("Mauvais _meta_architecture_version → IAE 'Incompatible architecture'")
    void testValidateMetadata_wrongArchVersion() {
      NpzData data = syntheticValidData();
      data.stringScalars().put(NetworkLoader.META_ARCH_VERSION, "resnet12x128");
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Incompatible architecture")
          .hasMessageContaining("resnet12x128");
    }

    @Test
    @DisplayName("Manque _meta_input_plane_format → IAE")
    void testValidateMetadata_missingInputFormat() {
      NpzData data = syntheticValidData();
      data.stringScalars().remove(NetworkLoader.META_INPUT_PLANE_FORMAT);
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("Missing _meta_input_plane_format");
    }

    @Test
    @DisplayName("Mauvais _meta_input_plane_format → IAE 'Incompatible input plane format'")
    void testValidateMetadata_wrongInputFormat() {
      NpzData data = syntheticValidData();
      data.stringScalars().put(NetworkLoader.META_INPUT_PLANE_FORMAT, "stockfish-768");
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Incompatible input plane format")
          .hasMessageContaining("stockfish-768");
    }

    @Test
    @DisplayName("Manque _meta_training_step → IAE")
    void testValidateMetadata_missingTrainingStep() {
      NpzData data = syntheticValidData();
      data.int64Tensors().remove(NetworkLoader.META_TRAINING_STEP);
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("_meta_training_step");
    }

    @Test
    @DisplayName("Manque _meta_model_hash → IAE")
    void testValidateMetadata_missingModelHash() {
      NpzData data = syntheticValidData();
      data.stringScalars().remove(NetworkLoader.META_MODEL_HASH);
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("Missing _meta_model_hash");
    }

    @Test
    @DisplayName("Manque _meta_export_date → IAE")
    void testValidateMetadata_missingExportDate() {
      NpzData data = syntheticValidData();
      data.stringScalars().remove(NetworkLoader.META_EXPORT_DATE);
      assertThatThrownBy(() -> NetworkLoader.validateMetadata(data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("Missing _meta_export_date");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Rejets §6.6 : validateTensorPresenceAndShape
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Rejets §6.6 : validateTensorPresenceAndShape")
  class TensorRejection {

    @Test
    @DisplayName("Tenseur manquant : IAE 'Missing tensor'")
    void testValidateTensor_missing() {
      NpzData data = syntheticValidData();
      data.floatTensors().remove("block_3.conv1.weight");
      data.tensorShapes().remove("block_3.conv1.weight");
      assertThatThrownBy(
              () ->
                  NetworkLoader.validateTensorPresenceAndShape(
                      data.floatTensors(),
                      data.tensorShapes(),
                      "block_3.conv1.weight",
                      new int[] {96, 96, 3, 3}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Missing tensor")
          .hasMessageContaining("block_3.conv1.weight");
    }

    @Test
    @DisplayName("Shape différent : IAE 'Wrong shape' avec attendu vs obtenu")
    void testValidateTensor_wrongShape() {
      NpzData data = syntheticValidData();
      data.tensorShapes().put("block_3.conv1.weight", new int[] {96, 96, 5, 5});
      assertThatThrownBy(
              () ->
                  NetworkLoader.validateTensorPresenceAndShape(
                      data.floatTensors(),
                      data.tensorShapes(),
                      "block_3.conv1.weight",
                      new int[] {96, 96, 3, 3}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Wrong shape")
          .hasMessageContaining("block_3.conv1.weight")
          .hasMessageContaining("[96, 96, 3, 3]")
          .hasMessageContaining("[96, 96, 5, 5]");
    }

    @Test
    @DisplayName("Mauvais dtype (présent dans shapes mais pas dans floats) : IAE 'Wrong dtype'")
    void testValidateTensor_wrongDtype() {
      NpzData data = syntheticValidData();
      // Simule un tenseur stocké comme int64 : présent dans shapes mais absent de floats.
      data.floatTensors().remove("input_conv.bias");
      // shape conservé, dtype changé
      assertThatThrownBy(
              () ->
                  NetworkLoader.validateTensorPresenceAndShape(
                      data.floatTensors(), data.tensorShapes(), "input_conv.bias", new int[] {96}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Wrong dtype")
          .hasMessageContaining("input_conv.bias");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Rejets §6.6 : validateValues
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Rejets §6.6 : sanity values §5.2.2")
  class ValueRejection {

    @Test
    @DisplayName("NaN détecté avec localisation tensor + index")
    void testValidateValues_nan() {
      float[] data = new float[100];
      data[42] = Float.NaN;
      assertThatThrownBy(() -> NetworkLoader.validateValues("test_tensor", data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("NaN")
          .hasMessageContaining("test_tensor")
          .hasMessageContaining("index=42");
    }

    @Test
    @DisplayName("+Infinity détecté avec localisation")
    void testValidateValues_inf() {
      float[] data = new float[100];
      data[7] = Float.POSITIVE_INFINITY;
      assertThatThrownBy(() -> NetworkLoader.validateValues("test_tensor", data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Infinite")
          .hasMessageContaining("test_tensor")
          .hasMessageContaining("index=7");
    }

    @Test
    @DisplayName("-Infinity détecté avec localisation")
    void testValidateValues_negInf() {
      float[] data = new float[100];
      data[3] = Float.NEGATIVE_INFINITY;
      assertThatThrownBy(() -> NetworkLoader.validateValues("test_tensor", data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Infinite")
          .hasMessageContaining("index=3");
    }

    @Test
    @DisplayName("Magnitude > 100.0 détectée avec localisation et valeur")
    void testValidateValues_largeMagnitude() {
      float[] data = new float[100];
      data[55] = 150.0f;
      assertThatThrownBy(() -> NetworkLoader.validateValues("test_tensor", data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Magnitude exceeds")
          .hasMessageContaining("test_tensor")
          .hasMessageContaining("index=55")
          .hasMessageContaining("150");
    }

    @Test
    @DisplayName("Magnitude négative -150.0 détectée également")
    void testValidateValues_largeNegativeMagnitude() {
      float[] data = new float[100];
      data[1] = -150.0f;
      assertThatThrownBy(() -> NetworkLoader.validateValues("test_tensor", data))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Magnitude exceeds");
    }

    @Test
    @DisplayName("Tenseur tout à 99.9f passe (juste sous le seuil 100.0)")
    void testValidateValues_belowThreshold() {
      float[] data = new float[100];
      java.util.Arrays.fill(data, 99.9f);
      // Doit passer sans exception.
      NetworkLoader.validateValues("test_tensor", data);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Rejets §6.6 : verifyHash
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Rejets §6.6 : verifyHash + computeWeightsHash")
  class HashRejection {

    @Test
    @DisplayName("Hash mismatch sur Map synthétique : IAE 'Hash mismatch'")
    void testHashCheckRejectsCorruptHash() {
      Map<String, float[]> floats = new HashMap<>();
      for (var entry : NetworkLoader.EXPECTED_TENSORS.entrySet()) {
        int total = 1;
        for (int d : entry.getValue()) {
          total *= d;
        }
        floats.put(entry.getKey(), new float[total]);
      }
      String wrongHash = "0".repeat(64);
      assertThatThrownBy(() -> NetworkLoader.verifyHash(floats, wrongHash))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Hash mismatch")
          .hasMessageContaining("expected " + wrongHash);
    }

    @Test
    @DisplayName("computeWeightsHash : exclut les noms _meta_* du hash")
    void testComputeWeightsHash_excludesMetaPrefix() {
      Map<String, float[]> withMeta = new HashMap<>();
      withMeta.put("a.weight", new float[] {1.0f, 2.0f});
      withMeta.put("_meta_should_be_excluded", new float[] {99.0f, 99.0f});
      Map<String, float[]> noMeta = new HashMap<>();
      noMeta.put("a.weight", new float[] {1.0f, 2.0f});
      assertThat(NetworkLoader.computeWeightsHash(withMeta))
          .isEqualTo(NetworkLoader.computeWeightsHash(noMeta));
    }

    @Test
    @DisplayName("computeWeightsHash : tri alphabétique (z_first.bytes APRÈS a_second.bytes)")
    void testComputeWeightsHash_alphabeticalOrder() {
      Map<String, float[]> a = new HashMap<>();
      a.put("a_second", new float[] {1.0f});
      a.put("z_first", new float[] {2.0f});
      Map<String, float[]> b = new HashMap<>();
      b.put("z_first", new float[] {2.0f});
      b.put("a_second", new float[] {1.0f});
      // Insertion order différent, hash identique (tri stable par TreeSet).
      assertThat(NetworkLoader.computeWeightsHash(a))
          .isEqualTo(NetworkLoader.computeWeightsHash(b));
    }

    @Test
    @DisplayName(
        "computeWeightsHash sur fixture test-realistic.npz : reproduit exactement le hash Python")
    void testComputeWeightsHash_matchesPythonReference() throws IOException {
      // Référence : le hash a été calculé côté Python par
      // scripts/python/generate_test_realistic_npz.py et stocké dans _meta_model_hash.
      // Si le calcul Java diverge (endianness, ordre, inclusion _meta_), ce test casse.
      var data = NpzReader.read(fixturePath());
      String javaHash = NetworkLoader.computeWeightsHash(data.floatTensors());
      assertThat(javaHash).isEqualTo(FIXTURE_HASH);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Performance indicative @Tag("perf")
  // ---------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Bench load(skipHashCheck) : < 500 ms (cible §13 phase 8)")
  void testLoadPerformance() throws IOException {
    Path p = fixturePath();
    // Warmup 5 loads.
    for (int i = 0; i < 5; i++) {
      NetworkLoader.load(p, LoadOptions.skipHashCheck());
    }
    long t0 = System.nanoTime();
    Network net = NetworkLoader.load(p, LoadOptions.skipHashCheck());
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
    System.out.printf("load(skipHashCheck) : %d ms%n", elapsedMs);
    assertThat(net).isNotNull();
    assertThat(elapsedMs).as("boot < 500 ms hors hash check").isLessThan(500L);
  }

  @Test
  @Tag("perf")
  @DisplayName("Bench load(defaults) : < 1000 ms (cible §13 phase 8)")
  void testLoadPerformanceWithHash() throws IOException {
    Path p = fixturePath();
    for (int i = 0; i < 5; i++) {
      NetworkLoader.load(p, LoadOptions.defaults());
    }
    long t0 = System.nanoTime();
    Network net = NetworkLoader.load(p, LoadOptions.defaults());
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
    System.out.printf("load(defaults) : %d ms%n", elapsedMs);
    assertThat(net).isNotNull();
    assertThat(elapsedMs).as("boot < 1000 ms avec hash check").isLessThan(1000L);
  }
}
