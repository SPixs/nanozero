package org.nanozero.nn.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests unitaires de {@link NpzReader} (cf. SPEC §13 phase 2). Couvre le critère de complétion :
 * chargement de {@code test-realistic.npz} retourne tous les tenseurs avec dimensions et valeurs
 * correctes ; rejet de {@code test-malformed.npz} avec exception claire.
 */
class NpzReaderTest {

  private static Path resourcePath(String name) {
    // Les fixtures sont commitées dans src/test/resources/npz/ et accessibles via le classpath
    // de test, mais la signature de NpzReader.read prend un Path filesystem. On résout via
    // ClassLoader pour rester portable (le cwd des tests Maven est par défaut le module root).
    var url = NpzReaderTest.class.getResource("/npz/" + name);
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/" + name);
    }
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/" + name, e);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // test-tiny.npz : 4 tenseurs avec valeurs prévisibles
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("test-tiny.npz : 4 tenseurs simples avec valeurs prévisibles")
  class TinyFixture {

    @Test
    @DisplayName("constants : float32 [4] = [1.0, 2.0, 3.0, 4.0]")
    void constantsFloat1d() throws IOException {
      var data = NpzReader.read(resourcePath("test-tiny.npz"));
      assertThat(data.floatTensors()).containsKey("constants");
      assertThat(data.floatTensors().get("constants")).containsExactly(1.0f, 2.0f, 3.0f, 4.0f);
      assertThat(data.tensorShapes().get("constants")).containsExactly(4);
    }

    @Test
    @DisplayName("sequence_2d : float32 [3,4] = arange(12).reshape(3,4)")
    void sequence2dFloat() throws IOException {
      var data = NpzReader.read(resourcePath("test-tiny.npz"));
      float[] arr = data.floatTensors().get("sequence_2d");
      assertThat(arr).hasSize(12);
      for (int i = 0; i < 12; i++) {
        assertThat(arr[i]).isEqualTo((float) i);
      }
      assertThat(data.tensorShapes().get("sequence_2d")).containsExactly(3, 4);
    }

    @Test
    @DisplayName("scalar_int : int64 scalaire = 42")
    void scalarInt64() throws IOException {
      var data = NpzReader.read(resourcePath("test-tiny.npz"));
      assertThat(data.int64Tensors()).containsKey("scalar_int");
      assertThat(data.int64Tensors().get("scalar_int")).containsExactly(42L);
      assertThat(data.tensorShapes().get("scalar_int")).isEmpty();
    }

    @Test
    @DisplayName("text : string scalaire = \"hello\" (UTF-32 LE)")
    void stringScalar() throws IOException {
      var data = NpzReader.read(resourcePath("test-tiny.npz"));
      assertThat(data.stringScalars()).containsEntry("text", "hello");
      assertThat(data.tensorShapes().get("text")).isEmpty();
    }

    @Test
    @DisplayName("Tous les tenseurs présents dans tensorShapes (4 entrées au total)")
    void allTensorsHaveShapes() throws IOException {
      var data = NpzReader.read(resourcePath("test-tiny.npz"));
      assertThat(data.tensorShapes())
          .containsOnlyKeys("constants", "sequence_2d", "scalar_int", "text");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // test-realistic.npz : structure §6.3 (42 weights + 5 _meta_*)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("test-realistic.npz : 42 tenseurs poids + 5 méta")
  class RealisticFixture {

    @Test
    @DisplayName("47 tenseurs au total dans tensorShapes")
    void totalTensorCount() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      assertThat(data.tensorShapes()).hasSize(47);
      assertThat(data.floatTensors()).hasSize(42);
      assertThat(data.int64Tensors()).hasSize(1); // _meta_training_step
      assertThat(data.stringScalars()).hasSize(4); // _meta_* strings
    }

    @Test
    @DisplayName("input_conv : weight (96, 119, 3, 3) + bias (96,)")
    void inputConvShapes() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      assertThat(data.tensorShapes().get("input_conv.weight")).containsExactly(96, 119, 3, 3);
      assertThat(data.tensorShapes().get("input_conv.bias")).containsExactly(96);
      assertThat(data.floatTensors().get("input_conv.weight")).hasSize(96 * 119 * 3 * 3);
      assertThat(data.floatTensors().get("input_conv.bias")).hasSize(96);
    }

    @Test
    @DisplayName("8 blocs résiduels conv1+conv2 (chacun weight (96,96,3,3) + bias (96,))")
    void residualBlocksShapes() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      for (int i = 0; i < 8; i++) {
        assertThat(data.tensorShapes().get("block_" + i + ".conv1.weight"))
            .as("block_%d.conv1.weight", i)
            .containsExactly(96, 96, 3, 3);
        assertThat(data.tensorShapes().get("block_" + i + ".conv1.bias")).containsExactly(96);
        assertThat(data.tensorShapes().get("block_" + i + ".conv2.weight"))
            .containsExactly(96, 96, 3, 3);
        assertThat(data.tensorShapes().get("block_" + i + ".conv2.bias")).containsExactly(96);
      }
    }

    @Test
    @DisplayName("Policy head : weight (73, 96, 1, 1) + bias (73,)")
    void policyHeadShapes() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      assertThat(data.tensorShapes().get("policy_head.conv.weight")).containsExactly(73, 96, 1, 1);
      assertThat(data.tensorShapes().get("policy_head.conv.bias")).containsExactly(73);
    }

    @Test
    @DisplayName("Value head WDL : conv (1,96,1,1)+bias, fc1 (64,64), fc2 (3,64)")
    void valueHeadShapes() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      assertThat(data.tensorShapes().get("value_head.conv.weight")).containsExactly(1, 96, 1, 1);
      assertThat(data.tensorShapes().get("value_head.conv.bias")).containsExactly(1);
      assertThat(data.tensorShapes().get("value_head.fc1.weight")).containsExactly(64, 64);
      assertThat(data.tensorShapes().get("value_head.fc1.bias")).containsExactly(64);
      assertThat(data.tensorShapes().get("value_head.fc2.weight"))
          .containsExactly(3, 64); // WDL v1.5.0
      assertThat(data.tensorShapes().get("value_head.fc2.bias")).containsExactly(3);
    }

    @Test
    @DisplayName(
        "Méta : 4 strings (architecture_version, input_plane_format, model_hash, export_date) + 1"
            + " int64 (training_step)")
    void metaTensors() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      // Le fixture a été régénéré en phase 8 (cf. scripts/python/generate_test_realistic_npz.py)
      // avec hash SHA-256 réel, training_step=1_000_000 et export_date figée 2026-05-08T12:00:00Z.
      assertThat(data.stringScalars())
          .containsEntry("_meta_architecture_version", "resnet8x96-v1")
          .containsEntry("_meta_input_plane_format", "alphazero-119")
          .containsEntry("_meta_export_date", "2026-05-08T12:00:00Z")
          .containsEntry(
              "_meta_model_hash",
              "1a5ee35245d96833f450ca486fe49cbd5f1f84936ddc2e69182c6ccf55d0d2bb");
      assertThat(data.int64Tensors().get("_meta_training_step")).containsExactly(1_000_000L);
    }

    @Test
    @DisplayName(
        "Valeurs reproductibles default_rng(42) × sigma=0.05 : input_conv.weight flat[0..3]")
    void reproducibleSeed42Values() throws IOException {
      // Référence calculée via Python (cf. scripts/python/generate_test_realistic_npz.py) :
      //   rng = np.random.default_rng(42)
      //   x = (rng.standard_normal((96,119,3,3)) * 0.05).astype(np.float32)
      // Valeurs flat[0..3] : 0.015235854, -0.051999204, 0.037522558, 0.047028236.
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      float[] inputW = data.floatTensors().get("input_conv.weight");
      assertThat(inputW[0]).isCloseTo(0.015235854f, within(1e-7f));
      assertThat(inputW[1]).isCloseTo(-0.051999204f, within(1e-7f));
      assertThat(inputW[2]).isCloseTo(0.037522558f, within(1e-7f));
      assertThat(inputW[3]).isCloseTo(0.047028236f, within(1e-7f));
    }

    @Test
    @DisplayName("Biais à 0 par convention fixture phase 8 (sanity-compliant)")
    void biasesAreZero() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      // Convention fixture phase 8 : biais à 0 pour conformité sanity §5.2.2 simple. Les vrais
      // biais BN-folded d'un export PyTorch sont non triviaux ; testés en parité phase 9.
      assertThat(data.floatTensors().get("input_conv.bias")).containsOnly(0.0f);
      assertThat(data.floatTensors().get("policy_head.conv.bias")).containsOnly(0.0f);
      assertThat(data.floatTensors().get("value_head.fc2.bias")).containsOnly(0.0f);
    }

    @Test
    @DisplayName("Total paramètres = 1 443 085 (102 912 + 1 328 640 + 7 081 + 4 452) — WDL v1.5.0")
    void totalParameterCount() throws IOException {
      var data = NpzReader.read(resourcePath("test-realistic.npz"));
      long total = 0;
      for (var entry : data.floatTensors().entrySet()) {
        total += entry.getValue().length;
      }
      // Décomposition canonique (value head WDL v1.5.0 : fc2 64->3 au lieu de 64->1) :
      //   input_conv : 96*119*9 + 96 = 102 912
      //   tour résiduelle : 8 * 2 * (96*96*9 + 96) = 8 * 2 * 83 040 = 1 328 640
      //   policy_head : 73*96 + 73 = 7 081
      //   value_head : (96+1) + (64*64+64) + (3*64+3) = 97 + 4 160 + 195 = 4 452
      //   total = 102 912 + 1 328 640 + 7 081 + 4 452 = 1 443 085
      assertThat(total).isEqualTo(102_912L + 1_328_640L + 7_081L + 4_452L);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // test-malformed.npz : ZIP tronqué, doit être rejeté
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("test-malformed.npz : rejeté avec IOException ou IllegalArgumentException")
  void malformedZipRejected() {
    Path malformed = resourcePath("test-malformed.npz");
    assertThatThrownBy(() -> NpzReader.read(malformed))
        .isInstanceOfAny(IOException.class, IllegalArgumentException.class);
  }

  // ---------------------------------------------------------------------------------------------
  // Cas dégénérés
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Cas dégénérés : fichier inexistant, vide, non-zip")
  class DegenerateInputs {

    @Test
    @DisplayName("Fichier inexistant : NoSuchFileException (extends IOException)")
    void missingFile(@TempDir Path tmp) {
      Path absent = tmp.resolve("does-not-exist.npz");
      assertThatThrownBy(() -> NpzReader.read(absent)).isInstanceOf(IOException.class);
    }

    @Test
    @DisplayName("Fichier vide (0 octet) : IOException (ZipException 'zip file is empty')")
    void emptyFile(@TempDir Path tmp) throws IOException {
      Path empty = tmp.resolve("empty.npz");
      Files.write(empty, new byte[0]);
      assertThatThrownBy(() -> NpzReader.read(empty)).isInstanceOf(IOException.class);
    }

    @Test
    @DisplayName("Fichier non-zip (texte arbitraire) : IOException")
    void nonZipFile(@TempDir Path tmp) throws IOException {
      Path txt = tmp.resolve("not-a-zip.npz");
      Files.writeString(txt, "this is not a zip file at all, just plain text\n");
      assertThatThrownBy(() -> NpzReader.read(txt)).isInstanceOf(IOException.class);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Cas synthétiques : .npz construit en mémoire avec .npy malformés (error paths du parser .npy)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Cas synthétiques : .npy malformés (error paths du parser)")
  class MalformedNpyContent {

    /** Crée un .npz contenant un seul .npy avec le payload donné. */
    private Path makeNpz(Path dir, String tensorName, byte[] payload) throws IOException {
      Path npz = dir.resolve("synth.npz");
      try (OutputStream fos = Files.newOutputStream(npz);
          ZipOutputStream zos = new ZipOutputStream(fos)) {
        zos.putNextEntry(new ZipEntry(tensorName + ".npy"));
        zos.write(payload);
        zos.closeEntry();
      }
      return npz;
    }

    /** Construit un payload .npy v1 valide minimal (magic + version + header + données). */
    private byte[] buildNpyV1(String header, byte[] data) {
      // Header doit être padded à un multiple de 64 octets après magic+version+headerLen,
      // mais ZipFile ne s'en plaint pas et le parser non plus tant que la longueur déclarée
      // est cohérente.
      byte[] hdrBytes = header.getBytes(StandardCharsets.US_ASCII);
      int hdrLen = hdrBytes.length;
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      out.write(0x93);
      out.writeBytes(new byte[] {'N', 'U', 'M', 'P', 'Y'});
      out.write(1);
      out.write(0); // version 1.0
      out.write(hdrLen & 0xFF);
      out.write((hdrLen >>> 8) & 0xFF);
      out.writeBytes(hdrBytes);
      out.writeBytes(data);
      return out.toByteArray();
    }

    @Test
    @DisplayName("Magic .npy invalide → IllegalArgumentException")
    void wrongMagic(@TempDir Path tmp) throws IOException {
      // Payload commence par "ABCDEF..." au lieu de "\x93NUMPY"
      byte[] payload = new byte[20];
      for (int i = 0; i < payload.length; i++) {
        payload[i] = (byte) ('A' + i);
      }
      Path npz = makeNpz(tmp, "broken", payload);
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Magic .npy invalide");
    }

    @Test
    @DisplayName("Payload trop court (< 10 octets) → IllegalArgumentException")
    void payloadTooShort(@TempDir Path tmp) throws IOException {
      byte[] payload = {(byte) 0x93, 'N', 'U', 'M', 'P'};
      Path npz = makeNpz(tmp, "short", payload);
      assertThatThrownBy(() -> NpzReader.read(npz)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Version .npy 3.0 non supportée → IllegalArgumentException")
    void unsupportedVersion(@TempDir Path tmp) throws IOException {
      byte[] payload = new byte[12];
      payload[0] = (byte) 0x93;
      payload[1] = 'N';
      payload[2] = 'U';
      payload[3] = 'M';
      payload[4] = 'P';
      payload[5] = 'Y';
      payload[6] = 3;
      payload[7] = 0;
      Path npz = makeNpz(tmp, "v3", payload);
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Version");
    }

    @Test
    @DisplayName(
        "Header .npy tronqué (longueur déclarée > taille du fichier) → IllegalArgumentException")
    void headerTooLong(@TempDir Path tmp) throws IOException {
      byte[] payload = new byte[10];
      payload[0] = (byte) 0x93;
      payload[1] = 'N';
      payload[2] = 'U';
      payload[3] = 'M';
      payload[4] = 'P';
      payload[5] = 'Y';
      payload[6] = 1;
      payload[7] = 0;
      // Déclare un header de 10 000 octets alors qu'il n'y a rien
      payload[8] = 0x10;
      payload[9] = 0x27; // 0x2710 = 10 000 LE
      Path npz = makeNpz(tmp, "trunc", payload);
      assertThatThrownBy(() -> NpzReader.read(npz)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("fortran_order = True → IllegalArgumentException")
    void fortranOrderTrue(@TempDir Path tmp) throws IOException {
      String header = "{'descr': '<f4', 'fortran_order': True, 'shape': (4,), }   \n";
      byte[] data = new byte[16]; // 4 floats
      Path npz = makeNpz(tmp, "fortran", buildNpyV1(header, data));
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("fortran_order");
    }

    @Test
    @DisplayName("Dtype non supporté ('<f8') → IllegalArgumentException")
    void unsupportedDtype(@TempDir Path tmp) throws IOException {
      String header = "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }   \n";
      byte[] data = new byte[32]; // 4 doubles
      Path npz = makeNpz(tmp, "f8", buildNpyV1(header, data));
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Dtype non supporté");
    }

    @Test
    @DisplayName("Header sans 'descr' → IllegalArgumentException")
    void missingDescrField(@TempDir Path tmp) throws IOException {
      String header = "{'fortran_order': False, 'shape': (4,), }                  \n";
      Path npz = makeNpz(tmp, "no-descr", buildNpyV1(header, new byte[16]));
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("descr");
    }

    @Test
    @DisplayName("Strings non scalaires non supportées (shape=(2,)) → IllegalArgumentException")
    void nonScalarStringRejected(@TempDir Path tmp) throws IOException {
      String header = "{'descr': '<U5', 'fortran_order': False, 'shape': (2,), }  \n";
      byte[] data = new byte[2 * 5 * 4]; // 2 strings de 5 chars × 4 bytes
      Path npz = makeNpz(tmp, "u5_2", buildNpyV1(header, data));
      assertThatThrownBy(() -> NpzReader.read(npz))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("non scalaire");
    }

    @Test
    @DisplayName("Header v2 (longueur 4 octets) lu correctement")
    void headerV2() throws IOException, java.net.URISyntaxException {
      // Construit un .npy v2 valide en mémoire et vérifie que le parser le lit.
      String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }  \n";
      byte[] hdrBytes = header.getBytes(StandardCharsets.US_ASCII);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      out.write(0x93);
      out.writeBytes(new byte[] {'N', 'U', 'M', 'P', 'Y'});
      out.write(2);
      out.write(0);
      // Longueur sur 4 octets LE
      int n = hdrBytes.length;
      out.write(n & 0xFF);
      out.write((n >>> 8) & 0xFF);
      out.write((n >>> 16) & 0xFF);
      out.write((n >>> 24) & 0xFF);
      out.writeBytes(hdrBytes);
      // 3 floats LE : 1.0f, 2.0f, 3.0f
      java.nio.ByteBuffer buf =
          java.nio.ByteBuffer.allocate(12).order(java.nio.ByteOrder.LITTLE_ENDIAN);
      buf.putFloat(1.0f).putFloat(2.0f).putFloat(3.0f);
      out.writeBytes(buf.array());

      Path tmp = Files.createTempDirectory("npz-v2-");
      try {
        Path npz = tmp.resolve("v2.npz");
        try (OutputStream fos = Files.newOutputStream(npz);
            ZipOutputStream zos = new ZipOutputStream(fos)) {
          zos.putNextEntry(new ZipEntry("v2_tensor.npy"));
          zos.write(out.toByteArray());
          zos.closeEntry();
        }
        var data = NpzReader.read(npz);
        assertThat(data.floatTensors().get("v2_tensor")).containsExactly(1.0f, 2.0f, 3.0f);
        assertThat(data.tensorShapes().get("v2_tensor")).containsExactly(3);
      } finally {
        Files.walk(tmp)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(
                p -> {
                  try {
                    Files.deleteIfExists(p);
                  } catch (IOException ignored) {
                    // Cleanup best-effort.
                  }
                });
      }
    }

    @Test
    @DisplayName("Lecture |S* (ASCII bytes scalar) avec null padding correctement détaché")
    void asciiByteString(@TempDir Path tmp) throws IOException {
      // Header pour |S10 = 10 bytes ASCII fixed width
      String header = "{'descr': '|S10', 'fortran_order': False, 'shape': (), }   \n";
      byte[] data = new byte[10];
      byte[] str = "hi".getBytes(StandardCharsets.US_ASCII);
      System.arraycopy(str, 0, data, 0, str.length);
      // bytes 2..9 = null padding
      Path npz = makeNpz(tmp, "ascii", buildNpyV1(header, data));
      var npzData = NpzReader.read(npz);
      assertThat(npzData.stringScalars().get("ascii")).isEqualTo("hi");
    }

    @Test
    @DisplayName("Entrée non-.npy dans le ZIP : ignorée silencieusement")
    void nonNpyEntryIgnored(@TempDir Path tmp) throws IOException {
      Path npz = tmp.resolve("mixed.npz");
      try (OutputStream fos = Files.newOutputStream(npz);
          ZipOutputStream zos = new ZipOutputStream(fos)) {
        zos.putNextEntry(new ZipEntry("readme.txt"));
        zos.write("ignore me\n".getBytes(StandardCharsets.US_ASCII));
        zos.closeEntry();

        // Plus une entrée .npy valide
        String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }  \n";
        zos.putNextEntry(new ZipEntry("real.npy"));
        zos.write(buildNpyV1(header, new byte[] {0x00, 0x00, (byte) 0x80, 0x3F})); // 1.0f LE
        zos.closeEntry();
      }
      var data = NpzReader.read(npz);
      assertThat(data.tensorShapes()).containsOnlyKeys("real");
      assertThat(data.floatTensors().get("real")).containsExactly(1.0f);
    }
  }
}
