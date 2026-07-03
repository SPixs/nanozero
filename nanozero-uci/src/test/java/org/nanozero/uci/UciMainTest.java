package org.nanozero.uci;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link UciMain#parseArgs}. La méthode {@code main} (lecture stdin / appel
 * {@code System.exit}) n'est pas directement testée — la boucle UCI elle-même est couverte par les
 * tests d'intégration {@link org.nanozero.uci.internal.UciScriptIT} et {@link
 * org.nanozero.uci.internal.UciIntegrationRaceTest}.
 */
class UciMainTest {

  @Test
  @DisplayName("parseArgs --network <path> : OK")
  void parseNetworkOnly() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/path/to/model.npz"});
    assertThat(args).isNotNull();
    assertThat(args.networkPath()).isEqualTo("/path/to/model.npz");
    assertThat(args.debugAtBoot()).isFalse();
    assertThat(args.useCuda()).as("v1.3.0 useCuda default false").isFalse();
  }

  @Test
  @DisplayName("parseArgs --network <path> --debug : OK avec debug=true")
  void parseNetworkAndDebug() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/p.npz", "--debug"});
    assertThat(args).isNotNull();
    assertThat(args.networkPath()).isEqualTo("/p.npz");
    assertThat(args.debugAtBoot()).isTrue();
  }

  @Test
  @DisplayName("parseArgs --debug avant --network : OK, ordre arbitraire")
  void parseDebugBeforeNetwork() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--debug", "--network", "/p.npz"});
    assertThat(args).isNotNull();
    assertThat(args.networkPath()).isEqualTo("/p.npz");
    assertThat(args.debugAtBoot()).isTrue();
  }

  @Test
  @DisplayName("parseArgs sans --network : retourne null")
  void parseMissingNetworkReturnsNull() {
    assertThat(UciMain.parseArgs(new String[0])).isNull();
    assertThat(UciMain.parseArgs(new String[] {"--debug"})).isNull();
  }

  @Test
  @DisplayName("parseArgs --network sans valeur : retourne null")
  void parseNetworkWithoutValueReturnsNull() {
    assertThat(UciMain.parseArgs(new String[] {"--network"})).isNull();
    assertThat(UciMain.parseArgs(new String[] {"--debug", "--network"})).isNull();
  }

  @Test
  @DisplayName("parseArgs ignore les flags inconnus (tolérance futurs ajouts)")
  void parseIgnoresUnknownFlags() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--foo", "--bar", "--network", "/p.npz"});
    assertThat(args).isNotNull();
    assertThat(args.networkPath()).isEqualTo("/p.npz");
  }

  // -------------------------------------------------------------------------------------------
  // (v1.3.0) Flag --cuda (cf. ADR-008-uci)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("v1.3.0 parseArgs --network <path> --cuda : useCuda=true")
  void parseCudaFlag() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/p.onnx", "--cuda"});
    assertThat(args).isNotNull();
    assertThat(args.useCuda()).isTrue();
  }

  @Test
  @DisplayName("v1.3.0 parseArgs --cuda + --debug + --network : tous les flags actifs")
  void parseCudaWithDebugAndNetwork() {
    UciMain.Args args =
        UciMain.parseArgs(new String[] {"--cuda", "--debug", "--network", "/p.onnx"});
    assertThat(args).isNotNull();
    assertThat(args.useCuda()).isTrue();
    assertThat(args.debugAtBoot()).isTrue();
    assertThat(args.networkPath()).isEqualTo("/p.onnx");
  }

  @Test
  @DisplayName("v1.3.0 parseArgs sans --cuda : useCuda=false (compat v1.2.0)")
  void parseWithoutCudaDefaultsFalse() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/p.npz"});
    assertThat(args).isNotNull();
    assertThat(args.useCuda()).isFalse();
  }

  // -------------------------------------------------------------------------------------------
  // (ADR-018) Flag --nncache
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("ADR-018 parseArgs sans --nncache : nnCacheSize=0 (cache désactivé)")
  void parseWithoutNnCacheDefaultsZero() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/p.npz"});
    assertThat(args).isNotNull();
    assertThat(args.nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 parseArgs --nncache 200000 : nnCacheSize=200000")
  void parseNnCacheFlag() {
    UciMain.Args args =
        UciMain.parseArgs(new String[] {"--network", "/p.onnx", "--nncache", "200000"});
    assertThat(args).isNotNull();
    assertThat(args.nnCacheSize()).isEqualTo(200000);
  }

  @Test
  @DisplayName("ADR-018 parseArgs --nncache négatif : clamp à 0")
  void parseNnCacheNegativeClampedToZero() {
    UciMain.Args args = UciMain.parseArgs(new String[] {"--network", "/p.onnx", "--nncache", "-7"});
    assertThat(args).isNotNull();
    assertThat(args.nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 parseArgs --nncache sans valeur : retourne null")
  void parseNnCacheWithoutValueReturnsNull() {
    assertThat(UciMain.parseArgs(new String[] {"--network", "/p.npz", "--nncache"})).isNull();
  }

  @Test
  @DisplayName("ADR-018 parseArgs --nncache non entier : retourne null")
  void parseNnCacheNonIntegerReturnsNull() {
    assertThat(UciMain.parseArgs(new String[] {"--network", "/p.npz", "--nncache", "abc"}))
        .isNull();
  }
}
