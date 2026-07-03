package org.nanozero.uci;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;
import org.nanozero.nn.NetworkOnnx;
import org.nanozero.uci.internal.UciAdapterState;
import org.nanozero.uci.internal.UciCommand;
import org.nanozero.uci.internal.UciCommandHandler;
import org.nanozero.uci.internal.UciCommandParser;
import org.nanozero.uci.internal.UciOptionsState;
import org.nanozero.uci.internal.UciResponseWriter;

/**
 * Point d'entrée exécutable du module {@code nanozero-uci} (cf. SPEC §4.1, §5.1, §12 phase 7).
 *
 * <p>Lance la boucle UCI principale sur stdin/stdout :
 *
 * <pre>{@code
 * java --add-modules jdk.incubator.vector \
 *      -jar nanozero-uci-1.0.0.jar \
 *      --network /path/to/model.npz [--debug]
 * }</pre>
 *
 * <p>Arguments CLI :
 *
 * <ul>
 *   <li>{@code --network <path>} (obligatoire) : chemin vers le fichier {@code .npz} du réseau
 *       pré-entraîné.
 *   <li>{@code --debug} (optionnel) : active les logs debug sur stderr dès le boot. Peut être
 *       activé/désactivé runtime via la commande UCI {@code debug on|off}.
 *   <li>{@code --nncache <entries>} (optionnel, ADR-018) : taille du cache d'évaluation NN
 *       optionnel (nombre d'entrées). {@code 0} (défaut) = désactivé (comportement bit-pour-bit).
 *       Pré-amorce l'option UCI {@code NNCacheSize} (surchargeable runtime via {@code setoption}).
 * </ul>
 *
 * <p><strong>Boucle UCI</strong> :
 *
 * <ol>
 *   <li>Lit ligne par ligne sur stdin (UTF-8).
 *   <li>Parse via {@link UciCommandParser#parse}.
 *   <li>Dispatch via {@link UciCommandHandler#handle} → {@link UciCommandHandler.HandleAction}.
 *   <li>Sort sur {@code QUIT} ou EOF (stdin fermé).
 * </ol>
 *
 * <p>Le cleanup ({@code engine.close} + session active stoppée) est garanti par {@code
 * try-with-resources} sur {@link UciAdapterState}, même en cas d'exception ou d'EOF abrupt.
 *
 * <p><strong>Convention I/O</strong> (cf. SPEC §7.2, ADR-007) :
 *
 * <ul>
 *   <li>Stdin / stdout en UTF-8 strict.
 *   <li>Stdout configuré en auto-flush (le {@code "\n"} déclenche le flush).
 *   <li>Stdout strictement réservé aux lignes UCI conformes. Les logs debug vont sur
 *       <strong>stderr</strong>.
 * </ul>
 */
public final class UciMain {

  private UciMain() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Point d'entrée. Parse les arguments CLI, charge le réseau, instancie l'engine, lance la boucle
   * UCI sur stdin/stdout.
   *
   * @param args arguments CLI (cf. Javadoc classe)
   * @throws IOException si le chargement du réseau ou la lecture stdin échoue
   */
  public static void main(String[] args) throws IOException {
    Args cliArgs = parseArgs(args);
    if (cliArgs == null) {
      printUsage();
      System.exit(1);
      return;
    }

    // v1.3.0 : helper avec fallback CPU + warning si --cuda demandé mais CUDA EP indisponible.
    Network network = loadNetworkWithCudaFallback(Path.of(cliArgs.networkPath), cliArgs.useCuda);

    PrintStream out = new PrintStream(System.out, /* autoFlush */ true, StandardCharsets.UTF_8);
    UciResponseWriter writer = new UciResponseWriter(out);
    UciOptionsState options = new UciOptionsState();

    // (ADR-018) Pré-amorce l'option UCI NNCacheSize depuis le flag CLI --nncache. La valeur est
    // lue par buildEngineConfigFromOptions à la création lazy de l'Engine ; un setoption
    // NNCacheSize ultérieur peut la surcharger. La valeur 0 (défaut) est un no-op (= défaut de
    // l'option) : pré-amorçage inconditionnel pour éviter une branche non testable dans main
    // (les tests unitaires ciblent parseArgs, pas main qui lit stdin / appelle System.exit).
    options.set("NNCacheSize", Integer.toString(cliArgs.nnCacheSize));

    // v1.1.0 : constructor lazy de UciAdapterState. L'Engine est créé au premier
    // accès (typiquement la première commande `go`), lisant à ce moment les
    // hidden options Dirichlet courantes (cf. SPEC §6.4). Permet aux clients
    // training de set les options Dirichlet via setoption avant le premier go.
    // v1.3.0 : useCuda propagé pour l'auto-config batchSize dans buildEngineConfigFromOptions.
    try (UciAdapterState state = new UciAdapterState(network, writer, options, cliArgs.useCuda)) {
      state.setDebugMode(cliArgs.debugAtBoot);
      runLoop(System.in, state);
    }
  }

  /**
   * (v1.3.0) Charge le {@link Network} avec gestion CUDA + fallback CPU explicite (cf.
   * ADR-008-uci).
   *
   * <ul>
   *   <li>Si {@code useCuda=true} : tente {@link NetworkOnnx#loadCuda} ; en cas d'échec, log {@code
   *       [warn] CUDA EP unavailable...} sur stderr et fallback vers {@link NetworkLoader#loadAuto}
   *       (dispatch sur extension : {@code .onnx} → ORT CPU EP, {@code .npz} → Vector API SIMD).
   *   <li>Si {@code useCuda=false} (défaut) : direct {@link NetworkLoader#loadAuto}, comportement
   *       v1.2.0 strict.
   * </ul>
   *
   * <p>Le warning sur stderr préserve l'ADR-006 (stdio exclusive : stdout = UCI protocol only). Le
   * binaire ne crash JAMAIS sur échec CUDA — fallback CPU garanti.
   *
   * @param networkPath chemin vers le fichier modèle, non null
   * @param useCuda flag CLI {@code --cuda} (intention utilisateur ; pas garantie de succès)
   * @return Network chargé (CUDA ou CPU EP selon outcome)
   * @throws IOException si même le fallback CPU échoue (fichier corrompu / inexistant)
   */
  static Network loadNetworkWithCudaFallback(Path networkPath, boolean useCuda) throws IOException {
    if (useCuda) {
      try {
        return NetworkOnnx.loadCuda(networkPath);
      } catch (IOException | RuntimeException e) {
        System.err.println("[warn] CUDA EP unavailable, falling back to CPU EP: " + e.getMessage());
        // Fallback : loadAuto choisit ORT CPU EP pour .onnx, NetworkVectorApi pour .npz.
      }
    }
    return NetworkLoader.loadAuto(networkPath);
  }

  /**
   * Boucle UCI principale, factorisée en méthode publique pour permettre les tests d'intégration
   * (injection d'un {@link InputStream} alternatif au lieu de {@code System.in}).
   *
   * <p>Sort sur :
   *
   * <ul>
   *   <li>{@code QUIT} retourné par le handler ({@code quit} UCI reçu).
   *   <li>EOF sur l'{@code InputStream} ({@code readLine} retourne {@code null}).
   * </ul>
   *
   * <p>Le caller est responsable du cleanup ({@link UciAdapterState#close}).
   *
   * @param in source UTF-8, non null
   * @param state holder process-wide, non null
   * @throws IOException si la lecture échoue (rare en pratique)
   */
  public static void runLoop(InputStream in, UciAdapterState state) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
    String line;
    while ((line = reader.readLine()) != null) {
      UciCommand cmd = UciCommandParser.parse(line);
      UciCommandHandler.HandleAction action = UciCommandHandler.handle(cmd, state);
      if (action == UciCommandHandler.HandleAction.QUIT) {
        return;
      }
    }
    // EOF : la boucle se termine naturellement, cleanup par try-with-resources.
  }

  /**
   * Container immutable pour les arguments CLI parsés. Visibilité package-private pour tests.
   *
   * @param networkPath chemin vers le modèle {@code .npz}/{@code .onnx} (obligatoire)
   * @param debugAtBoot flag {@code --debug}
   * @param useCuda flag {@code --cuda}
   * @param nnCacheSize (ADR-018) flag {@code --nncache <int>} ; {@code 0} = cache désactivé
   *     (défaut)
   */
  record Args(String networkPath, boolean debugAtBoot, boolean useCuda, int nnCacheSize) {}

  /**
   * Parse les arguments CLI. Retourne null si invalides (caller appellera printUsage +
   * System.exit). Visibilité package-private pour tests directs.
   */
  static Args parseArgs(String[] args) {
    String networkPath = null;
    boolean debugAtBoot = false;
    boolean useCuda = false;
    int nnCacheSize = 0;
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--network" -> {
          if (i + 1 >= args.length) {
            System.err.println("--network requires a path argument");
            return null;
          }
          networkPath = args[i + 1];
          i++;
        }
        case "--debug" -> debugAtBoot = true;
        case "--cuda" -> useCuda = true;
        // (ADR-018) Cache d'évaluation NN optionnel. Valeur invalide / hors borne → ignorée
        // (reste 0 = désactivé), cohérent avec la tolérance UCI sur les options entières.
        case "--nncache" -> {
          if (i + 1 >= args.length) {
            System.err.println("--nncache requires an integer argument");
            return null;
          }
          try {
            nnCacheSize = Math.max(0, Integer.parseInt(args[i + 1].trim()));
          } catch (NumberFormatException e) {
            System.err.println("--nncache requires an integer argument, got: " + args[i + 1]);
            return null;
          }
          i++;
        }
        default -> {
          // Ignore unknown CLI flags pour tolérance (futurs ajouts compatibles).
        }
      }
    }
    if (networkPath == null) {
      System.err.println("Missing required argument: --network <path>");
      return null;
    }
    return new Args(networkPath, debugAtBoot, useCuda, nnCacheSize);
  }

  private static void printUsage() {
    System.err.println(
        "Usage: java --add-modules jdk.incubator.vector -jar nanozero-uci-1.3.0.jar "
            + "--network <path-to-npz-or-onnx> [--debug] [--cuda] [--nncache <entries>]");
  }
}
