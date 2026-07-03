package org.nanozero.worker;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkOnnx;

/**
 * Local on-disk cache of model {@code .onnx} files by version + in-memory {@link Network} instances
 * <b>refcountées</b> (multi-parties v1.6.0, ADR-001-worker Phase 3).
 *
 * <p>Design:
 *
 * <ul>
 *   <li>Files cached as {@code modelsDir/model-v{NNN}.onnx} (atomic-renamed by JobserverClient).
 *   <li>{@link #acquire(int)} / {@link #release(int)} : chaque partie en cours détient une
 *       référence sur la version qu'elle joue. Lors d'une promotion (nouvelle version demandée
 *       pendant que des parties jouent encore l'ancienne), l'ancien {@link Network} n'est fermé que
 *       lorsque sa dernière partie le relâche — jamais sous les pieds d'un thread de jeu.
 *   <li>Thread-safe : méthodes synchronized. Le téléchargement d'une version manquante se fait sous
 *       le verrou — les autres threads attendent puis touchent le cache (un seul download, pas de
 *       thundering herd intra-process).
 * </ul>
 */
public final class ModelCache implements AutoCloseable {

  private final Path modelsDir;
  private final JobserverClient client;
  private final NetworkLoader networkLoader;

  /** Versions chargées en mémoire, avec leur compte de références. */
  private static final class Entry {
    final Network network;
    int refs;

    Entry(Network network) {
      this.network = network;
    }
  }

  private final Map<Integer, Entry> entries = new HashMap<>();
  private int currentVersion = -1;

  /**
   * Loads a {@link Network} from a local model file. Testability seam — the production default is
   * {@link NetworkOnnx#load(Path)} (which throws {@link IOException}, so a plain {@link
   * java.util.function.Function} cannot model it); tests inject a fake returning an in-memory
   * Network without an ORT runtime. Le mode GPU injecte {@code path -> new
   * BatchingNetwork(NetworkOnnx.loadCuda(path), ..., true)} (cf. WorkerMain).
   */
  @FunctionalInterface
  public interface NetworkLoader {
    Network load(Path path) throws IOException;
  }

  /**
   * Production constructor : loads {@code .onnx} files via {@link NetworkOnnx#load(Path)}.
   *
   * @param modelsDir local cache directory for {@code .onnx} files.
   * @param client jobserver client used to download missing model versions.
   */
  public ModelCache(Path modelsDir, JobserverClient client) {
    this(modelsDir, client, NetworkOnnx::load);
  }

  /**
   * Seam constructor for testability + sélection de backend : the {@code networkLoader} decouples
   * {@link #acquire} from the ONNX Runtime native binding.
   *
   * @param modelsDir local cache directory for model files.
   * @param client jobserver client used to download missing model versions.
   * @param networkLoader maps a local model file path to a loaded {@link Network}.
   */
  public ModelCache(Path modelsDir, JobserverClient client, NetworkLoader networkLoader) {
    this.modelsDir = Objects.requireNonNull(modelsDir, "modelsDir");
    this.client = Objects.requireNonNull(client, "client");
    this.networkLoader = Objects.requireNonNull(networkLoader, "networkLoader");
  }

  /**
   * Acquiert une référence sur le {@link Network} de la version donnée, en le
   * téléchargeant/chargeant si nécessaire. DOIT être apparié à un {@link #release(int)} (pattern
   * {@code try/finally} dans WorkerLoop).
   *
   * @param version target model version (≥ 1).
   * @return loaded Network, ready for {@code Engine} construction.
   * @throws IOException on download or load failure.
   * @throws InterruptedException if the calling thread is interrupted during HTTP download.
   */
  public synchronized Network acquire(int version) throws IOException, InterruptedException {
    Entry entry = entries.get(version);
    if (entry == null) {
      Path localPath = pathFor(version);
      if (!Files.exists(localPath)) {
        client.downloadModel(version, localPath);
      }
      entry = new Entry(networkLoader.load(localPath));
      entries.put(version, entry);
    }
    entry.refs++;
    currentVersion = version;
    closeStaleUnreferenced();
    return entry.network;
  }

  /**
   * Relâche une référence acquise par {@link #acquire(int)}. Si la version n'est plus la courante
   * et n'a plus aucune référence, son {@link Network} est fermé immédiatement.
   *
   * @param version la version passée à {@code acquire}.
   */
  public synchronized void release(int version) {
    Entry entry = entries.get(version);
    if (entry == null) {
      return;
    }
    entry.refs = Math.max(0, entry.refs - 1);
    if (entry.refs == 0 && version != currentVersion) {
      entries.remove(version);
      closeNetwork(entry.network);
    }
  }

  /** Ferme les versions non-courantes dont plus aucune partie ne détient de référence. */
  private void closeStaleUnreferenced() {
    var it = entries.entrySet().iterator();
    while (it.hasNext()) {
      var e = it.next();
      if (e.getKey() != currentVersion && e.getValue().refs == 0) {
        it.remove();
        closeNetwork(e.getValue().network);
      }
    }
  }

  /** Currently-loaded model version (dernière acquise), or {@code -1} if none. */
  public synchronized int loadedVersion() {
    return currentVersion;
  }

  /** Nombre de versions actuellement chargées en mémoire (courante + en cours de drain). */
  synchronized int loadedCount() {
    return entries.size();
  }

  /** Local cache file path for a given version (does not check existence). */
  public Path pathFor(int version) {
    return modelsDir.resolve(String.format("model-v%04d.onnx", version));
  }

  @Override
  public synchronized void close() {
    for (Entry entry : entries.values()) {
      closeNetwork(entry.network);
    }
    entries.clear();
    currentVersion = -1;
  }

  private static void closeNetwork(Network network) {
    if (network instanceof AutoCloseable c) {
      try {
        c.close();
      } catch (Exception e) {
        // Best effort — log to stderr, don't propagate.
        System.err.println("[ModelCache] failed to close previous network: " + e);
      }
    }
  }
}
