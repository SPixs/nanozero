package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nanozero.nn.Network;

/** Exercises {@link ModelCache} cache/download/swap logic via the loader seam. */
class ModelCacheTest {

  private HttpServer server;
  private JobserverClient client;

  @TempDir Path modelsDir;

  @BeforeEach
  void startServer() throws IOException {
    server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
    server.start();
    int port = server.getAddress().getPort();
    WorkerConfig config =
        new WorkerConfig(
            "http://localhost:" + port,
            "k",
            "w",
            modelsDir,
            Duration.ofSeconds(10),
            Duration.ofSeconds(1));
    client = new JobserverClient(config);
  }

  @AfterEach
  void stopServer() {
    server.stop(0);
  }

  /** Serves a tiny blob for {@code /models/{version}/download}. */
  private void serveModel(int version, String body) {
    server.createContext(
        "/models/" + version + "/download",
        ex -> {
          byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
          ex.sendResponseHeaders(200, bytes.length);
          ex.getResponseBody().write(bytes);
          ex.close();
        });
  }

  // ---------------------------------------------------------------------------------------
  // pathFor
  // ---------------------------------------------------------------------------------------

  @Test
  void pathForFormatsVersionZeroPadded() {
    ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork());
    assertThat(cache.pathFor(7)).isEqualTo(modelsDir.resolve("model-v0007.onnx"));
    assertThat(cache.pathFor(123)).isEqualTo(modelsDir.resolve("model-v0123.onnx"));
    cache.close();
  }

  @Test
  void loadedVersionIsMinusOneInitially() {
    ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork());
    assertThat(cache.loadedVersion()).isEqualTo(-1);
    cache.close();
  }

  // ---------------------------------------------------------------------------------------
  // acquire : download-then-load
  // ---------------------------------------------------------------------------------------

  @Test
  void acquireDownloadsThenLoadsWhenFileAbsent() throws Exception {
    serveModel(2, "model-bytes");
    List<Path> loadedPaths = new ArrayList<>();
    ModelCache.NetworkLoader loader =
        p -> {
          loadedPaths.add(p);
          return new FakeNetwork();
        };
    try (ModelCache cache = new ModelCache(modelsDir, client, loader)) {
      Network net = cache.acquire(2);

      assertThat(net).isNotNull();
      assertThat(cache.loadedVersion()).isEqualTo(2);
      // The file was downloaded into the cache dir, then handed to the loader.
      Path expected = modelsDir.resolve("model-v0002.onnx");
      assertThat(Files.exists(expected)).isTrue();
      assertThat(loadedPaths).containsExactly(expected);
    }
  }

  @Test
  void acquireSkipsDownloadWhenFileAlreadyPresent() throws Exception {
    // Pre-place the file ; server has NO handler, so any download attempt would fail.
    Path local = modelsDir.resolve("model-v0004.onnx");
    Files.writeString(local, "preexisting");
    AtomicInteger loadCount = new AtomicInteger();
    ModelCache.NetworkLoader loader =
        p -> {
          loadCount.incrementAndGet();
          return new FakeNetwork();
        };
    try (ModelCache cache = new ModelCache(modelsDir, client, loader)) {
      Network net = cache.acquire(4);
      assertThat(net).isNotNull();
      assertThat(cache.loadedVersion()).isEqualTo(4);
      assertThat(loadCount.get()).isEqualTo(1);
    }
  }

  // ---------------------------------------------------------------------------------------
  // acquire : cache-hit (same version)
  // ---------------------------------------------------------------------------------------

  @Test
  void acquireReturnsCachedInstanceForSameVersion() throws Exception {
    Files.writeString(modelsDir.resolve("model-v0001.onnx"), "v1");
    AtomicInteger loadCount = new AtomicInteger();
    ModelCache.NetworkLoader loader =
        p -> {
          loadCount.incrementAndGet();
          return new FakeNetwork();
        };
    try (ModelCache cache = new ModelCache(modelsDir, client, loader)) {
      Network first = cache.acquire(1);
      Network second = cache.acquire(1);

      assertThat(second).isSameAs(first);
      assertThat(loadCount.get()).isEqualTo(1); // loaded only once
    }
  }

  // ---------------------------------------------------------------------------------------
  // acquire/release : refcount — le swap ne ferme jamais une version encore jouée
  // ---------------------------------------------------------------------------------------

  @Test
  void swapDefersCloseUntilLastReferenceReleased() throws Exception {
    Files.writeString(modelsDir.resolve("model-v0001.onnx"), "v1");
    Files.writeString(modelsDir.resolve("model-v0002.onnx"), "v2");
    List<FakeNetwork> created = new ArrayList<>();
    ModelCache.NetworkLoader loader =
        p -> {
          FakeNetwork n = new FakeNetwork();
          created.add(n);
          return n;
        };
    try (ModelCache cache = new ModelCache(modelsDir, client, loader)) {
      cache.acquire(1); // une partie joue encore la v1...
      cache.acquire(2); // ...quand la promotion v2 arrive

      assertThat(created).hasSize(2);
      // v1 encore référencée : PAS fermée sous les pieds de la partie en cours.
      assertThat(created.get(0).closeCount()).isEqualTo(0);
      assertThat(cache.loadedVersion()).isEqualTo(2);
      assertThat(cache.loadedCount()).isEqualTo(2);

      cache.release(1); // la partie v1 se termine → fermeture immédiate (non-courante, refs=0)
      assertThat(created.get(0).closeCount()).isEqualTo(1);
      assertThat(cache.loadedCount()).isEqualTo(1);

      cache.release(2); // version courante : reste chargée même à refs=0
      assertThat(created.get(1).closeCount()).isEqualTo(0);
      assertThat(cache.loadedCount()).isEqualTo(1);
    }
  }

  @Test
  void swapClosesPreviousNetworkWhenAlreadyReleased() throws Exception {
    Files.writeString(modelsDir.resolve("model-v0001.onnx"), "v1");
    Files.writeString(modelsDir.resolve("model-v0002.onnx"), "v2");
    List<FakeNetwork> created = new ArrayList<>();
    ModelCache.NetworkLoader loader =
        p -> {
          FakeNetwork n = new FakeNetwork();
          created.add(n);
          return n;
        };
    try (ModelCache cache = new ModelCache(modelsDir, client, loader)) {
      cache.acquire(1);
      cache.release(1); // plus aucune partie sur v1 (mais reste chargée : courante)
      assertThat(created.get(0).closeCount()).isEqualTo(0);

      cache.acquire(2); // le swap balaie les versions non-courantes sans référence
      assertThat(created.get(0).closeCount()).isEqualTo(1);
      assertThat(cache.loadedVersion()).isEqualTo(2);
      assertThat(cache.loadedCount()).isEqualTo(1);
    }
  }

  @Test
  void releaseOfUnknownVersionIsNoOp() {
    try (ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork())) {
      cache.release(42); // must not throw
      assertThat(cache.loadedVersion()).isEqualTo(-1);
    }
  }

  // ---------------------------------------------------------------------------------------
  // close
  // ---------------------------------------------------------------------------------------

  @Test
  void closeClosesLoadedNetworkAndResetsVersion() throws Exception {
    Files.writeString(modelsDir.resolve("model-v0003.onnx"), "v3");
    FakeNetwork[] holder = new FakeNetwork[1];
    ModelCache.NetworkLoader loader =
        p -> {
          holder[0] = new FakeNetwork();
          return holder[0];
        };
    ModelCache cache = new ModelCache(modelsDir, client, loader);
    cache.acquire(3);
    cache.close();

    assertThat(holder[0].closeCount()).isEqualTo(1);
    assertThat(cache.loadedVersion()).isEqualTo(-1);
  }

  @Test
  void closeIsNoOpWhenNothingLoaded() {
    ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork());
    cache.close(); // must not throw
    assertThat(cache.loadedVersion()).isEqualTo(-1);
  }

  // ---------------------------------------------------------------------------------------
  // closeLoaded best-effort exception branch
  // ---------------------------------------------------------------------------------------

  @Test
  void closeSwallowsExceptionFromNetworkClose() throws Exception {
    Files.writeString(modelsDir.resolve("model-v0006.onnx"), "v6");
    ModelCache.NetworkLoader loader = p -> FakeNetwork.closeableThrowing();
    ModelCache cache = new ModelCache(modelsDir, client, loader);
    cache.acquire(6);

    // close() must NOT propagate the IllegalStateException thrown by the network's close().
    cache.close();
    assertThat(cache.loadedVersion()).isEqualTo(-1);
  }

  // ---------------------------------------------------------------------------------------
  // constructor validation
  // ---------------------------------------------------------------------------------------

  @Test
  void rejectsNullLoader() {
    assertThatThrownBy(() -> new ModelCache(modelsDir, client, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("networkLoader");
  }

  @Test
  void rejectsNullModelsDir() {
    assertThatThrownBy(() -> new ModelCache(null, client))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("modelsDir");
  }

  @Test
  void rejectsNullClient() {
    assertThatThrownBy(() -> new ModelCache(modelsDir, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("client");
  }
}
