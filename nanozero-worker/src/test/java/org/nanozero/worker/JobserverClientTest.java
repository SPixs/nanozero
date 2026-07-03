package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises {@link JobserverClient} against a real in-process {@link HttpServer} on port 0. */
class JobserverClientTest {

  private HttpServer server;
  private JobserverClient client;
  private final Map<String, String> capturedApiKey = new ConcurrentHashMap<>();
  private final Map<String, String> capturedWorkerId = new ConcurrentHashMap<>();

  @BeforeEach
  void startServer() throws IOException {
    server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
    server.start();
    int port = server.getAddress().getPort();
    WorkerConfig config =
        new WorkerConfig(
            "http://localhost:" + port,
            "secret-key",
            "worker-xyz",
            Path.of("/tmp/unused"),
            Duration.ofSeconds(10),
            Duration.ofSeconds(1));
    client = new JobserverClient(config);
  }

  @AfterEach
  void stopServer() {
    server.stop(0);
  }

  /** Registers a handler that records the auth headers it received under {@code key}. */
  private void handle(String path, String key, HttpHandler delegate) {
    server.createContext(
        path,
        exchange -> {
          capturedApiKey.put(key, header(exchange, "X-API-Key"));
          capturedWorkerId.put(key, header(exchange, "X-Worker-Id"));
          delegate.handle(exchange);
        });
  }

  private static String header(HttpExchange ex, String name) {
    String v = ex.getRequestHeaders().getFirst(name);
    return v == null ? "" : v;
  }

  private static void respond(HttpExchange ex, int status, String body) throws IOException {
    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
    if (status == 204) {
      ex.sendResponseHeaders(204, -1);
    } else {
      ex.sendResponseHeaders(status, bytes.length);
      ex.getResponseBody().write(bytes);
    }
    ex.close();
  }

  // ---------------------------------------------------------------------------------------
  // claim
  // ---------------------------------------------------------------------------------------

  @Test
  void claimReturnsParsedJobOn200() throws Exception {
    handle(
        "/jobs/claim",
        "claim",
        ex -> respond(ex, 200, "{\"job_id\":\"abc\",\"model_version\":3,\"num_sims\":7}"));

    Optional<Map<String, Object>> job = client.claim();

    assertThat(job).isPresent();
    assertThat(job.get().get("job_id")).isEqualTo("abc");
    assertThat(job.get().get("model_version")).isEqualTo(3L);
    // Auth headers were sent.
    assertThat(capturedApiKey.get("claim")).isEqualTo("secret-key");
    assertThat(capturedWorkerId.get("claim")).isEqualTo("worker-xyz");
  }

  @Test
  void claimReturnsEmptyOn204() throws Exception {
    handle("/jobs/claim", "claim", ex -> respond(ex, 204, ""));
    assertThat(client.claim()).isEmpty();
  }

  @Test
  void claimThrowsOnOtherStatus() {
    handle("/jobs/claim", "claim", ex -> respond(ex, 500, "boom"));
    assertThatThrownBy(() -> client.claim())
        .isInstanceOf(IOException.class)
        .hasMessageContaining("HTTP 500");
  }

  // ---------------------------------------------------------------------------------------
  // submit
  // ---------------------------------------------------------------------------------------

  @Test
  void submitReturnsParsedResponseOn200() throws Exception {
    handle(
        "/jobs/job-1/submit",
        "submit",
        ex -> {
          // Drain the request body so the connection completes cleanly.
          ex.getRequestBody().readAllBytes();
          respond(ex, 200, "{\"status\":\"ok\",\"stored\":2}");
        });

    Map<String, Object> resp = client.submit("job-1", Map.of("k", "v"));

    assertThat(resp.get("status")).isEqualTo("ok");
    assertThat(resp.get("stored")).isEqualTo(2L);
    assertThat(capturedApiKey.get("submit")).isEqualTo("secret-key");
  }

  @Test
  void submitThrowsOnNon200() {
    handle(
        "/jobs/job-2/submit",
        "submit",
        ex -> {
          ex.getRequestBody().readAllBytes();
          respond(ex, 410, "gone");
        });

    assertThatThrownBy(() -> client.submit("job-2", Map.of("k", "v")))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("HTTP 410");
  }

  @Test
  void submitWithUuidJobIdHitsExpectedPath() throws Exception {
    // UUID-style ids contain only URL-safe chars : urlEncode is a no-op, path is built verbatim.
    String jobId = "550e8400-e29b-41d4-a716-446655440000";
    handle(
        "/jobs/" + jobId + "/submit",
        "submit",
        ex -> {
          ex.getRequestBody().readAllBytes();
          respond(ex, 200, "{\"ok\":true}");
        });

    Map<String, Object> resp = client.submit(jobId, Map.of());
    assertThat(resp.get("ok")).isEqualTo(Boolean.TRUE);
  }

  // ---------------------------------------------------------------------------------------
  // downloadModel
  // ---------------------------------------------------------------------------------------

  @Test
  void downloadModelWritesFileOn200(@TempDir Path dir) throws Exception {
    byte[] payload = "ONNX-BYTES-12345".getBytes(StandardCharsets.UTF_8);
    server.createContext(
        "/models/5/download",
        ex -> {
          ex.sendResponseHeaders(200, payload.length);
          ex.getResponseBody().write(payload);
          ex.close();
        });

    Path target = dir.resolve("sub").resolve("model-v0005.onnx");
    client.downloadModel(5, target);

    assertThat(Files.exists(target)).isTrue();
    assertThat(Files.readAllBytes(target)).isEqualTo(payload);
    // The .part temp file was renamed away.
    assertThat(Files.exists(target.resolveSibling("model-v0005.onnx.part"))).isFalse();
  }

  @Test
  void downloadModelThrowsOnNon200(@TempDir Path dir) {
    server.createContext("/models/9/download", ex -> respond(ex, 404, "not found"));
    Path target = dir.resolve("model-v0009.onnx");

    assertThatThrownBy(() -> client.downloadModel(9, target))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("HTTP 404");
    assertThat(Files.exists(target)).isFalse();
  }
}
