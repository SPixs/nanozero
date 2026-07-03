package org.nanozero.worker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.Optional;
import java.util.zip.GZIPOutputStream;

/**
 * Thin HTTP wrapper around nanozero-jobserver. Stdlib {@code java.net.http.HttpClient} + the
 * in-house {@link Json} serializer. No external dependencies.
 *
 * <p>All methods raise {@link IOException} on transport failure or non-success HTTP codes (except
 * {@link #claim()} which converts {@code 204 No Content} to an empty Optional).
 */
public final class JobserverClient {

  private static final String API_KEY_HEADER = "X-API-Key";
  private static final String WORKER_ID_HEADER = "X-Worker-Id";

  private final HttpClient http;
  private final WorkerConfig config;

  public JobserverClient(WorkerConfig config) {
    this(config, defaultHttpClient(config));
  }

  /** Constructor for testing : inject a custom {@link HttpClient} (e.g., wrapping a mock). */
  public JobserverClient(WorkerConfig config, HttpClient http) {
    this.config = config;
    this.http = http;
  }

  private static HttpClient defaultHttpClient(WorkerConfig config) {
    // Force HTTP/1.1 : JDK HttpClient defaults to HTTP/2 and tries to upgrade via
    // "Connection: Upgrade, HTTP2-Settings" + "Upgrade: h2c". uvicorn (used by
    // nanozero-jobserver) does NOT honor h2c upgrade and silently drops the body
    // on requests carrying the upgrade hint. Symptom : every POST /jobs/{id}/submit
    // returns HTTP 422 with {"input": null} even though Content-Length is correct.
    // Live diagnosed 2026-05-16 with -Djdk.httpclient.HttpClient.log=requests,headers.
    return HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .connectTimeout(config.requestTimeout())
        .build();
  }

  // ---------------------------------------------------------------------------------------
  // Claim
  // ---------------------------------------------------------------------------------------

  /**
   * POST {@code /jobs/claim}. Returns the parsed job descriptor, or empty if the server returned
   * 204 (queue empty).
   *
   * @throws IOException on transport error or non-200/204 status.
   * @throws InterruptedException if the calling thread is interrupted while waiting.
   */
  public Optional<Map<String, Object>> claim() throws IOException, InterruptedException {
    HttpRequest req =
        baseRequestBuilder("/jobs/claim").POST(HttpRequest.BodyPublishers.noBody()).build();
    HttpResponse<String> resp = http.send(req, BodyHandlers.ofString());
    if (resp.statusCode() == 204) {
      return Optional.empty();
    }
    if (resp.statusCode() != 200) {
      throw new IOException("claim returned HTTP " + resp.statusCode() + ": " + resp.body());
    }
    return Optional.of(Json.parseObject(resp.body()));
  }

  // ---------------------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------------------

  /**
   * POST {@code /jobs/{id}/submit} with the given JSON-serializable body.
   *
   * @return the parsed response body.
   * @throws IOException on transport error or non-200 status (e.g., 410 if the job is gone).
   */
  public Map<String, Object> submit(String jobId, Map<String, Object> body)
      throws IOException, InterruptedException {
    // gzip the body : the dense input planes (~30 KB) + policy target (~18.7 KB) per position
    // compress ~3-5x, slashing DevSrv WiFi bandwidth on self-play submits. The jobserver inflates
    // transparently (GzipRequestMiddleware) and stays backward-compatible with un-gzipped workers.
    byte[] gz = gzip(Json.write(body).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    HttpRequest req =
        baseRequestBuilder("/jobs/" + urlEncode(jobId) + "/submit")
            .header("Content-Type", "application/json")
            .header("Content-Encoding", "gzip")
            .POST(HttpRequest.BodyPublishers.ofByteArray(gz))
            .build();
    HttpResponse<String> resp = http.send(req, BodyHandlers.ofString());
    if (resp.statusCode() != 200) {
      throw new IOException("submit returned HTTP " + resp.statusCode() + ": " + resp.body());
    }
    return Json.parseObject(resp.body());
  }

  /** Gzip a byte array — compresses the self-play submit body for the wire. */
  private static byte[] gzip(byte[] data) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream(Math.max(64, data.length / 3));
    try (GZIPOutputStream gz = new GZIPOutputStream(baos)) {
      gz.write(data);
    }
    return baos.toByteArray();
  }

  // ---------------------------------------------------------------------------------------
  // Model download
  // ---------------------------------------------------------------------------------------

  /**
   * GET {@code /models/{version}/download} and stream-write the body to {@code targetPath}. Atomic
   * via rename : writes to {@code targetPath + ".part"} first, then moves.
   *
   * @throws IOException on transport error, non-200 status, or local write error.
   */
  public void downloadModel(int version, Path targetPath) throws IOException, InterruptedException {
    Files.createDirectories(targetPath.getParent());
    Path tmp = targetPath.resolveSibling(targetPath.getFileName() + ".part");

    HttpRequest req = baseRequestBuilder("/models/" + version + "/download").GET().build();
    HttpResponse<InputStream> resp = http.send(req, BodyHandlers.ofInputStream());
    if (resp.statusCode() != 200) {
      try (InputStream body = resp.body()) {
        body.readAllBytes(); // drain
      }
      throw new IOException("downloadModel v" + version + " returned HTTP " + resp.statusCode());
    }
    try (InputStream body = resp.body()) {
      Files.copy(body, tmp, StandardCopyOption.REPLACE_EXISTING);
    }
    Files.move(
        tmp, targetPath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
  }

  // ---------------------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------------------

  private HttpRequest.Builder baseRequestBuilder(String path) {
    return HttpRequest.newBuilder()
        .uri(URI.create(config.serverUrl() + path))
        .timeout(config.requestTimeout())
        .header(API_KEY_HEADER, config.apiKey())
        .header(WORKER_ID_HEADER, config.workerId());
  }

  /** Minimal URL-encode for path segments (the job id is a UUID4 so just %-encode safety chars). */
  private static String urlEncode(String s) {
    return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8)
        .replace("+", "%20");
  }
}
