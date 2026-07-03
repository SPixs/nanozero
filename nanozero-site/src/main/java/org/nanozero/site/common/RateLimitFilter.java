package org.nanozero.site.common;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import org.nanozero.site.config.SiteProperties;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Rate-limit par IP (token-bucket maison, in-memory) — anti-abus sans comptes.
 * Buckets séparés POST / GET. L'IP (X-Forwarded-For via Caddy) sert UNIQUEMENT au throttle
 * transitoire ; elle n'est JAMAIS persistée.
 */
@Component
@Order(1)
public class RateLimitFilter extends OncePerRequestFilter {

  private final int postPerMinute;
  private final int getPerMinute;
  private final ConcurrentHashMap<String, Bucket> buckets = new ConcurrentHashMap<>();

  public RateLimitFilter(SiteProperties props) {
    this.postPerMinute = props.getRateLimit().getPostPerMinute();
    this.getPerMinute = props.getRateLimit().getGetPerMinute();
  }

  @Override
  protected void doFilterInternal(HttpServletRequest req, HttpServletResponse res, FilterChain chain)
      throws ServletException, IOException {
    boolean post = "POST".equalsIgnoreCase(req.getMethod());
    int limit = post ? postPerMinute : getPerMinute;
    String key = clientIp(req) + (post ? "|P" : "|G");
    // Garde-fou mémoire : remise à zéro si la table explose (échelle club → jamais atteint en pratique).
    if (buckets.size() > 100_000) buckets.clear();
    Bucket bucket = buckets.computeIfAbsent(key, k -> new Bucket(limit));
    if (!bucket.tryConsume()) {
      res.setStatus(429);
      res.setHeader("Retry-After", "30");
      res.setContentType("application/json");
      res.getWriter().write("{\"error\":\"RATE_LIMIT_EXCEEDED\",\"retryAfterSeconds\":30}");
      return;
    }
    chain.doFilter(req, res);
  }

  private static String clientIp(HttpServletRequest req) {
    String xff = req.getHeader("X-Forwarded-For");
    if (xff != null && !xff.isBlank()) return xff.split(",")[0].trim();
    return req.getRemoteAddr();
  }

  /** Token-bucket simple : capacité = N/minute, recharge continue. */
  private static final class Bucket {
    private final double capacity;
    private final double refillPerMs;
    private double tokens;
    private long lastMs;

    Bucket(int perMinute) {
      this.capacity = perMinute;
      this.refillPerMs = perMinute / 60_000.0;
      this.tokens = perMinute;
      this.lastMs = System.currentTimeMillis();
    }

    synchronized boolean tryConsume() {
      long now = System.currentTimeMillis();
      tokens = Math.min(capacity, tokens + (now - lastMs) * refillPerMs);
      lastMs = now;
      if (tokens >= 1.0) {
        tokens -= 1.0;
        return true;
      }
      return false;
    }
  }
}
