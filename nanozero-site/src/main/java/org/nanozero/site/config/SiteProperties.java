package org.nanozero.site.config;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

/** Paramètres applicatifs (préfixe {@code nanozero.site}). */
@ConfigurationProperties(prefix = "nanozero.site")
public class SiteProperties {

  /** Version réseau courante, imposée aux parties persistées (pas un champ libre client). */
  private String networkVersion = "gen-031";

  /** Version de la calibration (exposée par /info). */
  private int calibrationVersion = 1;

  /** Base publique pour construire l'URL de partage. */
  private String publicBaseUrl = "https://nanozero.org";

  /** Origines CORS autorisées (same-origin en prod). */
  private List<String> corsOrigins = List.of("https://nanozero.org");

  private RateLimit rateLimit = new RateLimit();

  public String getNetworkVersion() { return networkVersion; }
  public void setNetworkVersion(String v) { this.networkVersion = v; }

  public int getCalibrationVersion() { return calibrationVersion; }
  public void setCalibrationVersion(int v) { this.calibrationVersion = v; }

  public String getPublicBaseUrl() { return publicBaseUrl; }
  public void setPublicBaseUrl(String v) { this.publicBaseUrl = v; }

  public List<String> getCorsOrigins() { return corsOrigins; }
  public void setCorsOrigins(List<String> v) { this.corsOrigins = v; }

  public RateLimit getRateLimit() { return rateLimit; }
  public void setRateLimit(RateLimit v) { this.rateLimit = v; }

  /** Seuils de rate-limit par IP (token-bucket maison, cf. RateLimitFilter). */
  public static class RateLimit {
    private int postPerMinute = 10;
    private int getPerMinute = 60;

    public int getPostPerMinute() { return postPerMinute; }
    public void setPostPerMinute(int v) { this.postPerMinute = v; }

    public int getGetPerMinute() { return getPerMinute; }
    public void setGetPerMinute(int v) { this.getPerMinute = v; }
  }
}
