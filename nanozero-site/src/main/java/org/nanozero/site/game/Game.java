package org.nanozero.site.game;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;
import java.util.Map;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

/** Une partie jouée persistée (table {@code site.games}). cf. backend-persistance-parties.md §5. */
@Entity
@Table(name = "games")
public class Game {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @Column(name = "share_id", nullable = false, length = 12)
  private String shareId;

  @Column(nullable = false)
  private String pgn;

  @Column(name = "player_color", nullable = false, length = 1)
  private String playerColor;

  @Column(name = "level_id", nullable = false, length = 16)
  private String levelId;

  @Column(nullable = false, length = 8)
  private String result;

  @Column(name = "ply_count", nullable = false)
  private short plyCount;

  @Column(name = "network_version", nullable = false, length = 32)
  private String networkVersion;

  // --- diagnostic / hardware (anonyme, grossier) ---
  @Column(name = "device_class", length = 8)
  private String deviceClass;

  @Column(length = 8)
  private String backend;

  @Column(name = "effective_sims")
  private Short effectiveSims;

  @JdbcTypeCode(SqlTypes.JSON)
  @Column(columnDefinition = "jsonb")
  private Map<String, Object> client;

  // --- auth différée / extensions ---
  @Column(length = 64)
  private String pseudo;

  @JdbcTypeCode(SqlTypes.JSON)
  @Column(columnDefinition = "jsonb")
  private Map<String, Object> metadata;

  @CreationTimestamp
  @Column(name = "created_at", nullable = false, updatable = false)
  private OffsetDateTime createdAt;

  public Long getId() { return id; }

  public String getShareId() { return shareId; }
  public void setShareId(String v) { this.shareId = v; }

  public String getPgn() { return pgn; }
  public void setPgn(String v) { this.pgn = v; }

  public String getPlayerColor() { return playerColor; }
  public void setPlayerColor(String v) { this.playerColor = v; }

  public String getLevelId() { return levelId; }
  public void setLevelId(String v) { this.levelId = v; }

  public String getResult() { return result; }
  public void setResult(String v) { this.result = v; }

  public short getPlyCount() { return plyCount; }
  public void setPlyCount(short v) { this.plyCount = v; }

  public String getNetworkVersion() { return networkVersion; }
  public void setNetworkVersion(String v) { this.networkVersion = v; }

  public String getDeviceClass() { return deviceClass; }
  public void setDeviceClass(String v) { this.deviceClass = v; }

  public String getBackend() { return backend; }
  public void setBackend(String v) { this.backend = v; }

  public Short getEffectiveSims() { return effectiveSims; }
  public void setEffectiveSims(Short v) { this.effectiveSims = v; }

  public Map<String, Object> getClient() { return client; }
  public void setClient(Map<String, Object> v) { this.client = v; }

  public String getPseudo() { return pseudo; }
  public void setPseudo(String v) { this.pseudo = v; }

  public Map<String, Object> getMetadata() { return metadata; }
  public void setMetadata(Map<String, Object> v) { this.metadata = v; }

  public OffsetDateTime getCreatedAt() { return createdAt; }
}
