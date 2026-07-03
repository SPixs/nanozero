package org.nanozero.site.game;

import org.nanozero.site.config.SiteProperties;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Logique de persistance des parties (génération du shareId, garde version réseau). */
@Service
public class GameService {

  private final GameRepository repository;
  private final ShareIdGenerator shareIds;
  private final SiteProperties props;

  public GameService(GameRepository repository, ShareIdGenerator shareIds, SiteProperties props) {
    this.repository = repository;
    this.shareIds = shareIds;
    this.props = props;
  }

  @Transactional
  public Game create(CreateGameRequest req) {
    if (!props.getNetworkVersion().equals(req.networkVersion())) {
      throw new NetworkVersionMismatchException();
    }
    Game g = new Game();
    g.setShareId(uniqueShareId());
    g.setPgn(req.pgn());
    g.setPlayerColor(req.playerColor());
    g.setLevelId(req.levelId());
    g.setResult(req.result());
    g.setPlyCount((short) (int) req.plyCount());
    g.setNetworkVersion(req.networkVersion());
    g.setDeviceClass(req.deviceClass());
    g.setBackend(req.backend());
    g.setEffectiveSims(req.effectiveSims() == null ? null : (short) (int) req.effectiveSims());
    g.setClient(req.client());
    return repository.save(g);
  }

  @Transactional(readOnly = true)
  public Game getByShareId(String shareId) {
    return repository.findByShareId(shareId).orElseThrow(GameNotFoundException::new);
  }

  private String uniqueShareId() {
    for (int attempt = 0; attempt < 5; attempt++) {
      String candidate = shareIds.next();
      if (!repository.existsByShareId(candidate)) {
        return candidate;
      }
    }
    throw new IllegalStateException("impossible de générer un shareId unique");
  }
}
