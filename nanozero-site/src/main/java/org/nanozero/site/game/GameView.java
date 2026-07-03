package org.nanozero.site.game;

import java.time.OffsetDateTime;

/**
 * Projection publique d'une partie (lecture). Le diagnostic (device/backend/sims) n'est PAS exposé
 * dans l'UI publique — réservé à la BDD/stats (cf. story B2 AC-7).
 */
public record GameView(
    String shareId,
    String pgn,
    String playerColor,
    String levelId,
    String result,
    int plyCount,
    String networkVersion,
    OffsetDateTime createdAt) {

  public static GameView of(Game g) {
    return new GameView(
        g.getShareId(), g.getPgn(), g.getPlayerColor(), g.getLevelId(),
        g.getResult(), g.getPlyCount(), g.getNetworkVersion(), g.getCreatedAt());
  }
}
