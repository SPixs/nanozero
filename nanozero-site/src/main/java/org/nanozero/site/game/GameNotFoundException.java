package org.nanozero.site.game;

/** Aucune partie pour ce shareId → 404. */
public class GameNotFoundException extends RuntimeException {
  public GameNotFoundException() {
    super("game not found");
  }
}
