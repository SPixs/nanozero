package org.nanozero.site.game;

import java.time.OffsetDateTime;

/** Réponse 201 : identifiant public + URL partageable. */
public record CreateGameResponse(String shareId, String url, OffsetDateTime createdAt) {
}
