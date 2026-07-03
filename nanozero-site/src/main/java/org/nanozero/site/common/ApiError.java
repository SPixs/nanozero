package org.nanozero.site.common;

/** Réponse d'erreur uniforme : {@code {"error": "...", "details": ...}}. */
public record ApiError(String error, Object details) {
  public ApiError(String error) {
    this(error, null);
  }
}
