package org.nanozero.site.common;

import java.util.List;
import org.nanozero.site.game.GameNotFoundException;
import org.nanozero.site.game.NetworkVersionMismatchException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/** Traduit les exceptions en réponses ApiError homogènes. */
@RestControllerAdvice
public class GlobalExceptionHandler {

  @ExceptionHandler(MethodArgumentNotValidException.class)
  public ResponseEntity<ApiError> onValidation(MethodArgumentNotValidException ex) {
    List<String> details = ex.getBindingResult().getFieldErrors().stream()
        .map(e -> e.getField() + ": " + e.getDefaultMessage())
        .toList();
    return ResponseEntity.badRequest().body(new ApiError("VALIDATION_FAILED", details));
  }

  @ExceptionHandler(NetworkVersionMismatchException.class)
  public ResponseEntity<ApiError> onVersionMismatch(NetworkVersionMismatchException ex) {
    return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
        .body(new ApiError("NETWORK_VERSION_MISMATCH"));
  }

  @ExceptionHandler(GameNotFoundException.class)
  public ResponseEntity<ApiError> onNotFound(GameNotFoundException ex) {
    return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new ApiError("GAME_NOT_FOUND"));
  }
}
