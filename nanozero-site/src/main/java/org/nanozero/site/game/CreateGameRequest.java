package org.nanozero.site.game;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.Map;

/**
 * Payload de création d'une partie. Les champs diagnostic sont best-effort (nullables) : leur
 * absence n'invalide pas la requête. {@code pseudo} n'est PAS accepté au MVP (auth différée).
 */
public record CreateGameRequest(
    @NotBlank @Size(max = 65536) String pgn,
    @NotNull @Pattern(regexp = "[wb]") String playerColor,
    @NotNull @Pattern(regexp = "chill|club|full") String levelId,
    @NotNull @Pattern(regexp = "1-0|0-1|1/2-1/2|\\*") String result,
    @NotNull @Min(0) @Max(500) Integer plyCount,
    @NotBlank @Pattern(regexp = "gen-\\d+") String networkVersion,
    @Pattern(regexp = "mobile|desktop") String deviceClass,
    @Pattern(regexp = "webgpu|wasm") String backend,
    @Min(0) @Max(4096) Integer effectiveSims,
    Map<String, Object> client) {
}
