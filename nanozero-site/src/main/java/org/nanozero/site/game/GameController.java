package org.nanozero.site.game;

import jakarta.validation.Valid;
import java.time.Duration;
import org.nanozero.site.config.SiteProperties;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** API des parties persistées : {@code POST /api/site/games}, {@code GET /api/site/games/{shareId}}. */
@RestController
@RequestMapping("/games")
public class GameController {

  private final GameService service;
  private final SiteProperties props;

  public GameController(GameService service, SiteProperties props) {
    this.service = service;
    this.props = props;
  }

  @PostMapping
  public ResponseEntity<CreateGameResponse> create(@Valid @RequestBody CreateGameRequest req) {
    Game g = service.create(req);
    String url = props.getPublicBaseUrl() + "/play/?game=" + g.getShareId();
    return ResponseEntity.status(HttpStatus.CREATED)
        .body(new CreateGameResponse(g.getShareId(), url, g.getCreatedAt()));
  }

  @GetMapping("/{shareId}")
  public ResponseEntity<GameView> get(@PathVariable String shareId) {
    Game g = service.getByShareId(shareId);
    // Une partie persistée ne change jamais → cache long.
    return ResponseEntity.ok()
        .cacheControl(CacheControl.maxAge(Duration.ofHours(1)).cachePublic())
        .body(GameView.of(g));
  }
}
