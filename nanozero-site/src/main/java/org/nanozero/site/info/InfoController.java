package org.nanozero.site.info;

import java.util.Map;
import org.nanozero.site.config.SiteProperties;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/** {@code GET /api/site/info} — version réseau courante (le front l'annexe aux parties soumises). */
@RestController
public class InfoController {

  private final SiteProperties props;

  public InfoController(SiteProperties props) {
    this.props = props;
  }

  @GetMapping("/info")
  public Map<String, Object> info() {
    return Map.of(
        "networkVersion", props.getNetworkVersion(),
        "calibrationVersion", props.getCalibrationVersion());
  }
}
