package org.nanozero.site;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/** Backend site nanozero.org (incrément 1 : persistance des parties jouées). cf. DECISIONS.md D11. */
@SpringBootApplication
@ConfigurationPropertiesScan
public class NanozeroSiteApplication {
  public static void main(String[] args) {
    SpringApplication.run(NanozeroSiteApplication.class, args);
  }
}
