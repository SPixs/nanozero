package org.nanozero.site.game;

import java.security.SecureRandom;
import org.springframework.stereotype.Component;

/** Génère des identifiants publics base62 de 10 caractères (≈ 8,4·10^17 → non énumérables). */
@Component
public class ShareIdGenerator {

  private static final char[] ALPHABET =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".toCharArray();
  private static final int LENGTH = 10;

  private final SecureRandom random = new SecureRandom();

  public String next() {
    StringBuilder sb = new StringBuilder(LENGTH);
    for (int i = 0; i < LENGTH; i++) {
      sb.append(ALPHABET[random.nextInt(ALPHABET.length)]);
    }
    return sb.toString();
  }
}
