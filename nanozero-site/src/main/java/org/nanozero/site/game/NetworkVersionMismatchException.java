package org.nanozero.site.game;

/** La version réseau soumise ne correspond pas au champion courant → 422. */
public class NetworkVersionMismatchException extends RuntimeException {
  public NetworkVersionMismatchException() {
    super("network version mismatch");
  }
}
