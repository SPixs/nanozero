package org.nanozero.board;

/**
 * Constantes et utilitaires de manipulation des cases de l'échiquier.
 *
 * <p>Une case est représentée par un entier dans {@code [0, 63]} selon la convention LSB = a1 (cf.
 * ADR-008 et SPEC §3.1). La formule de conversion est {@code square = file + rank * 8} avec {@code
 * file ∈ [0, 7]} (a=0..h=7) et {@code rank ∈ [0, 7]} (rang 1=0..rang 8=7).
 *
 * <p>La constante {@link #NONE} ({@code -1}) sert de sentinelle pour « aucune case », par exemple
 * pour {@code epSquare} lorsqu'aucune prise en passant n'est disponible.
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC (ADR-007).
 */
public final class Square {

  private Square() {
    throw new AssertionError("Non-instantiable");
  }

  public static final int A1 = 0;
  public static final int B1 = 1;
  public static final int C1 = 2;
  public static final int D1 = 3;
  public static final int E1 = 4;
  public static final int F1 = 5;
  public static final int G1 = 6;
  public static final int H1 = 7;

  public static final int A2 = 8;
  public static final int B2 = 9;
  public static final int C2 = 10;
  public static final int D2 = 11;
  public static final int E2 = 12;
  public static final int F2 = 13;
  public static final int G2 = 14;
  public static final int H2 = 15;

  public static final int A3 = 16;
  public static final int B3 = 17;
  public static final int C3 = 18;
  public static final int D3 = 19;
  public static final int E3 = 20;
  public static final int F3 = 21;
  public static final int G3 = 22;
  public static final int H3 = 23;

  public static final int A4 = 24;
  public static final int B4 = 25;
  public static final int C4 = 26;
  public static final int D4 = 27;
  public static final int E4 = 28;
  public static final int F4 = 29;
  public static final int G4 = 30;
  public static final int H4 = 31;

  public static final int A5 = 32;
  public static final int B5 = 33;
  public static final int C5 = 34;
  public static final int D5 = 35;
  public static final int E5 = 36;
  public static final int F5 = 37;
  public static final int G5 = 38;
  public static final int H5 = 39;

  public static final int A6 = 40;
  public static final int B6 = 41;
  public static final int C6 = 42;
  public static final int D6 = 43;
  public static final int E6 = 44;
  public static final int F6 = 45;
  public static final int G6 = 46;
  public static final int H6 = 47;

  public static final int A7 = 48;
  public static final int B7 = 49;
  public static final int C7 = 50;
  public static final int D7 = 51;
  public static final int E7 = 52;
  public static final int F7 = 53;
  public static final int G7 = 54;
  public static final int H7 = 55;

  public static final int A8 = 56;
  public static final int B8 = 57;
  public static final int C8 = 58;
  public static final int D8 = 59;
  public static final int E8 = 60;
  public static final int F8 = 61;
  public static final int G8 = 62;
  public static final int H8 = 63;

  /** Sentinelle pour « aucune case ». */
  public static final int NONE = -1;

  /** Nombre total de cases sur l'échiquier. */
  public static final int NB_SQUARES = 64;

  /**
   * Retourne le fichier de la case (a=0..h=7).
   *
   * @param square index de case dans {@code [0, 63]}
   * @return fichier de la case dans {@code [0, 7]}
   */
  public static int file(int square) {
    return square & 7;
  }

  /**
   * Retourne le rang de la case (rang 1=0..rang 8=7).
   *
   * @param square index de case dans {@code [0, 63]}
   * @return rang de la case dans {@code [0, 7]}
   */
  public static int rank(int square) {
    return square >>> 3;
  }

  /**
   * Construit l'index de case à partir d'un fichier et d'un rang.
   *
   * @param file fichier dans {@code [0, 7]} (a=0..h=7)
   * @param rank rang dans {@code [0, 7]} (rang 1=0..rang 8=7)
   * @return index de case dans {@code [0, 63]}
   */
  public static int make(int file, int rank) {
    return rank * 8 + file;
  }

  /**
   * Convertit une case en notation algébrique (ex : {@code "e4"}, {@code "h1"}).
   *
   * @param square index de case dans {@code [0, 63]}
   * @return notation algébrique sur deux caractères
   * @throws IllegalArgumentException si {@code square} est hors plage {@code [0, 63]}
   */
  public static String toAlgebraic(int square) {
    if (square < 0 || square >= NB_SQUARES) {
      throw new IllegalArgumentException("Square hors plage [0,63] : " + square);
    }
    char fileChar = (char) ('a' + file(square));
    char rankChar = (char) ('1' + rank(square));
    return new String(new char[] {fileChar, rankChar});
  }

  /**
   * Décode une case depuis sa notation algébrique (ex : {@code "e4"} → 28).
   *
   * @param s notation algébrique sur deux caractères, lettre minuscule {@code [a-h]} suivie d'un
   *     chiffre {@code [1-8]}
   * @return index de case dans {@code [0, 63]}
   * @throws IllegalArgumentException si {@code s} est {@code null}, de longueur ≠ 2, ou contient
   *     des caractères hors plages autorisées
   */
  public static int fromAlgebraic(String s) {
    if (s == null || s.length() != 2) {
      throw new IllegalArgumentException("Notation algébrique invalide : " + s);
    }
    char fileChar = s.charAt(0);
    char rankChar = s.charAt(1);
    if (fileChar < 'a' || fileChar > 'h' || rankChar < '1' || rankChar > '8') {
      throw new IllegalArgumentException("Notation algébrique invalide : " + s);
    }
    return make(fileChar - 'a', rankChar - '1');
  }
}
