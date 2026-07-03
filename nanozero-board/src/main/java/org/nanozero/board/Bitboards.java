package org.nanozero.board;

/**
 * Tables d'attaque de bitboards et utilitaires associés (cf. SPEC §5.5, ADR-003).
 *
 * <p>Le bit {@code i} d'un {@code long} bitboard représente la case d'index {@code i} sous la
 * convention LSB = a1 (cf. ADR-008, SPEC §3.1).
 *
 * <p>Les tables d'attaque non-sliders (PAWN, KNIGHT, KING) sont calculées par énumération directe
 * des offsets. Les tables d'attaque sliders (BISHOP, ROOK) sont obtenues par <em>magic
 * bitboards</em> dont les multiplicateurs sont recherchés au static init via un PRNG xorshift64*
 * seedé à {@value #MAGIC_SEARCH_SEED} pour reproductibilité bit-à-bit (cf. ADR-003, SPEC §5.5.5).
 *
 * <p>Toutes les tables sont {@code static final} et en lecture seule après l'initialisation
 * statique. Cette classe est donc thread-safe pour la lecture concurrente (cf. SPEC §10.1).
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Bitboards {

  private Bitboards() {
    throw new AssertionError("Non-instantiable");
  }

  // ---------------------------------------------------------------------------------------------
  // Constantes internes
  // ---------------------------------------------------------------------------------------------

  /** Index interne du slider type BISHOP dans les tables magic à deux dimensions. */
  private static final int BISHOP = 0;

  /** Index interne du slider type ROOK dans les tables magic à deux dimensions. */
  private static final int ROOK = 1;

  /** Seed du PRNG de recherche des multiplicateurs magiques (cf. SPEC §5.5.5, ADR-003). */
  private static final long MAGIC_SEARCH_SEED = 0x9E3779B97F4A7C15L;

  /** Multiplicateur de la version « starred » du PRNG xorshift64* (Marsaglia, SPEC §5.6.1). */
  private static final long XORSHIFT64_MULT = 0x2545F4914F6CDD1DL;

  /** Borne supérieure du nombre d'essais pour la recherche d'un magic (SPEC §5.5.3). */
  private static final int MAGIC_SEARCH_MAX_ATTEMPTS = 100_000_000;

  // ---------------------------------------------------------------------------------------------
  // Tables d'attaque non-sliders
  // ---------------------------------------------------------------------------------------------

  /** {@code PAWN_ATTACKS[color][square]} : bitboard des attaques d'un pion sur une case. */
  private static final long[][] PAWN_ATTACKS = new long[2][64];

  /** {@code KNIGHT_ATTACKS[square]} : bitboard des attaques d'un cavalier. */
  private static final long[] KNIGHT_ATTACKS = new long[64];

  /** {@code KING_ATTACKS[square]} : bitboard des attaques d'un roi. */
  private static final long[] KING_ATTACKS = new long[64];

  // ---------------------------------------------------------------------------------------------
  // Tables magic pour sliders (BISHOP, ROOK)
  // ---------------------------------------------------------------------------------------------

  /** Mask de pertinence (bords exclus) pour chaque (square, sliderType). */
  private static final long[][] SLIDER_RELEVANT_MASK = new long[64][2];

  /** Multiplicateur magique pour chaque (square, sliderType). */
  private static final long[][] SLIDER_MAGIC = new long[64][2];

  /** Décalage pour chaque (square, sliderType). {@code shift = 64 - popcount(mask)}. */
  private static final int[][] SLIDER_SHIFT = new int[64][2];

  /** Table d'attaques par occupancy magic-hashée pour chaque (square, sliderType). */
  private static final long[][][] SLIDER_ATTACK_TABLE = new long[64][2][];

  // ---------------------------------------------------------------------------------------------
  // Tables utilitaires
  // ---------------------------------------------------------------------------------------------

  /** {@code FILE_BB[file]} : bitboard des 8 cases d'un fichier (file 0=a..7=h). */
  private static final long[] FILE_BB = new long[8];

  /** {@code RANK_BB[rank]} : bitboard des 8 cases d'un rang (rank 0=rang 1..7=rang 8). */
  private static final long[] RANK_BB = new long[8];

  /** {@code DIAGONAL_BB[square]} : diagonale (NE-SW, {@code rank-file} constant) passant par sq. */
  private static final long[] DIAGONAL_BB = new long[64];

  /**
   * {@code ANTI_DIAGONAL_BB[square]} : anti-diagonale ({@code rank+file} constant) passant par sq.
   */
  private static final long[] ANTI_DIAGONAL_BB = new long[64];

  /** {@code BETWEEN_BB[a][b]} : cases strictement entre a et b si alignés, sinon 0. */
  private static final long[][] BETWEEN_BB = new long[64][64];

  /** {@code LINE_THROUGH_BB[a][b]} : ligne complète passant par a et b si alignés, sinon 0. */
  private static final long[][] LINE_THROUGH_BB = new long[64][64];

  // ---------------------------------------------------------------------------------------------
  // Constantes publiques : cases de couleur (cf. SPEC §6.4)
  // ---------------------------------------------------------------------------------------------

  /**
   * Bitboard des cases blanches (cases « claires » de l'échiquier) : a8, c8, ..., g1.
   *
   * <p>Utilisé notamment par la détection FIDE de matériel insuffisant pour vérifier que tous les
   * fous sont sur des cases de même couleur (cf. SPEC §6.4).
   */
  public static final long LIGHT_SQUARES = 0x55AA55AA55AA55AAL;

  /**
   * Bitboard des cases noires (cases « sombres » de l'échiquier) : a1, c1, ..., h8.
   *
   * <p>Complémentaire de {@link #LIGHT_SQUARES} : {@code LIGHT_SQUARES | DARK_SQUARES == -1L}.
   */
  public static final long DARK_SQUARES = 0xAA55AA55AA55AA55L;

  // ---------------------------------------------------------------------------------------------
  // Initialisation statique
  // ---------------------------------------------------------------------------------------------

  static {
    initFilesAndRanks();
    initDiagonals();
    initNonSliderAttacks();
    initSliderMagics();
    initBetweenAndLineThrough();
  }

  /**
   * Force le chargement et l'initialisation des tables magiques et d'attaque.
   *
   * <p>Le static init s'exécute implicitement à la première utilisation de toute méthode publique
   * de cette classe (mécanisme d'initialisation des classes Java). Ce {@code warmup()} permet de
   * déclencher cette initialisation explicitement avant un benchmark, afin de ne pas mesurer le
   * coût d'init dans le résultat (cf. ADR-003, SPEC §10.3).
   *
   * <p>Le seul effet observable est le déclenchement du chargement de la classe : la méthode
   * elle-même n'effectue aucun calcul. Coût cible : &lt; 200 ms (cf. SPEC §5.5.5, §9.2).
   */
  public static void warmup() {
    // Le simple fait de référencer un membre statique (ici via cette méthode statique) provoque
    // le chargement et l'initialisation de la classe par la JVM si cela n'a pas déjà eu lieu.
  }

  // ---------------------------------------------------------------------------------------------
  // API publique : tables d'attaque
  // ---------------------------------------------------------------------------------------------

  /**
   * Retourne le bitboard des cases attaquées par un pion d'une couleur donnée depuis une case.
   *
   * @param color {@code Color.WHITE} ou {@code Color.BLACK}, soit 0 ou 1
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard des deux cases diagonales avant (avec masque correct sur les fichiers a et h)
   */
  public static long pawnAttacks(int color, int square) {
    return PAWN_ATTACKS[color][square];
  }

  /**
   * Retourne le bitboard des cases attaquées par un cavalier depuis une case.
   *
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard des cases attaquées
   */
  public static long knightAttacks(int square) {
    return KNIGHT_ATTACKS[square];
  }

  /**
   * Retourne le bitboard des cases attaquées par un roi depuis une case.
   *
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard des cases attaquées
   */
  public static long kingAttacks(int square) {
    return KING_ATTACKS[square];
  }

  /**
   * Retourne le bitboard des cases attaquées par un fou depuis une case, en tenant compte de
   * l'occupancy fournie. Utilise les tables magic.
   *
   * @param square index de case dans {@code [0, 63]}
   * @param occupancy bitboard des cases occupées (toutes pièces confondues)
   * @return bitboard des cases attaquées par le fou
   */
  public static long bishopAttacks(int square, long occupancy) {
    long mask = SLIDER_RELEVANT_MASK[square][BISHOP];
    long magic = SLIDER_MAGIC[square][BISHOP];
    int shift = SLIDER_SHIFT[square][BISHOP];
    int idx = (int) (((occupancy & mask) * magic) >>> shift);
    return SLIDER_ATTACK_TABLE[square][BISHOP][idx];
  }

  /**
   * Retourne le bitboard des cases attaquées par une tour depuis une case, en tenant compte de
   * l'occupancy fournie. Utilise les tables magic.
   *
   * @param square index de case dans {@code [0, 63]}
   * @param occupancy bitboard des cases occupées (toutes pièces confondues)
   * @return bitboard des cases attaquées par la tour
   */
  public static long rookAttacks(int square, long occupancy) {
    long mask = SLIDER_RELEVANT_MASK[square][ROOK];
    long magic = SLIDER_MAGIC[square][ROOK];
    int shift = SLIDER_SHIFT[square][ROOK];
    int idx = (int) (((occupancy & mask) * magic) >>> shift);
    return SLIDER_ATTACK_TABLE[square][ROOK][idx];
  }

  /**
   * Retourne le bitboard des cases attaquées par une dame depuis une case, en tenant compte de
   * l'occupancy fournie. Combinaison des attaques de fou et de tour.
   *
   * @param square index de case dans {@code [0, 63]}
   * @param occupancy bitboard des cases occupées (toutes pièces confondues)
   * @return bitboard des cases attaquées par la dame
   */
  public static long queenAttacks(int square, long occupancy) {
    return bishopAttacks(square, occupancy) | rookAttacks(square, occupancy);
  }

  // ---------------------------------------------------------------------------------------------
  // API publique : utilitaires bitboard
  // ---------------------------------------------------------------------------------------------

  /**
   * Retourne le bitboard d'une seule case ({@code 1L << square}).
   *
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard avec uniquement le bit {@code square} actif
   */
  public static long squareBB(int square) {
    return 1L << square;
  }

  /**
   * Retourne le bitboard des 8 cases d'un fichier.
   *
   * @param file fichier dans {@code [0, 7]} (a=0..h=7)
   * @return bitboard du fichier
   */
  public static long fileBB(int file) {
    return FILE_BB[file];
  }

  /**
   * Retourne le bitboard des 8 cases d'un rang.
   *
   * @param rank rang dans {@code [0, 7]} (rang 1=0..rang 8=7)
   * @return bitboard du rang
   */
  public static long rankBB(int rank) {
    return RANK_BB[rank];
  }

  /**
   * Retourne le bitboard de la diagonale (NE-SW, {@code rank - file} constant) passant par la case
   * donnée.
   *
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard de la diagonale
   */
  public static long diagonalBB(int square) {
    return DIAGONAL_BB[square];
  }

  /**
   * Retourne le bitboard de l'anti-diagonale ({@code rank + file} constant) passant par la case
   * donnée.
   *
   * @param square index de case dans {@code [0, 63]}
   * @return bitboard de l'anti-diagonale
   */
  public static long antiDiagonalBB(int square) {
    return ANTI_DIAGONAL_BB[square];
  }

  /**
   * Retourne le bitboard des cases strictement entre {@code sq1} et {@code sq2} sur leur ligne
   * commune (rang, fichier, diagonale ou anti-diagonale).
   *
   * @param sq1 première case dans {@code [0, 63]}
   * @param sq2 seconde case dans {@code [0, 63]}
   * @return bitboard des cases intermédiaires (exclusif), ou {@code 0L} si les cases ne sont pas
   *     alignées
   */
  public static long between(int sq1, int sq2) {
    return BETWEEN_BB[sq1][sq2];
  }

  /**
   * Retourne le bitboard de la ligne complète passant par {@code sq1} et {@code sq2}, cases
   * incluses, à travers tout l'échiquier.
   *
   * @param sq1 première case dans {@code [0, 63]}
   * @param sq2 seconde case dans {@code [0, 63]}
   * @return bitboard de la ligne complète, ou {@code 0L} si les cases ne sont pas alignées
   */
  public static long lineThrough(int sq1, int sq2) {
    return LINE_THROUGH_BB[sq1][sq2];
  }

  // ---------------------------------------------------------------------------------------------
  // Initialisation : files, ranks, diagonales
  // ---------------------------------------------------------------------------------------------

  private static void initFilesAndRanks() {
    for (int f = 0; f < 8; f++) {
      long bb = 0L;
      for (int r = 0; r < 8; r++) {
        bb |= 1L << (r * 8 + f);
      }
      FILE_BB[f] = bb;
    }
    for (int r = 0; r < 8; r++) {
      long bb = 0L;
      for (int f = 0; f < 8; f++) {
        bb |= 1L << (r * 8 + f);
      }
      RANK_BB[r] = bb;
    }
  }

  private static void initDiagonals() {
    for (int sq = 0; sq < 64; sq++) {
      int sf = sq & 7;
      int sr = sq >>> 3;
      long diag = 0L;
      long anti = 0L;
      for (int s = 0; s < 64; s++) {
        int f = s & 7;
        int r = s >>> 3;
        if (r - f == sr - sf) {
          diag |= 1L << s;
        }
        if (r + f == sr + sf) {
          anti |= 1L << s;
        }
      }
      DIAGONAL_BB[sq] = diag;
      ANTI_DIAGONAL_BB[sq] = anti;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Initialisation : attack tables non-sliders
  // ---------------------------------------------------------------------------------------------

  private static void initNonSliderAttacks() {
    for (int sq = 0; sq < 64; sq++) {
      KNIGHT_ATTACKS[sq] = computeKnightAttacks(sq);
      KING_ATTACKS[sq] = computeKingAttacks(sq);
      PAWN_ATTACKS[0][sq] = computePawnAttacks(sq, 0);
      PAWN_ATTACKS[1][sq] = computePawnAttacks(sq, 1);
    }
  }

  private static long computeKnightAttacks(int square) {
    int sf = square & 7;
    int sr = square >>> 3;
    int[][] offsets = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    long bb = 0L;
    for (int[] o : offsets) {
      int f = sf + o[0];
      int r = sr + o[1];
      if (f >= 0 && f < 8 && r >= 0 && r < 8) {
        bb |= 1L << (r * 8 + f);
      }
    }
    return bb;
  }

  private static long computeKingAttacks(int square) {
    int sf = square & 7;
    int sr = square >>> 3;
    long bb = 0L;
    for (int df = -1; df <= 1; df++) {
      for (int dr = -1; dr <= 1; dr++) {
        if (df == 0 && dr == 0) {
          continue;
        }
        int f = sf + df;
        int r = sr + dr;
        if (f >= 0 && f < 8 && r >= 0 && r < 8) {
          bb |= 1L << (r * 8 + f);
        }
      }
    }
    return bb;
  }

  /**
   * Calcule les cases attaquées par un pion d'une couleur donnée depuis {@code square}.
   *
   * <p>Pour {@code color == 0} (WHITE) : attaques diagonales NE et NW. Pour {@code color == 1}
   * (BLACK) : attaques diagonales SE et SW. Aucune attaque n'est générée si le pion serait sur le
   * dernier rang dans sa direction de marche (cas en théorie impossible : un pion ne reste pas en
   * rang de promotion).
   */
  private static long computePawnAttacks(int square, int color) {
    int sf = square & 7;
    int sr = square >>> 3;
    long bb = 0L;
    int dr = (color == 0) ? 1 : -1;
    int nr = sr + dr;
    if (nr < 0 || nr > 7) {
      return 0L;
    }
    if (sf - 1 >= 0) {
      bb |= 1L << (nr * 8 + (sf - 1));
    }
    if (sf + 1 <= 7) {
      bb |= 1L << (nr * 8 + (sf + 1));
    }
    return bb;
  }

  // ---------------------------------------------------------------------------------------------
  // Initialisation : tables magic pour sliders
  // ---------------------------------------------------------------------------------------------

  private static void initSliderMagics() {
    long state = MAGIC_SEARCH_SEED;
    for (int sliderType = BISHOP; sliderType <= ROOK; sliderType++) {
      for (int sq = 0; sq < 64; sq++) {
        long mask = (sliderType == BISHOP) ? bishopRelevantMask(sq) : rookRelevantMask(sq);
        int nbBits = Long.bitCount(mask);
        int nbVariations = 1 << nbBits;
        int shift = 64 - nbBits;

        long[] occupancies = new long[nbVariations];
        long[] referenceAttacks = new long[nbVariations];
        for (int i = 0; i < nbVariations; i++) {
          long occ = Long.expand(i, mask);
          occupancies[i] = occ;
          referenceAttacks[i] =
              (sliderType == BISHOP) ? bishopAttacksSlow(sq, occ) : rookAttacksSlow(sq, occ);
        }

        long[] used = new long[nbVariations];
        int[] usedAttempt = new int[nbVariations];

        long magic = 0L;
        boolean found = false;
        int attempt = 0;
        while (!found && attempt < MAGIC_SEARCH_MAX_ATTEMPTS) {
          attempt++;
          state = xorshift64Star(state);
          long candidate = state;
          state = xorshift64Star(state);
          candidate &= state;
          state = xorshift64Star(state);
          candidate &= state;

          // Heuristique de Tord Romstad : ne retenir que les magics dont le produit avec le mask
          // dispose d'au moins 6 bits hauts à 1, gage de bonne distribution (cf. SPEC §5.5.3).
          if (Long.bitCount((mask * candidate) & 0xFF00000000000000L) < 6) {
            continue;
          }

          boolean ok = true;
          for (int i = 0; i < nbVariations; i++) {
            int idx = (int) ((occupancies[i] * candidate) >>> shift);
            if (usedAttempt[idx] != attempt) {
              used[idx] = referenceAttacks[i];
              usedAttempt[idx] = attempt;
            } else if (used[idx] != referenceAttacks[i]) {
              ok = false;
              break;
            }
          }
          if (ok) {
            magic = candidate;
            found = true;
          }
        }

        if (!found) {
          throw new IllegalStateException(
              "Échec de recherche de magic pour square=" + sq + " sliderType=" + sliderType);
        }

        long[] table = new long[nbVariations];
        for (int i = 0; i < nbVariations; i++) {
          int idx = (int) ((occupancies[i] * magic) >>> shift);
          table[idx] = referenceAttacks[i];
        }

        SLIDER_RELEVANT_MASK[sq][sliderType] = mask;
        SLIDER_MAGIC[sq][sliderType] = magic;
        SLIDER_SHIFT[sq][sliderType] = shift;
        SLIDER_ATTACK_TABLE[sq][sliderType] = table;
      }
    }
  }

  /**
   * Calcule le mask de pertinence d'un fou en {@code square} : cases sur les diagonales depuis
   * {@code square} dont l'occupation peut influer sur l'attaque, à l'exclusion des cases de bord de
   * l'échiquier (rangs 1, 8 ; fichiers a, h).
   */
  private static long bishopRelevantMask(int square) {
    int sf = square & 7;
    int sr = square >>> 3;
    long mask = 0L;
    int[][] dirs = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int[] d : dirs) {
      int f = sf + d[0];
      int r = sr + d[1];
      while (f >= 1 && f <= 6 && r >= 1 && r <= 6) {
        mask |= 1L << (r * 8 + f);
        f += d[0];
        r += d[1];
      }
    }
    return mask;
  }

  /**
   * Calcule le mask de pertinence d'une tour en {@code square} : cases sur les lignes droites
   * depuis {@code square}, à l'exclusion des cases de bord de l'échiquier sur ces mêmes lignes.
   */
  private static long rookRelevantMask(int square) {
    int sf = square & 7;
    int sr = square >>> 3;
    long mask = 0L;
    for (int r = sr + 1; r <= 6; r++) {
      mask |= 1L << (r * 8 + sf);
    }
    for (int r = sr - 1; r >= 1; r--) {
      mask |= 1L << (r * 8 + sf);
    }
    for (int f = sf + 1; f <= 6; f++) {
      mask |= 1L << (sr * 8 + f);
    }
    for (int f = sf - 1; f >= 1; f--) {
      mask |= 1L << (sr * 8 + f);
    }
    return mask;
  }

  /**
   * Implémentation lente de référence pour les attaques de fou. Utilisée à l'init pour générer la
   * table de référence des attaques par occupancy, et exposée package-private pour les tests de
   * cross-validation.
   */
  static long bishopAttacksSlow(int square, long occupancy) {
    int[][] dirs = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    return rayAttacksSlow(square, occupancy, dirs);
  }

  /**
   * Implémentation lente de référence pour les attaques de tour. Utilisée à l'init pour générer la
   * table de référence des attaques par occupancy, et exposée package-private pour les tests.
   */
  static long rookAttacksSlow(int square, long occupancy) {
    int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    return rayAttacksSlow(square, occupancy, dirs);
  }

  /**
   * Calcule les attaques d'un slider via projection rayon-par-rayon. Pour chaque direction, on
   * avance case par case en s'arrêtant dès qu'une case occupée est rencontrée (la case occupée est
   * incluse dans les attaques car elle peut être prise).
   */
  private static long rayAttacksSlow(int square, long occupancy, int[][] dirs) {
    int sf = square & 7;
    int sr = square >>> 3;
    long bb = 0L;
    for (int[] d : dirs) {
      int f = sf + d[0];
      int r = sr + d[1];
      while (f >= 0 && f < 8 && r >= 0 && r < 8) {
        int s = r * 8 + f;
        bb |= 1L << s;
        if ((occupancy & (1L << s)) != 0L) {
          break;
        }
        f += d[0];
        r += d[1];
      }
    }
    return bb;
  }

  /**
   * Itération xorshift64* : applique la transformation xorshift64 puis multiplie par {@link
   * #XORSHIFT64_MULT}. Le multiplicateur final améliore l'équidistribution (variante « starred » de
   * Marsaglia, cf. SPEC §5.6.1).
   */
  private static long xorshift64Star(long state) {
    long s = state;
    s ^= s << 13;
    s ^= s >>> 7;
    s ^= s << 17;
    return s * XORSHIFT64_MULT;
  }

  // ---------------------------------------------------------------------------------------------
  // Initialisation : between & lineThrough
  // ---------------------------------------------------------------------------------------------

  private static void initBetweenAndLineThrough() {
    for (int a = 0; a < 64; a++) {
      for (int b = 0; b < 64; b++) {
        BETWEEN_BB[a][b] = computeBetween(a, b);
        LINE_THROUGH_BB[a][b] = computeLineThrough(a, b);
      }
    }
  }

  private static long computeBetween(int sq1, int sq2) {
    if (sq1 == sq2) {
      return 0L;
    }
    int f1 = sq1 & 7;
    int r1 = sq1 >>> 3;
    int f2 = sq2 & 7;
    int r2 = sq2 >>> 3;
    int df = Integer.signum(f2 - f1);
    int dr = Integer.signum(r2 - r1);
    boolean sameRank = (r1 == r2);
    boolean sameFile = (f1 == f2);
    boolean sameDiagonal = (Math.abs(f2 - f1) == Math.abs(r2 - r1));
    if (!sameRank && !sameFile && !sameDiagonal) {
      return 0L;
    }
    long bb = 0L;
    int f = f1 + df;
    int r = r1 + dr;
    while (f != f2 || r != r2) {
      bb |= 1L << (r * 8 + f);
      f += df;
      r += dr;
    }
    return bb;
  }

  private static long computeLineThrough(int sq1, int sq2) {
    if (sq1 == sq2) {
      return 0L;
    }
    int f1 = sq1 & 7;
    int r1 = sq1 >>> 3;
    int f2 = sq2 & 7;
    int r2 = sq2 >>> 3;
    if (r1 == r2) {
      return RANK_BB[r1];
    }
    if (f1 == f2) {
      return FILE_BB[f1];
    }
    if ((r2 - r1) == (f2 - f1)) {
      return DIAGONAL_BB[sq1];
    }
    if ((r2 - r1) == -(f2 - f1)) {
      return ANTI_DIAGONAL_BB[sq1];
    }
    return 0L;
  }
}
