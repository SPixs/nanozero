package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.net.URL;
import java.net.URLClassLoader;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests unitaires de {@link Bitboards}. Le critère de complétion de la phase 1 (cf. SPEC §14 phase
 * 1) impose la validation des attaques de sliders sur 64 cases × 100 occupancies aléatoires via une
 * méthode lente de référence (ray-casting naïf), ainsi qu'un boot {@code warmup()} en moins de 200
 * ms.
 */
class BitboardsTest {

  // -----------------------------------------------------------------------------------------
  // Utilitaires de référence (calcul lent pour validation)
  // -----------------------------------------------------------------------------------------

  /** Calcul de référence des attaques d'un cavalier (énumération des 8 offsets). */
  private static long knightAttacksReference(int square) {
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

  /** Calcul de référence des attaques d'un roi. */
  private static long kingAttacksReference(int square) {
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

  /** Calcul de référence des attaques d'un pion (color 0=WHITE, 1=BLACK). */
  private static long pawnAttacksReference(int color, int square) {
    int sf = square & 7;
    int sr = square >>> 3;
    int dr = (color == 0) ? 1 : -1;
    int nr = sr + dr;
    if (nr < 0 || nr > 7) {
      return 0L;
    }
    long bb = 0L;
    if (sf - 1 >= 0) {
      bb |= 1L << (nr * 8 + (sf - 1));
    }
    if (sf + 1 <= 7) {
      bb |= 1L << (nr * 8 + (sf + 1));
    }
    return bb;
  }

  /** Calcul lent des attaques de slider via projection rayon-par-rayon. */
  private static long sliderAttacksReference(int square, long occupancy, int[][] dirs) {
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

  private static final int[][] BISHOP_DIRS = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  private static final int[][] ROOK_DIRS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  // -----------------------------------------------------------------------------------------
  // Tables d'attaque non-sliders
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("knightAttacks : conformes au calcul de référence sur les 64 cases")
  void knightAttacksAllSquares() {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(Bitboards.knightAttacks(sq))
          .as("knightAttacks(%d)", sq)
          .isEqualTo(knightAttacksReference(sq));
    }
  }

  @Test
  @DisplayName("knightAttacks : valeurs connues")
  void knightAttacksKnownValues() {
    long expectedB1 = (1L << Square.A3) | (1L << Square.C3) | (1L << Square.D2);
    assertThat(Bitboards.knightAttacks(Square.B1)).isEqualTo(expectedB1);

    long expectedD4 =
        (1L << Square.B3)
            | (1L << Square.B5)
            | (1L << Square.C2)
            | (1L << Square.C6)
            | (1L << Square.E2)
            | (1L << Square.E6)
            | (1L << Square.F3)
            | (1L << Square.F5);
    assertThat(Bitboards.knightAttacks(Square.D4)).isEqualTo(expectedD4);

    long expectedA1 = (1L << Square.B3) | (1L << Square.C2);
    assertThat(Bitboards.knightAttacks(Square.A1)).isEqualTo(expectedA1);
  }

  @Test
  @DisplayName("kingAttacks : conformes au calcul de référence sur les 64 cases")
  void kingAttacksAllSquares() {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(Bitboards.kingAttacks(sq))
          .as("kingAttacks(%d)", sq)
          .isEqualTo(kingAttacksReference(sq));
    }
  }

  @Test
  @DisplayName("kingAttacks : valeurs connues aux coins et au centre")
  void kingAttacksKnownValues() {
    long expectedA1 = (1L << Square.A2) | (1L << Square.B1) | (1L << Square.B2);
    assertThat(Bitboards.kingAttacks(Square.A1)).isEqualTo(expectedA1);

    long expectedH8 = (1L << Square.G7) | (1L << Square.G8) | (1L << Square.H7);
    assertThat(Bitboards.kingAttacks(Square.H8)).isEqualTo(expectedH8);

    long expectedE4 =
        (1L << Square.D3)
            | (1L << Square.D4)
            | (1L << Square.D5)
            | (1L << Square.E3)
            | (1L << Square.E5)
            | (1L << Square.F3)
            | (1L << Square.F4)
            | (1L << Square.F5);
    assertThat(Bitboards.kingAttacks(Square.E4)).isEqualTo(expectedE4);
  }

  @Test
  @DisplayName("pawnAttacks WHITE : conformes au calcul de référence sur les 64 cases")
  void pawnAttacksWhiteAllSquares() {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(Bitboards.pawnAttacks(0, sq))
          .as("pawnAttacks(WHITE, %d)", sq)
          .isEqualTo(pawnAttacksReference(0, sq));
    }
  }

  @Test
  @DisplayName("pawnAttacks BLACK : conformes au calcul de référence sur les 64 cases")
  void pawnAttacksBlackAllSquares() {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(Bitboards.pawnAttacks(1, sq))
          .as("pawnAttacks(BLACK, %d)", sq)
          .isEqualTo(pawnAttacksReference(1, sq));
    }
  }

  @Test
  @DisplayName("pawnAttacks : pas de wrap-around sur les fichiers a et h")
  void pawnAttacksFileEdges() {
    // Pion blanc en a2 : attaque uniquement b3 (pas de NW)
    assertThat(Bitboards.pawnAttacks(0, Square.A2)).isEqualTo(1L << Square.B3);
    // Pion blanc en h2 : attaque uniquement g3 (pas de NE)
    assertThat(Bitboards.pawnAttacks(0, Square.H2)).isEqualTo(1L << Square.G3);
    // Pion noir en a7 : attaque uniquement b6
    assertThat(Bitboards.pawnAttacks(1, Square.A7)).isEqualTo(1L << Square.B6);
    // Pion noir en h7 : attaque uniquement g6
    assertThat(Bitboards.pawnAttacks(1, Square.H7)).isEqualTo(1L << Square.G6);
  }

  // -----------------------------------------------------------------------------------------
  // Tables d'attaque sliders (validation 64 × 100 occupancies aléatoires)
  // -----------------------------------------------------------------------------------------

  /**
   * Fournit (square, trial, occupancy) pour un test paramétré. Génère 100 occupancies aléatoires
   * pour chacune des 64 cases avec un PRNG seedé pour reproductibilité.
   */
  static Stream<org.junit.jupiter.params.provider.Arguments> sliderTrials() {
    Random rng = new Random(0xC0FFEE12345L);
    var builder = Stream.<org.junit.jupiter.params.provider.Arguments>builder();
    for (int sq = 0; sq < 64; sq++) {
      for (int trial = 0; trial < 100; trial++) {
        long occupancy = rng.nextLong();
        builder.add(org.junit.jupiter.params.provider.Arguments.of(sq, trial, occupancy));
      }
    }
    return builder.build();
  }

  @ParameterizedTest(name = "sq={0} trial={1}")
  @MethodSource("sliderTrials")
  @DisplayName("bishopAttacks : 64 cases × 100 occupancies aléatoires conformes au ray-cast naïf")
  void bishopAttacksMatchesReference(int square, int trial, long occupancy) {
    long expected = sliderAttacksReference(square, occupancy, BISHOP_DIRS);
    assertThat(Bitboards.bishopAttacks(square, occupancy))
        .as("bishopAttacks(%d, occ=0x%016x)", square, occupancy)
        .isEqualTo(expected);
  }

  @ParameterizedTest(name = "sq={0} trial={1}")
  @MethodSource("sliderTrials")
  @DisplayName("rookAttacks : 64 cases × 100 occupancies aléatoires conformes au ray-cast naïf")
  void rookAttacksMatchesReference(int square, int trial, long occupancy) {
    long expected = sliderAttacksReference(square, occupancy, ROOK_DIRS);
    assertThat(Bitboards.rookAttacks(square, occupancy))
        .as("rookAttacks(%d, occ=0x%016x)", square, occupancy)
        .isEqualTo(expected);
  }

  @ParameterizedTest(name = "sq={0} trial={1}")
  @MethodSource("sliderTrials")
  @DisplayName("queenAttacks : 64 cases × 100 occupancies = bishop | rook")
  void queenAttacksEqualsBishopOrRook(int square, int trial, long occupancy) {
    long expected =
        sliderAttacksReference(square, occupancy, BISHOP_DIRS)
            | sliderAttacksReference(square, occupancy, ROOK_DIRS);
    assertThat(Bitboards.queenAttacks(square, occupancy))
        .as("queenAttacks(%d, occ=0x%016x)", square, occupancy)
        .isEqualTo(expected);
  }

  @Test
  @DisplayName("bishopAttacks : occupancy vide (rayonnement plein)")
  void bishopAttacksEmptyOccupancy() {
    long actual = Bitboards.bishopAttacks(Square.D4, 0L);
    long expected = sliderAttacksReference(Square.D4, 0L, BISHOP_DIRS);
    assertThat(actual).isEqualTo(expected);
    // Un fou en D4 avec échiquier vide : 13 cases attaquées (a1,b2,c3,e5,f6,g7,h8 +
    // a7,b6,c5,e3,f2,g1)
    assertThat(Long.bitCount(actual)).isEqualTo(13);
  }

  @Test
  @DisplayName("rookAttacks : occupancy vide (toute la croix)")
  void rookAttacksEmptyOccupancy() {
    long actual = Bitboards.rookAttacks(Square.D4, 0L);
    long expected = sliderAttacksReference(Square.D4, 0L, ROOK_DIRS);
    assertThat(actual).isEqualTo(expected);
    // Tour en D4 avec échiquier vide : 14 cases (7 sur le fichier d, 7 sur le rang 4)
    assertThat(Long.bitCount(actual)).isEqualTo(14);
  }

  // -----------------------------------------------------------------------------------------
  // Utilitaires : squareBB, fileBB, rankBB, diagonalBB, antiDiagonalBB
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("squareBB(sq) == 1L << sq pour tous les sq")
  void squareBBAll() {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(Bitboards.squareBB(sq)).isEqualTo(1L << sq);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7})
  @DisplayName("fileBB : 8 bits sur le même fichier")
  void fileBBHasEightBits(int file) {
    long bb = Bitboards.fileBB(file);
    assertThat(Long.bitCount(bb)).isEqualTo(8);
    for (int rank = 0; rank < 8; rank++) {
      assertThat((bb >>> Square.make(file, rank)) & 1L).isEqualTo(1L);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7})
  @DisplayName("rankBB : 8 bits sur le même rang")
  void rankBBHasEightBits(int rank) {
    long bb = Bitboards.rankBB(rank);
    assertThat(Long.bitCount(bb)).isEqualTo(8);
    for (int file = 0; file < 8; file++) {
      assertThat((bb >>> Square.make(file, rank)) & 1L).isEqualTo(1L);
    }
  }

  @Test
  @DisplayName("fileBB et rankBB : valeurs canoniques")
  void fileBBRankBBKnownValues() {
    assertThat(Bitboards.fileBB(0)).isEqualTo(0x0101010101010101L); // fichier a
    assertThat(Bitboards.fileBB(7)).isEqualTo(0x8080808080808080L); // fichier h
    assertThat(Bitboards.rankBB(0)).isEqualTo(0x00000000000000FFL); // rang 1
    assertThat(Bitboards.rankBB(7)).isEqualTo(0xFF00000000000000L); // rang 8
  }

  @Test
  @DisplayName("diagonalBB : la diagonale a1-h8 contient 8 cases")
  void diagonalBBMainDiagonal() {
    long expected =
        (1L << Square.A1)
            | (1L << Square.B2)
            | (1L << Square.C3)
            | (1L << Square.D4)
            | (1L << Square.E5)
            | (1L << Square.F6)
            | (1L << Square.G7)
            | (1L << Square.H8);
    assertThat(Bitboards.diagonalBB(Square.A1)).isEqualTo(expected);
    assertThat(Bitboards.diagonalBB(Square.H8)).isEqualTo(expected);
    assertThat(Bitboards.diagonalBB(Square.D4)).isEqualTo(expected);
  }

  @Test
  @DisplayName("diagonalBB : la diagonale d'un coin H1 ne contient qu'une case")
  void diagonalBBSingleCorner() {
    assertThat(Bitboards.diagonalBB(Square.H1)).isEqualTo(1L << Square.H1);
    assertThat(Bitboards.diagonalBB(Square.A8)).isEqualTo(1L << Square.A8);
  }

  @Test
  @DisplayName("antiDiagonalBB : la diagonale a8-h1 contient 8 cases")
  void antiDiagonalBBMainAntiDiagonal() {
    long expected =
        (1L << Square.A8)
            | (1L << Square.B7)
            | (1L << Square.C6)
            | (1L << Square.D5)
            | (1L << Square.E4)
            | (1L << Square.F3)
            | (1L << Square.G2)
            | (1L << Square.H1);
    assertThat(Bitboards.antiDiagonalBB(Square.A8)).isEqualTo(expected);
    assertThat(Bitboards.antiDiagonalBB(Square.H1)).isEqualTo(expected);
    assertThat(Bitboards.antiDiagonalBB(Square.E4)).isEqualTo(expected);
  }

  @Test
  @DisplayName("antiDiagonalBB : coins isolés (a1 et h8) sont seuls sur leur anti-diagonale")
  void antiDiagonalBBIsolatedCorners() {
    assertThat(Bitboards.antiDiagonalBB(Square.A1)).isEqualTo(1L << Square.A1);
    assertThat(Bitboards.antiDiagonalBB(Square.H8)).isEqualTo(1L << Square.H8);
  }

  @Test
  @DisplayName("diagonalBB et antiDiagonalBB : cohérence avec rank-file et rank+file")
  void diagAntiDiagCoherence() {
    for (int sq = 0; sq < 64; sq++) {
      long diag = Bitboards.diagonalBB(sq);
      long anti = Bitboards.antiDiagonalBB(sq);
      int sf = Square.file(sq);
      int sr = Square.rank(sq);
      // Toute case de diag doit avoir rank-file == sr-sf.
      long b = diag;
      while (b != 0L) {
        int s = Long.numberOfTrailingZeros(b);
        b &= b - 1L;
        assertThat(Square.rank(s) - Square.file(s)).isEqualTo(sr - sf);
      }
      // Toute case de anti doit avoir rank+file == sr+sf.
      b = anti;
      while (b != 0L) {
        int s = Long.numberOfTrailingZeros(b);
        b &= b - 1L;
        assertThat(Square.rank(s) + Square.file(s)).isEqualTo(sr + sf);
      }
      // sq doit appartenir aux deux.
      assertThat((diag >>> sq) & 1L).isEqualTo(1L);
      assertThat((anti >>> sq) & 1L).isEqualTo(1L);
    }
  }

  // -----------------------------------------------------------------------------------------
  // between & lineThrough
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("between : cas alignés (rang, fichier, diagonale, anti-diagonale)")
  void betweenAlignedCases() {
    long b18 = Bitboards.between(Square.A1, Square.A8);
    assertThat(b18)
        .isEqualTo(
            (1L << Square.A2)
                | (1L << Square.A3)
                | (1L << Square.A4)
                | (1L << Square.A5)
                | (1L << Square.A6)
                | (1L << Square.A7));

    long b1h = Bitboards.between(Square.A1, Square.H1);
    assertThat(b1h)
        .isEqualTo(
            (1L << Square.B1)
                | (1L << Square.C1)
                | (1L << Square.D1)
                | (1L << Square.E1)
                | (1L << Square.F1)
                | (1L << Square.G1));

    long bdg = Bitboards.between(Square.A1, Square.H8);
    assertThat(bdg)
        .isEqualTo(
            (1L << Square.B2)
                | (1L << Square.C3)
                | (1L << Square.D4)
                | (1L << Square.E5)
                | (1L << Square.F6)
                | (1L << Square.G7));

    long banti = Bitboards.between(Square.A8, Square.H1);
    assertThat(banti)
        .isEqualTo(
            (1L << Square.B7)
                | (1L << Square.C6)
                | (1L << Square.D5)
                | (1L << Square.E4)
                | (1L << Square.F3)
                | (1L << Square.G2));
  }

  @Test
  @DisplayName("between : cases adjacentes ou identiques")
  void betweenAdjacentAndSame() {
    assertThat(Bitboards.between(Square.A1, Square.A1)).isZero();
    assertThat(Bitboards.between(Square.A1, Square.A2)).isZero();
    assertThat(Bitboards.between(Square.A1, Square.B2)).isZero();
    assertThat(Bitboards.between(Square.E4, Square.E5)).isZero();
  }

  @Test
  @DisplayName("between : cases non alignées renvoient 0")
  void betweenNonAligned() {
    assertThat(Bitboards.between(Square.A1, Square.B3)).isZero(); // pas un saut diagonal
    assertThat(Bitboards.between(Square.A1, Square.C2)).isZero();
    assertThat(Bitboards.between(Square.D4, Square.G7)).isNotZero(); // diagonale alignée
    assertThat(Bitboards.between(Square.D4, Square.G6)).isZero(); // pas alignée
  }

  @Test
  @DisplayName("between(a, b) == between(b, a)")
  void betweenIsSymmetric() {
    for (int a = 0; a < 64; a++) {
      for (int b = 0; b < 64; b++) {
        assertThat(Bitboards.between(a, b)).isEqualTo(Bitboards.between(b, a));
      }
    }
  }

  @Test
  @DisplayName("lineThrough : lignes complètes pour cases alignées")
  void lineThroughAlignedCases() {
    assertThat(Bitboards.lineThrough(Square.A1, Square.A8)).isEqualTo(Bitboards.fileBB(0));
    assertThat(Bitboards.lineThrough(Square.A1, Square.H1)).isEqualTo(Bitboards.rankBB(0));
    assertThat(Bitboards.lineThrough(Square.A1, Square.H8))
        .isEqualTo(Bitboards.diagonalBB(Square.A1));
    assertThat(Bitboards.lineThrough(Square.A8, Square.H1))
        .isEqualTo(Bitboards.antiDiagonalBB(Square.A8));
  }

  @Test
  @DisplayName("lineThrough : 0 si non alignées ou identiques")
  void lineThroughNonAlignedOrSame() {
    assertThat(Bitboards.lineThrough(Square.A1, Square.A1)).isZero();
    assertThat(Bitboards.lineThrough(Square.A1, Square.B3)).isZero();
    assertThat(Bitboards.lineThrough(Square.D4, Square.F5)).isZero();
  }

  @Test
  @DisplayName("lineThrough(a, b) == lineThrough(b, a) pour cases alignées")
  void lineThroughSymmetricForAligned() {
    int[][] pairs = {
      {Square.A1, Square.H8},
      {Square.A1, Square.H1},
      {Square.A8, Square.H1},
      {Square.D4, Square.D7},
      {Square.B2, Square.G7}
    };
    for (int[] p : pairs) {
      assertThat(Bitboards.lineThrough(p[0], p[1])).isEqualTo(Bitboards.lineThrough(p[1], p[0]));
    }
  }

  // -----------------------------------------------------------------------------------------
  // warmup
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("warmup ne lève pas d'exception et l'init est cohérent")
  void warmupIdempotent() {
    Bitboards.warmup();
    Bitboards.warmup();
    // Sanity post-warmup : les tables doivent répondre.
    assertThat(Bitboards.knightAttacks(Square.E4)).isNotZero();
    assertThat(Bitboards.bishopAttacks(Square.E4, 0L)).isNotZero();
    assertThat(Bitboards.rookAttacks(Square.E4, 0L)).isNotZero();
  }

  /**
   * Limite supérieure relâchée pour le test CI de boot. La cible SPEC §5.5.5 / §14 phase 1 est &lt;
   * 200 ms sur CPU desktop moderne (Zen 3+ ou Intel 12e gen+) avec JIT chauffée. Le test ci-
   * dessous mesure dans un {@link URLClassLoader} isolé : la classe {@code Bitboards} y est
   * recompilée à froid (le JIT n'a pas encore observé son code), ce qui ajoute un overhead notable.
   *
   * <p>Phase 10 : ce test s'exécute désormais dans un fork JVM dédié à {@code BitboardsTest} (cf.
   * configuration Surefire {@code bitboards-isolated} dans le POM, {@code reuseForks=false}). Le
   * fork élimine la pression GC induite par les autres suites de la même JVM (notamment chesslib
   * qui instancie ~10 000 {@code Board} dans la cross-validation), ce qui avait imposé un seuil
   * dégradé à 1500 ms en phase 3. Le seuil avait été resserré à 600 ms (marge au-dessus des
   * ~415-452 ms observés en phase 1). ⚠️ 2026-07-02 : ce seuil s'avère IRRÉALISTE sur du 12e gen
   * réel — DevSrv (i9-12900HK, cœurs hybrides E/P + throttle) mesure **1,1-1,4 s CONSTANTS** à froid
   * (classloader isolé → clinit interprété + génération des magic tables sans JIT), machine au
   * repos comme chargée. Ce test étant un GARDE-FOU contre une régression algorithmique MAJEURE (et
   * non la mesure fine, réservée à JMH dans {@code nanozero-bench}, SPEC §9.2/§9.4), le budget
   * passe à 2000 ms : il attrape toujours une init pathologique (×3+) sans flaker sur le matériel
   * cible réel.
   */
  private static final long WARMUP_BOOT_BUDGET_MS = 2000L;

  /**
   * Mesure le coût de l'initialisation statique ({@code <clinit>}) de {@link Bitboards} dans un
   * {@link URLClassLoader} isolé. La classe est d'abord chargée sans déclencher son init, puis on
   * isole la mesure du seul coût d'init (génération des tables d'attaque non-sliders + recherche
   * runtime des magic tables sliders).
   *
   * <p>Cf. {@link #WARMUP_BOOT_BUDGET_MS} pour la justification du seuil retenu, divergent de la
   * cible SPEC normative {@code < 200 ms}.
   */
  @Test
  @DisplayName("Bitboards init (<clinit>) en classloader isolé : durée mesurable et bornée")
  void warmupBootUnderBudget() throws Exception {
    URL classesUrl = Bitboards.class.getProtectionDomain().getCodeSource().getLocation();
    assertThat(classesUrl).isNotNull();

    try (URLClassLoader isolated =
        new URLClassLoader(new URL[] {classesUrl}, ClassLoader.getPlatformClassLoader())) {
      // 1) Charge le bytecode (lecture, verify, link) SANS déclencher <clinit>.
      Class<?> klass = Class.forName("org.nanozero.board.Bitboards", false, isolated);
      // 2) Mesure le seul coût de l'initialisation statique.
      long t0 = System.nanoTime();
      Class.forName("org.nanozero.board.Bitboards", true, isolated);
      long elapsedNanos = System.nanoTime() - t0;
      long elapsedMillis = elapsedNanos / 1_000_000L;
      // 3) Sanity post-init : warmup() est appelable et trivial.
      klass.getDeclaredMethod("warmup").invoke(null);
      assertThat(elapsedMillis)
          .as("Init Bitboards (<clinit>) en classloader isolé : %d ms", elapsedMillis)
          .isLessThan(WARMUP_BOOT_BUDGET_MS);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Constructeur privé
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Bitboards.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
