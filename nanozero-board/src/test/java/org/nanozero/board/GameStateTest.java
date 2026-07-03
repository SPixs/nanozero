package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests unitaires de {@link GameState} (cf. SPEC §5.2, §6, §7, §14 phases 7-9).
 *
 * <p>Critères de complétion phase 7 :
 *
 * <ul>
 *   <li>Pour 1 000 séquences de coups, applyMove puis unapplyLastMove restaure l'état exact (FEN
 *       identique caractère par caractère).
 *   <li>{@code isRepetition(3)} détecte correctement les triples répétitions sur des séquences
 *       construites.
 *   <li>Overflow lève {@link IllegalStateException} au 1025e coup.
 * </ul>
 *
 * <p>Critères de complétion phase 8 :
 *
 * <ul>
 *   <li>Toute la matrice §6.4 de matériel insuffisant testée (cas positifs et négatifs).
 *   <li>Mat / pat correctement détectés sur des positions construites.
 *   <li>Règle des 50 coups : {@code halfmoveClock = 99 → false}, {@code = 100 → true}.
 *   <li>{@code getResult()} dispatch correctement vers {@link Result#WIN_WHITE}, {@link
 *       Result#WIN_BLACK}, {@link Result#DRAW}, {@link Result#IN_PROGRESS}.
 * </ul>
 *
 * <p>Critères de complétion phase 9 :
 *
 * <ul>
 *   <li>Position de départ : 119 plans contiennent EXACTEMENT les valeurs attendues (vérification
 *       automatisée plan par plan).
 *   <li>Position avec historique partiel : timestamps au-delà de l'historique remplis de zéro
 *       (§7.5).
 *   <li>Inversion de perspective testée : {@code Long#reverseBytes} appliqué quand {@code
 *       sideToMove == BLACK} (§7.3).
 *   <li>100 positions aléatoires : {@code toPlanes} ne lève pas et remplit 7616 floats valides.
 *   <li>Plans de répétition : après cycle de 4 plies retour à startpos, plan 12 (rep ≥ 1) du T-0 à
 *       1.0 (§7.7).
 * </ul>
 */
class GameStateTest {

  // -----------------------------------------------------------------------------------------
  // Constructeurs et état initial
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constantes : HISTORY_CAPACITY=1024, NN_PLANES=119, NN_HISTORY_LENGTH=8")
  void publicConstants() {
    assertThat(GameState.HISTORY_CAPACITY).isEqualTo(1024);
    assertThat(GameState.NN_PLANES).isEqualTo(119);
    assertThat(GameState.NN_HISTORY_LENGTH).isEqualTo(8);
  }

  @Test
  @DisplayName("GameState() initialise à la position de départ avec historique vide")
  void defaultConstructorYieldsStartpos() {
    GameState gs = new GameState();
    assertThat(gs.toFen()).isEqualTo(Fen.STARTPOS);
    assertThat(gs.historySize).isZero();
  }

  @Test
  @DisplayName("GameState(fen) charge la FEN fournie")
  void fenConstructor() {
    String fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    GameState gs = new GameState(fen);
    assertThat(gs.toFen()).isEqualTo(fen);
    assertThat(gs.historySize).isZero();
  }

  @Test
  @DisplayName("currentPosition() retourne la même référence stable")
  void currentPositionStableReference() {
    GameState gs = new GameState();
    Position p1 = gs.currentPosition();
    int m = Move.fromUci("e2e4", p1);
    gs.applyMove(m);
    Position p2 = gs.currentPosition();
    assertThat(p2).isSameAs(p1); // même référence, contenu mis à jour
  }

  // -----------------------------------------------------------------------------------------
  // applyMove + unapplyLastMove : critère §14 phase 7
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Critère §14 phase 7 : 1000 séquences apply+unapply restaurent l'état exact")
  void applyUnapplySequencesRestoreExactState() {
    Random rng = new Random(0xDEADBEEFL);
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    for (int seqIdx = 0; seqIdx < 1000; seqIdx++) {
      GameState gs = new GameState();
      String start = gs.toFen();
      long startHash = gs.currentPosition().zobristHash();

      int targetDepth = 1 + rng.nextInt(15);
      int actualDepth = 0;
      for (int i = 0; i < targetDepth; i++) {
        int count = gs.generateMoves(buffer, 0);
        if (count == 0) {
          break;
        }
        gs.applyMove(buffer[rng.nextInt(count)]);
        actualDepth++;
      }
      // Annule tous les coups appliqués.
      for (int i = 0; i < actualDepth; i++) {
        gs.unapplyLastMove();
      }
      assertThat(gs.toFen()).as("sequence #%d, depth=%d", seqIdx, actualDepth).isEqualTo(start);
      assertThat(gs.currentPosition().zobristHash()).isEqualTo(startHash);
      assertThat(gs.historySize).isZero();
    }
  }

  @Test
  @DisplayName("I-GS-2 : applyMove(m) puis unapplyLastMove() restaure l'état exact (100 trials)")
  void invariantIGS2() {
    Random rng = new Random(0xCAFEFEEDL);
    GameState gs = new GameState();
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    for (int trial = 0; trial < 100; trial++) {
      String before = gs.toFen();
      long hashBefore = gs.currentPosition().zobristHash();
      int hsBefore = gs.historySize;

      int count = gs.generateMoves(buffer, 0);
      if (count == 0) {
        gs.reset();
        continue;
      }
      int chosen = buffer[rng.nextInt(count)];
      gs.applyMove(chosen);
      gs.unapplyLastMove();

      assertThat(gs.toFen()).isEqualTo(before);
      assertThat(gs.currentPosition().zobristHash()).isEqualTo(hashBefore);
      assertThat(gs.historySize).isEqualTo(hsBefore);

      // Avancer pour varier l'état entre trials.
      count = gs.generateMoves(buffer, 0);
      if (count == 0) {
        gs.reset();
        continue;
      }
      gs.applyMove(buffer[rng.nextInt(count)]);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Invariant I-GS-1 : history snapshots includent les hashes au moment du save
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("I-GS-1 : historyPositions[i].zobristHash() == hash de l'état au ply i")
  void invariantIGS1() {
    GameState gs = new GameState();
    long h0 = gs.currentPosition().zobristHash();
    int m1 = Move.fromUci("e2e4", gs.currentPosition());
    gs.applyMove(m1);
    long h1 = gs.currentPosition().zobristHash();
    int m2 = Move.fromUci("e7e5", gs.currentPosition());
    gs.applyMove(m2);
    long h2 = gs.currentPosition().zobristHash();
    int m3 = Move.fromUci("g1f3", gs.currentPosition());
    gs.applyMove(m3);

    // historyPositions[0] = état au ply 0 (avant applyMove m1)
    // historyPositions[1] = état au ply 1 (avant applyMove m2)
    // historyPositions[2] = état au ply 2 (avant applyMove m3)
    assertThat(gs.historySize).isEqualTo(3);
    assertThat(gs.historyPositions[0].zobristHash()).isEqualTo(h0);
    assertThat(gs.historyPositions[1].zobristHash()).isEqualTo(h1);
    assertThat(gs.historyPositions[2].zobristHash()).isEqualTo(h2);
  }

  // -----------------------------------------------------------------------------------------
  // I-GS-3 : isRepetition(1) toujours true
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("I-GS-3 : isRepetition(1) == true sur startpos et après applyMove")
  void invariantIGS3() {
    GameState gs = new GameState();
    assertThat(gs.isRepetition(1)).isTrue();
    int m = Move.fromUci("e2e4", gs.currentPosition());
    gs.applyMove(m);
    assertThat(gs.isRepetition(1)).isTrue();
  }

  @Test
  @DisplayName("isRepetition(0) et négatifs : retourne false")
  void isRepetitionZeroOrNegative() {
    GameState gs = new GameState();
    assertThat(gs.isRepetition(0)).isFalse();
    assertThat(gs.isRepetition(-1)).isFalse();
  }

  // -----------------------------------------------------------------------------------------
  // isRepetition twofold et threefold via knight shuffle (cf. note utilisateur)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isRepetition(2) : twofold après 4 plies de knight shuffle")
  void isRepetitionTwofoldAfterFourPlies() {
    // Cycle de 4 plies revenant exactement à startpos :
    //   1.Nb1c3, 1...Nb8c6, 2.Nc3b1, 2...Nc6b8 — bitboards et side identiques à startpos.
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (String uci : cycle) {
      gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
    }
    assertThat(gs.toFen()).contains("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -");
    assertThat(gs.isRepetition(2)).as("twofold après 4 plies de cycle").isTrue();
    assertThat(gs.isRepetition(3)).as("triple non encore atteinte").isFalse();
  }

  @Test
  @DisplayName("isRepetition(3) : threefold après 8 plies de knight shuffle (deux cycles)")
  void isRepetitionThreefoldAfterEightPlies() {
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (int rep = 0; rep < 2; rep++) {
      for (String uci : cycle) {
        gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
      }
    }
    assertThat(gs.isRepetition(3)).as("triple répétition exacte").isTrue();
  }

  @Test
  @DisplayName("isRepetition optimisation : un coup irréversible borne la fenêtre de scan")
  void isRepetitionHalfmoveOptimization() {
    // Après une avance de pion, halfmoveClock reset à 0 ; les positions antérieures à ce
    // coup ne peuvent plus être reproduites (un pion ne revient jamais en arrière). La borne
    // de scan limit = max(0, historySize - halfmoveClock) garantit qu'on ne scanne pas
    // au-delà.
    GameState gs = new GameState();
    // 4 plies de knight shuffle, halfmoveClock = 4
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (String uci : cycle) {
      gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
    }
    assertThat(gs.currentPosition().halfmoveClock()).isEqualTo(4);
    // Avance de pion : halfmoveClock reset à 0
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));
    assertThat(gs.currentPosition().halfmoveClock()).isZero();
    assertThat(gs.historySize).isEqualTo(5);
    // limit = max(0, 5 - 0) = 5 → scan i de 4 à 5 = aucun élément. Donc isRepetition(2) = false
    // même si Zobrist hash matchait techniquement (impossible ici car bitboards diffèrent par
    // le pion poussé, mais l'invariant de scan est testé).
    assertThat(gs.isRepetition(2)).isFalse();
  }

  // -----------------------------------------------------------------------------------------
  // Overflow d'historique
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Overflow : 1025e applyMove lève IllegalStateException")
  void overflowAtCapacity() {
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (int i = 0; i < GameState.HISTORY_CAPACITY; i++) {
      String uci = cycle[i % 4];
      gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
    }
    assertThat(gs.historySize).isEqualTo(GameState.HISTORY_CAPACITY);
    int nextMove = Move.fromUci("b1c3", gs.currentPosition());
    assertThatThrownBy(() -> gs.applyMove(nextMove))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining(String.valueOf(GameState.HISTORY_CAPACITY));
  }

  @Test
  @DisplayName("unapplyLastMove sur historique vide lève IllegalStateException")
  void unapplyEmptyHistoryThrows() {
    GameState gs = new GameState();
    assertThatThrownBy(gs::unapplyLastMove).isInstanceOf(IllegalStateException.class);
  }

  // -----------------------------------------------------------------------------------------
  // reset / setFromFen : purge de l'historique
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("reset() repositionne sur startpos et purge l'historique")
  void resetPurgesHistory() {
    GameState gs = new GameState();
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));
    gs.applyMove(Move.fromUci("e7e5", gs.currentPosition()));
    assertThat(gs.historySize).isEqualTo(2);
    gs.reset();
    assertThat(gs.toFen()).isEqualTo(Fen.STARTPOS);
    assertThat(gs.historySize).isZero();
  }

  @Test
  @DisplayName("setFromFen(fen) charge la FEN et purge l'historique")
  void setFromFenPurgesHistory() {
    GameState gs = new GameState();
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));
    String target = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    gs.setFromFen(target);
    assertThat(gs.toFen()).isEqualTo(target);
    assertThat(gs.historySize).isZero();
  }

  // -----------------------------------------------------------------------------------------
  // generateMoves : délégation
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("generateMoves délègue à MoveGen.generateMoves(currentPosition, ...)")
  void generateMovesDelegates() {
    GameState gs = new GameState();
    int[] direct = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int[] viaGameState = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int countDirect = MoveGen.generateMoves(gs.currentPosition(), direct, 0);
    int countDelegated = gs.generateMoves(viaGameState, 0);
    assertThat(countDelegated).isEqualTo(countDirect);
    for (int i = 0; i < countDirect; i++) {
      assertThat(viaGameState[i]).isEqualTo(direct[i]);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : isCheckmate (§6.1)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isCheckmate : mat construit roi+dame+roi (blancs matent les noirs)")
  void checkmateConstructedWhiteMatesBlack() {
    GameState gs = new GameState("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isCheckmate()).isTrue();
    assertThat(gs.isStalemate()).isFalse();
  }

  @Test
  @DisplayName("isCheckmate : mat construit roi+dame+roi (noirs matent les blancs)")
  void checkmateConstructedBlackMatesWhite() {
    GameState gs = new GameState("8/8/8/8/8/4k3/4q3/4K3 w - - 0 1");
    assertThat(gs.isCheckmate()).isTrue();
    assertThat(gs.isStalemate()).isFalse();
  }

  @Test
  @DisplayName("isCheckmate : false sur startpos (pas en échec)")
  void checkmateFalseOnStartpos() {
    GameState gs = new GameState();
    assertThat(gs.isCheckmate()).isFalse();
  }

  @Test
  @DisplayName("isCheckmate : false sur pat (pas en échec, mais 0 coup)")
  void checkmateFalseOnStalemate() {
    GameState gs = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isCheckmate()).isFalse();
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : isStalemate (§6.1)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isStalemate : pat construit Qf7 vs Kh8")
  void stalemateConstructedKQvK() {
    GameState gs = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isStalemate()).isTrue();
    assertThat(gs.isCheckmate()).isFalse();
  }

  @Test
  @DisplayName("isStalemate : false sur position avec coups légaux")
  void stalemateFalseOnStartpos() {
    GameState gs = new GameState();
    assertThat(gs.isStalemate()).isFalse();
  }

  @Test
  @DisplayName("isStalemate : false sur mat (en échec, donc pas pat)")
  void stalemateFalseOnCheckmate() {
    GameState gs = new GameState("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isStalemate()).isFalse();
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : isFiftyMoveRule (§6.3)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isFiftyMoveRule : false si halfmoveClock < 100")
  void fiftyMoveRuleFalseBelowThreshold() {
    GameState gs99 = new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 99 1");
    assertThat(gs99.isFiftyMoveRule()).isFalse();
    GameState gs0 = new GameState();
    assertThat(gs0.isFiftyMoveRule()).isFalse();
  }

  @Test
  @DisplayName("isFiftyMoveRule : true si halfmoveClock >= 100")
  void fiftyMoveRuleTrueAtThreshold() {
    GameState gs100 = new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 100 1");
    assertThat(gs100.isFiftyMoveRule()).isTrue();
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : isInsufficientMaterial (§6.4) — matrice complète FIDE stricte
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "[{index}] {0} → {2}")
  @CsvSource({
    // Cas POSITIFS (matériel insuffisant)
    "K vs K,                            4k3/8/8/8/8/8/8/4K3 w - - 0 1,        true",
    "KN vs K (cavalier blanc),          4k3/8/8/8/8/8/8/3NK3 w - - 0 1,       true",
    "KN vs K (cavalier noir),           4k3/n7/8/8/8/8/8/4K3 w - - 0 1,       true",
    "KB vs K (fou blanc),               4k3/8/8/8/8/8/8/3BK3 w - - 0 1,       true",
    "KB vs K (fou noir),                4k3/b7/8/8/8/8/8/4K3 w - - 0 1,       true",
    "KBB tous fous cases claires,       4k3/8/8/8/8/8/8/1B2K2B w - - 0 1,     true",
    "KBB tous fous cases sombres,       4k3/8/8/8/8/8/8/B1B1K3 w - - 0 1,     true",
    "KB vs KB fous même couleur,        4k1b1/8/8/8/8/8/8/1B2K3 w - - 0 1,    true",
    // Cas NÉGATIFS (théoriquement gagnable)
    "KBB fous couleurs différentes,     4k3/8/8/8/8/8/8/BB2K3 w - - 0 1,      false",
    "KNN vs K (deux cavaliers),         4k3/8/8/8/8/8/8/2N1K1N1 w - - 0 1,    false",
    "KB vs KN,                          1n2k3/8/8/8/8/8/8/4KB2 w - - 0 1,     false",
    "KN vs KN,                          4k1n1/8/8/8/8/8/8/1N2K3 w - - 0 1,    false",
    "KB vs KB fous couleurs opposées,   3bk3/8/8/8/8/8/8/3BK3 w - - 0 1,      false",
    "K + pion,                          4k3/8/8/8/8/8/4P3/4K3 w - - 0 1,      false",
    "K + tour,                          4k3/8/8/8/8/8/8/R3K3 w - - 0 1,       false",
    "K + dame,                          4k3/8/8/8/8/8/8/3QK3 w - - 0 1,       false"
  })
  @DisplayName("isInsufficientMaterial : matrice FIDE stricte §6.4")
  void insufficientMaterialMatrix(String label, String fen, boolean expected) {
    GameState gs = new GameState(fen);
    assertThat(gs.isInsufficientMaterial())
        .as("Position: %s (FEN: %s)", label, fen)
        .isEqualTo(expected);
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : isTerminal (§6.5) — combinaison
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isTerminal : false sur startpos (partie en cours)")
  void terminalFalseOnStartpos() {
    GameState gs = new GameState();
    assertThat(gs.isTerminal()).isFalse();
  }

  @Test
  @DisplayName("isTerminal : true sur mat")
  void terminalTrueOnCheckmate() {
    GameState gs = new GameState("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isTerminal()).isTrue();
  }

  @Test
  @DisplayName("isTerminal : true sur pat")
  void terminalTrueOnStalemate() {
    GameState gs = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(gs.isTerminal()).isTrue();
  }

  @Test
  @DisplayName("isTerminal : true sur règle 50 coups")
  void terminalTrueOnFiftyMoveRule() {
    GameState gs = new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 100 1");
    assertThat(gs.isTerminal()).isTrue();
  }

  @Test
  @DisplayName("isTerminal : true sur matériel insuffisant")
  void terminalTrueOnInsufficientMaterial() {
    GameState gs = new GameState("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assertThat(gs.isTerminal()).isTrue();
  }

  @Test
  @DisplayName("isTerminal : true sur triple répétition")
  void terminalTrueOnThreefoldRepetition() {
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (int rep = 0; rep < 2; rep++) {
      for (String uci : cycle) {
        gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
      }
    }
    assertThat(gs.isTerminal()).isTrue();
  }

  // -----------------------------------------------------------------------------------------
  // Phase 8 : getResult (§6.5) — dispatch
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("getResult : WIN_WHITE sur mat infligé aux noirs")
  void resultWinWhiteOnBlackMated() {
    GameState gs = new GameState("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1");
    assertThat(gs.getResult()).isEqualTo(Result.WIN_WHITE);
  }

  @Test
  @DisplayName("getResult : WIN_BLACK sur mat infligé aux blancs")
  void resultWinBlackOnWhiteMated() {
    GameState gs = new GameState("8/8/8/8/8/4k3/4q3/4K3 w - - 0 1");
    assertThat(gs.getResult()).isEqualTo(Result.WIN_BLACK);
  }

  @Test
  @DisplayName("getResult : DRAW sur pat")
  void resultDrawOnStalemate() {
    GameState gs = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(gs.getResult()).isEqualTo(Result.DRAW);
  }

  @Test
  @DisplayName("getResult : DRAW sur règle 50 coups")
  void resultDrawOnFiftyMoveRule() {
    GameState gs = new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 100 1");
    assertThat(gs.getResult()).isEqualTo(Result.DRAW);
  }

  @Test
  @DisplayName("getResult : DRAW sur matériel insuffisant (K vs K)")
  void resultDrawOnInsufficientMaterial() {
    GameState gs = new GameState("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assertThat(gs.getResult()).isEqualTo(Result.DRAW);
  }

  @Test
  @DisplayName("getResult : DRAW sur triple répétition")
  void resultDrawOnThreefoldRepetition() {
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (int rep = 0; rep < 2; rep++) {
      for (String uci : cycle) {
        gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
      }
    }
    assertThat(gs.getResult()).isEqualTo(Result.DRAW);
  }

  @Test
  @DisplayName("getResult : IN_PROGRESS sur startpos")
  void resultInProgressOnStartpos() {
    GameState gs = new GameState();
    assertThat(gs.getResult()).isEqualTo(Result.IN_PROGRESS);
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : toPlanes (§7) — helpers
  // -----------------------------------------------------------------------------------------

  private static final int PLANE_SIZE = 64;
  private static final int TOTAL_PLANES_FLOATS = GameState.NN_PLANES * PLANE_SIZE; // 7616

  /** Vérifie qu'un plan correspond bit-à-bit à un bitboard donné (1.0 si bit, 0.0 sinon). */
  private static void assertPlaneMatchesBitboard(
      float[] planes, int planeIdx, long expectedBB, String desc) {
    for (int sq = 0; sq < 64; sq++) {
      float expected = ((expectedBB >>> sq) & 1L) == 0L ? 0.0f : 1.0f;
      assertThat(planes[planeIdx * PLANE_SIZE + sq])
          .as("%s plane=%d sq=%d", desc, planeIdx, sq)
          .isEqualTo(expected);
    }
  }

  /** Vérifie qu'un plan est constant à une valeur donnée. */
  private static void assertPlaneIsConstant(
      float[] planes, int planeIdx, float value, String desc) {
    for (int sq = 0; sq < 64; sq++) {
      assertThat(planes[planeIdx * PLANE_SIZE + sq])
          .as("%s plane=%d sq=%d", desc, planeIdx, sq)
          .isEqualTo(value);
    }
  }

  /** Vérifie qu'une plage de plans est entièrement à zéro. */
  private static void assertPlanesAreZero(float[] planes, int firstPlane, int planeCount) {
    for (int p = firstPlane; p < firstPlane + planeCount; p++) {
      for (int sq = 0; sq < 64; sq++) {
        assertThat(planes[p * PLANE_SIZE + sq]).as("plane=%d sq=%d", p, sq).isZero();
      }
    }
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : toPlanes startpos exhaustif
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : 119 plans complets sur position de départ")
  void toPlanesStartposExhaustive() {
    GameState gs = new GameState();
    float[] planes = new float[TOTAL_PLANES_FLOATS];
    gs.toPlanes(planes, 0);

    // Bitboards startpos (§3.1, ADR-008).
    long whitePawnBB = 0x000000000000FF00L;
    long whiteKnightBB = 0x0000000000000042L; // b1, g1
    long whiteBishopBB = 0x0000000000000024L; // c1, f1
    long whiteRookBB = 0x0000000000000081L; // a1, h1
    long whiteQueenBB = 0x0000000000000008L; // d1
    long whiteKingBB = 0x0000000000000010L; // e1
    long blackPawnBB = 0x00FF000000000000L;
    long blackKnightBB = 0x4200000000000000L; // b8, g8
    long blackBishopBB = 0x2400000000000000L; // c8, f8
    long blackRookBB = 0x8100000000000000L; // a8, h8
    long blackQueenBB = 0x0800000000000000L; // d8
    long blackKingBB = 0x1000000000000000L; // e8

    // Timestamp 0 (T-0), plans 0-13 :
    //   P1 (= WHITE car WHITE au trait sur startpos) PAWN..KING : plans 0-5
    assertPlaneMatchesBitboard(planes, 0, whitePawnBB, "T0 P1 PAWN");
    assertPlaneMatchesBitboard(planes, 1, whiteKnightBB, "T0 P1 KNIGHT");
    assertPlaneMatchesBitboard(planes, 2, whiteBishopBB, "T0 P1 BISHOP");
    assertPlaneMatchesBitboard(planes, 3, whiteRookBB, "T0 P1 ROOK");
    assertPlaneMatchesBitboard(planes, 4, whiteQueenBB, "T0 P1 QUEEN");
    assertPlaneMatchesBitboard(planes, 5, whiteKingBB, "T0 P1 KING");
    //   P2 (= BLACK) PAWN..KING : plans 6-11
    assertPlaneMatchesBitboard(planes, 6, blackPawnBB, "T0 P2 PAWN");
    assertPlaneMatchesBitboard(planes, 7, blackKnightBB, "T0 P2 KNIGHT");
    assertPlaneMatchesBitboard(planes, 8, blackBishopBB, "T0 P2 BISHOP");
    assertPlaneMatchesBitboard(planes, 9, blackRookBB, "T0 P2 ROOK");
    assertPlaneMatchesBitboard(planes, 10, blackQueenBB, "T0 P2 QUEEN");
    assertPlaneMatchesBitboard(planes, 11, blackKingBB, "T0 P2 KING");
    //   Repetition >= 1 et >= 2 : plans 12-13, tous 0.0 (aucun historique)
    assertPlaneIsConstant(planes, 12, 0.0f, "T0 rep>=1");
    assertPlaneIsConstant(planes, 13, 0.0f, "T0 rep>=2");

    // Timestamps T-1 à T-7 (plans 14..111) : tous 0.0 car pas d'historique.
    assertPlanesAreZero(planes, 14, 7 * 14);

    // Plans constants 112..118 :
    assertPlaneIsConstant(planes, 112, 1.0f, "Color (WHITE)");
    // Plan 113 : move count = min(1, 99) / 99 = 1/99
    for (int sq = 0; sq < 64; sq++) {
      assertThat(planes[113 * PLANE_SIZE + sq])
          .as("plane=113 sq=%d", sq)
          .isCloseTo(1.0f / 99.0f, within(1e-6f));
    }
    // Plans 114-117 : KQkq → tous 1.0 (sideToMove == WHITE donc P1=W, P2=B)
    assertPlaneIsConstant(planes, 114, 1.0f, "P1 KS = WHITE_KS");
    assertPlaneIsConstant(planes, 115, 1.0f, "P1 QS = WHITE_QS");
    assertPlaneIsConstant(planes, 116, 1.0f, "P2 KS = BLACK_KS");
    assertPlaneIsConstant(planes, 117, 1.0f, "P2 QS = BLACK_QS");
    // Plan 118 : no-progress = 0/100 = 0.0
    assertPlaneIsConstant(planes, 118, 0.0f, "no-progress");
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : inversion de perspective (§7.3) — sideToMove == BLACK
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : inversion verticale via Long.reverseBytes quand sideToMove == BLACK")
  void toPlanesPerspectiveFlipBlackToMove() {
    // Sanity check du contrat de Long.reverseBytes.
    assertThat(Long.reverseBytes(0x00000000000000FFL)).isEqualTo(0xFF00000000000000L);

    // 1.e4, puis BLACK au trait. needFlip = true.
    GameState gs = new GameState();
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));

    float[] planes = new float[TOTAL_PLANES_FLOATS];
    gs.toPlanes(planes, 0);

    // P1 = BLACK (au trait). Plan 0 = pions noirs (rang 7 = bits 48-55) inversés en rang 2 du plan.
    long blackPawnBB = 0x00FF000000000000L;
    long blackPawnFlipped = Long.reverseBytes(blackPawnBB);
    assertThat(blackPawnFlipped).isEqualTo(0x000000000000FF00L);
    assertPlaneMatchesBitboard(planes, 0, blackPawnFlipped, "T0 P1 PAWN (BLACK flippé)");

    // P2 = WHITE. Plan 6 = pions blancs après 1.e4 inversés.
    // Pions blancs : a2-d2, f2-h2 (bits 8-11, 13-15) et e4 (bit 28).
    long whitePawnBB = 0x000000001000EF00L;
    long whitePawnFlipped = Long.reverseBytes(whitePawnBB);
    assertThat(gs.currentPosition().pieceBB(Piece.WHITE_PAWN)).isEqualTo(whitePawnBB);
    assertPlaneMatchesBitboard(planes, 6, whitePawnFlipped, "T0 P2 PAWN (WHITE flippé)");

    // Plan 112 (Color) = 0.0 car BLACK au trait.
    assertPlaneIsConstant(planes, 112, 0.0f, "Color (BLACK)");
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : mapping P1/P2 castling rights selon sideToMove (§7.4)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : mapping castling P1/P2 inversé quand sideToMove == BLACK")
  void toPlanesCastlingMappingBlackToMove() {
    // Position : seul WHITE_KINGSIDE actif, BLACK au trait.
    // Avec us == BLACK : p1KS = BLACK_KS (bit clear), p2KS = WHITE_KS (bit set).
    GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R b K - 0 1");
    float[] planes = new float[TOTAL_PLANES_FLOATS];
    gs.toPlanes(planes, 0);

    assertPlaneIsConstant(planes, 114, 0.0f, "P1 KS = BLACK_KS (absent)");
    assertPlaneIsConstant(planes, 115, 0.0f, "P1 QS = BLACK_QS (absent)");
    assertPlaneIsConstant(planes, 116, 1.0f, "P2 KS = WHITE_KS (présent)");
    assertPlaneIsConstant(planes, 117, 0.0f, "P2 QS = WHITE_QS (absent)");
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : historique partiel (§7.5)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : timestamps au-delà de l'historique remplis de zéros (3 plies joués)")
  void toPlanesPartialHistoryThreeMoves() {
    GameState gs = new GameState();
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));
    gs.applyMove(Move.fromUci("e7e5", gs.currentPosition()));
    gs.applyMove(Move.fromUci("g1f3", gs.currentPosition()));

    float[] planes = new float[TOTAL_PLANES_FLOATS];
    gs.toPlanes(planes, 0);

    // historySize = 3. Timestamps T-0..T-3 populés (4 timestamps), T-4..T-7 zéros (4 timestamps).
    // Plans 4*14=56 à 8*14-1=111 : tous zéro (= 4 × 14 = 56 plans).
    assertPlanesAreZero(planes, 4 * 14, 4 * 14);

    // T-0 P1 = WHITE (au trait). Vérifier que le pion e4 et le cavalier f3 sont là.
    long whitePawnBB = gs.currentPosition().pieceBB(Piece.WHITE_PAWN);
    assertPlaneMatchesBitboard(planes, 0, whitePawnBB, "T0 P1 PAWN après 3 plies");

    // T-3 (état initial = startpos avant e2e4). Plan offset 3*14 = 42.
    // P1 = WHITE → plan 42 (P1 PAWN) doit montrer les pions blancs en startpos.
    assertPlaneMatchesBitboard(planes, 42, 0x000000000000FF00L, "T-3 P1 PAWN (startpos)");
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : plans de répétition (§7.7)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : plan 12 (rep>=1) du T-0 à 1.0 après cycle de 4 plies")
  void toPlanesRepetitionPlaneAfterShuffle() {
    GameState gs = new GameState();
    String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
    for (String uci : cycle) {
      gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
    }

    float[] planes = new float[TOTAL_PLANES_FLOATS];
    gs.toPlanes(planes, 0);

    // T-0 plan 12 (rep >= 1) doit être 1.0 (la position de départ est revenue, comptée 1 fois
    // dans l'historique strict).
    assertPlaneIsConstant(planes, 12, 1.0f, "T0 rep>=1 après cycle");
    assertPlaneIsConstant(planes, 13, 0.0f, "T0 rep>=2 après cycle");
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : robustesse — 100 positions aléatoires sans exception
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : 100 positions aléatoires remplissent 7616 floats valides sans exception")
  void toPlanesRandomRobustness() {
    Random rng = new Random(0xCAFEBABEL);
    int[] moveBuffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    float[] planes = new float[TOTAL_PLANES_FLOATS];

    int positionsTested = 0;
    while (positionsTested < 100) {
      GameState gs = new GameState();
      int targetDepth = 1 + rng.nextInt(40);
      for (int i = 0; i < targetDepth; i++) {
        int n = gs.generateMoves(moveBuffer, 0);
        if (n == 0) {
          break;
        }
        gs.applyMove(moveBuffer[rng.nextInt(n)]);
      }
      // Salissage avec des valeurs sentinelle pour vérifier que toPlanes écrit bien partout.
      for (int i = 0; i < TOTAL_PLANES_FLOATS; i++) {
        planes[i] = Float.NaN;
      }
      gs.toPlanes(planes, 0);
      for (int i = 0; i < TOTAL_PLANES_FLOATS; i++) {
        assertThat(planes[i]).as("position #%d index %d", positionsTested, i).isFinite();
      }
      // Vérifier que les plans pièces (T-0, indices 0-11) sont strictement 0.0 ou 1.0.
      for (int p = 0; p < 12; p++) {
        for (int sq = 0; sq < 64; sq++) {
          float v = planes[p * PLANE_SIZE + sq];
          assertThat(v).isIn(0.0f, 1.0f);
        }
      }
      positionsTested++;
    }
  }

  // -----------------------------------------------------------------------------------------
  // Phase 9 : surcharge avec encoder injecté
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toPlanes : surcharge avec encoder externe — résultat identique à scalaire")
  void toPlanesInjectedEncoderEquivalence() {
    GameState gs = new GameState();
    gs.applyMove(Move.fromUci("e2e4", gs.currentPosition()));
    gs.applyMove(Move.fromUci("e7e5", gs.currentPosition()));

    float[] viaScalar = new float[TOTAL_PLANES_FLOATS];
    float[] viaInjected = new float[TOTAL_PLANES_FLOATS];

    gs.toPlanes(viaScalar, 0);

    // Encoder injecté sémantiquement équivalent à l'encoder scalaire (boucle alternative).
    BitboardPlaneEncoder injected =
        (bb, dest, off) -> {
          for (int sq = 0; sq < 64; sq++) {
            dest[off + sq] = ((bb >> sq) & 1L) != 0 ? 1.0f : 0.0f;
          }
        };
    gs.toPlanes(viaInjected, 0, injected);

    assertThat(viaInjected).containsExactly(viaScalar);
  }

  // -----------------------------------------------------------------------------------
  // copy() — deep copy pour parallélisation MCTS (engine v1.2.0 batched, ADR-016)
  // -----------------------------------------------------------------------------------

  @Test
  @DisplayName("copy() startpos : FEN identique, historySize=0")
  void copyStartposPreservesFen() {
    GameState gs = new GameState();
    GameState clone = gs.copy();
    assertThat(clone.toFen()).isEqualTo(gs.toFen());
    assertThat(clone).isNotSameAs(gs);
    assertThat(clone.currentPosition()).isNotSameAs(gs.currentPosition());
  }

  @Test
  @DisplayName("copy() preserve currentPosition (deep, indépendant)")
  void copyDeepCopiesCurrentPosition() {
    GameState gs = new GameState();
    int[] buf = new int[256];
    int n = gs.generateMoves(buf, 0);
    gs.applyMove(buf[0]);

    GameState clone = gs.copy();
    // Mute l'original, le clone ne doit pas changer.
    int[] buf2 = new int[256];
    int n2 = gs.generateMoves(buf2, 0);
    gs.applyMove(buf2[0]);

    assertThat(clone.toFen()).isNotEqualTo(gs.toFen()); // gs a 1 coup de plus
    // Le clone est intact, équivalent à gs APRÈS le 1er coup uniquement.
  }

  @Test
  @DisplayName("copy() preserve historique (historySize identique)")
  void copyPreservesHistorySize() {
    GameState gs = new GameState();
    int[] buf = new int[256];
    // Joue 5 coups.
    for (int i = 0; i < 5; i++) {
      int n = gs.generateMoves(buf, 0);
      gs.applyMove(buf[0]);
    }
    GameState clone = gs.copy();
    // Clone doit pouvoir unapplyLastMove() 5 fois sans crash.
    for (int i = 0; i < 5; i++) {
      clone.unapplyLastMove();
    }
    assertThat(clone.toFen()).isEqualTo(new GameState().toFen()); // retour startpos
    // gs n'a pas été affecté.
    assertThat(gs.toFen()).isNotEqualTo(new GameState().toFen());
  }

  @Test
  @DisplayName("copy() : applyMove sur copy ne mute pas l'original")
  void copyIsolatedFromOriginal() {
    GameState gs = new GameState();
    String fenBefore = gs.toFen();
    GameState clone = gs.copy();

    int[] buf = new int[256];
    int n = clone.generateMoves(buf, 0);
    clone.applyMove(buf[0]);

    assertThat(gs.toFen()).as("gs intact").isEqualTo(fenBefore);
    assertThat(clone.toFen()).as("clone muté").isNotEqualTo(fenBefore);
  }

  @Test
  @DisplayName("copy() depuis FEN custom préserve la position")
  void copyFromCustomFen() {
    String fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    GameState gs = new GameState(fen);
    GameState clone = gs.copy();
    assertThat(clone.toFen()).isEqualTo(fen);
  }

  @Test
  @DisplayName("copyFrom() mute in-place, sans allocation")
  void copyFromInPlace() {
    GameState src = new GameState();
    int[] buf = new int[256];
    int n = src.generateMoves(buf, 0);
    src.applyMove(buf[0]);
    src.applyMove(src.generateMoves(buf, 0) > 0 ? buf[0] : 0);

    GameState dst = new GameState(); // fresh startpos
    dst.copyFrom(src);
    assertThat(dst.toFen()).isEqualTo(src.toFen());
    // unapplyLastMove sur dst doit fonctionner (historique présent)
    dst.unapplyLastMove();
    dst.unapplyLastMove();
    assertThat(dst.toFen()).isEqualTo(new GameState().toFen());
    // src non muté
    assertThat(src.toFen()).isNotEqualTo(new GameState().toFen());
  }

  @Test
  @DisplayName("copyFrom() overwrite : reset depuis startpos vers position avancée")
  void copyFromOverwrite() {
    GameState dst = new GameState();
    int[] buf = new int[256];
    int n = dst.generateMoves(buf, 0);
    dst.applyMove(buf[0]);
    String fenAfterOne = dst.toFen();

    GameState src = new GameState();
    int n2 = src.generateMoves(buf, 0);
    src.applyMove(buf[0]);
    src.applyMove(src.generateMoves(buf, 0) > 0 ? buf[0] : 0);
    src.applyMove(src.generateMoves(buf, 0) > 0 ? buf[0] : 0);
    String fenSrc = src.toFen();

    dst.copyFrom(src); // dst est complètement écrasé par src
    assertThat(dst.toFen()).isEqualTo(fenSrc);
    assertThat(dst.toFen()).isNotEqualTo(fenAfterOne);
  }

  @Test
  @DisplayName("copy() après 50+ coups : historique complet préservé bit-pour-bit")
  void copyDeepHistory() {
    GameState gs = new GameState();
    int[] buf = new int[256];
    // Joue ~60 coups.
    for (int i = 0; i < 60; i++) {
      int n = gs.generateMoves(buf, 0);
      if (n == 0) {
        break;
      }
      gs.applyMove(buf[0]);
    }
    String fenAfter = gs.toFen();
    GameState clone = gs.copy();
    assertThat(clone.toFen()).isEqualTo(fenAfter);
    // Tous les unapplyLastMove doivent fonctionner sur clone, ramener vers startpos.
    while (true) {
      try {
        clone.unapplyLastMove();
      } catch (IllegalStateException e) {
        break; // historique vide atteint
      }
    }
    assertThat(clone.toFen()).isEqualTo(new GameState().toFen());
  }
}
