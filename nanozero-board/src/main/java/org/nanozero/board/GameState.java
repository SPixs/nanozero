package org.nanozero.board;

import java.util.Arrays;

/**
 * État d'une partie d'échecs : composite de {@link Position} + historiques (cf. SPEC §4.2.2, §5.2,
 * ADR-005).
 *
 * <p>Là où {@link Position} ne représente qu'un instantané (état immédiat de l'échiquier), {@link
 * GameState} ajoute :
 *
 * <ul>
 *   <li>Un stack pré-alloué de positions précédentes pour permettre {@link #unapplyLastMove()} et
 *       la détection de répétition.
 *   <li>L'API {@link #isRepetition(int)} avec optimisation par {@code halfmoveClock} (§5.2.3).
 *   <li>Les méthodes terminales {@link #isCheckmate()}, {@link #isStalemate()}, {@link
 *       #isFiftyMoveRule()}, {@link #isInsufficientMaterial()}, {@link #isTerminal()}, {@link
 *       #getResult()} (implémentées en phase 8).
 *   <li>L'extraction des plans NN via {@link #toPlanes(float[], int)} (implémentée en phase 9).
 * </ul>
 *
 * <p>{@code GameState} est <strong>mutable</strong> et <strong>non thread-safe</strong>. Une
 * instance par thread (cf. SPEC §10.1).
 *
 * <p><strong>Phase 9 (état au commit courant)</strong> : l'extraction des 119 plans NN AlphaZero
 * via {@link #toPlanes(float[], int)} et {@link #toPlanes(float[], int, BitboardPlaneEncoder)} est
 * implémentée conformément à SPEC §7. L'encodeur scalaire de référence est utilisé par défaut ; un
 * foncteur vectorisé (Vector API) peut être injecté par {@code nanozero-nn}.
 *
 * <p>Invariants maintenus :
 *
 * <ul>
 *   <li>I-GS-1 : {@code currentPosition.zobristHash} cohérent avec les hash des snapshots
 *       d'historique aux positions correspondantes (garanti par {@code copyFrom} dans {@code
 *       applyMove}).
 *   <li>I-GS-2 : {@code unapplyLastMove} après {@code applyMove} restaure l'état exact (bitboards,
 *       hash, compteurs).
 *   <li>I-GS-3 : {@code isRepetition(1) == true} toujours (la position courante compte comme
 *       première occurrence).
 * </ul>
 */
public final class GameState {

  /**
   * Capacité du buffer d'historique. Largement supérieure à la durée de toute partie réaliste
   * (longueur record en compétition : ~270 demi-coups).
   */
  public static final int HISTORY_CAPACITY = 1024;

  /** Nombre de plans NN au format AlphaZero pour les échecs (cf. SPEC §7). */
  public static final int NN_PLANES = 119;

  /** Nombre de positions historiques utilisées dans les plans NN (cf. SPEC §7). */
  public static final int NN_HISTORY_LENGTH = 8;

  // ---------------------------------------------------------------------------------------------
  // État interne
  // ---------------------------------------------------------------------------------------------

  /** Position courante. Référence stable pendant toute la durée de vie du GameState. */
  final Position currentPosition = new Position();

  /**
   * Stack pré-alloué de snapshots des positions précédentes. {@code historyPositions[i]} pour
   * {@code i ∈ [0, historySize)} contient l'état de {@code currentPosition} avant l'application du
   * {@code (i+1)}-ième coup. Les entrées au-delà de {@code historySize} sont des Position
   * pré-allouées dont le contenu est obsolète et sera écrasé par le prochain {@code applyMove}.
   */
  final Position[] historyPositions = new Position[HISTORY_CAPACITY];

  /** Nombre d'entrées valides dans {@link #historyPositions}. */
  int historySize;

  /**
   * Buffer scratch pour la génération de coups dans {@link #isCheckmate()}, {@link #isStalemate()},
   * {@link #isTerminal()} et {@link #getResult()}. Pré-alloué pour garantir zéro allocation dans la
   * chaîne de détection terminale (cf. SPEC §5.2.1).
   */
  private final int[] scratchMoveBuffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];

  // Initializer pré-alloue les 1024 instances Position pour éliminer les allocations en hot path
  // d'applyMove (cf. SPEC §5.2.1).
  {
    for (int i = 0; i < HISTORY_CAPACITY; i++) {
      historyPositions[i] = new Position();
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Constructeurs
  // ---------------------------------------------------------------------------------------------

  /**
   * Construit un {@code GameState} à la position de départ standard.
   *
   * <p>Équivalent à {@code new GameState(Fen.STARTPOS)}.
   */
  public GameState() {
    this(Fen.STARTPOS);
  }

  /**
   * Construit un {@code GameState} à partir d'une FEN.
   *
   * @param fen FEN à parser dans la position courante
   * @throws IllegalArgumentException si la FEN est mal formée ou la position invalide
   */
  public GameState(String fen) {
    Fen.parse(fen, currentPosition);
    historySize = 0;
  }

  // ---------------------------------------------------------------------------------------------
  // API : accès et mutation
  // ---------------------------------------------------------------------------------------------

  /**
   * Retourne la {@link Position} courante. La référence est stable pendant toute la durée de vie du
   * GameState ; ses champs internes mutent à chaque {@code applyMove}.
   */
  public Position currentPosition() {
    return currentPosition;
  }

  /**
   * Applique un coup à la position courante après avoir snapshoté l'état pour permettre une future
   * {@link #unapplyLastMove()} et la détection de répétition. Conforme à SPEC §5.2.1.
   *
   * @param move coup encodé selon le format {@link Move} 16-bit
   * @throws IllegalStateException si l'historique a atteint {@link #HISTORY_CAPACITY}. Ce cas ne
   *     peut survenir que pour des parties pathologiques de plus de 1024 demi-coups, situation hors
   *     de tout cadre normal d'utilisation (record en compétition : ~270 demi-coups).
   */
  public void applyMove(int move) {
    if (historySize == HISTORY_CAPACITY) {
      throw new IllegalStateException(
          "Historique GameState saturé : " + HISTORY_CAPACITY + " demi-coups.");
    }
    historyPositions[historySize].copyFrom(currentPosition);
    historySize += 1;
    currentPosition.applyMove(move);
  }

  /**
   * Annule le dernier coup appliqué via {@link #applyMove(int)}, restaurant l'état précédent exact
   * (bitboards, hash, compteurs).
   *
   * @throws IllegalStateException si aucun coup n'a été appliqué (historique vide)
   */
  public void unapplyLastMove() {
    if (historySize == 0) {
      throw new IllegalStateException("Aucun coup à annuler : historique vide.");
    }
    historySize -= 1;
    currentPosition.copyFrom(historyPositions[historySize]);
  }

  // ---------------------------------------------------------------------------------------------
  // API : détection de répétition
  // ---------------------------------------------------------------------------------------------

  /**
   * Indique si la position courante est apparue au moins {@code times} fois dans l'historique
   * (incluant la position courante elle-même). Conforme à SPEC §5.2.3 et ADR-012.
   *
   * <p>Optimisation : la borne basse de la fenêtre de scan est {@code max(0, historySize -
   * halfmoveClock)} car aucune répétition ne peut traverser un coup irréversible (capture, avance
   * de pion, EP).
   *
   * @param times {@code 2} pour la convention twofold (recherche), {@code 3} pour threefold (règles
   *     FIDE) ; {@code 1} retourne toujours {@code true} (I-GS-3) ; {@code <1} retourne {@code
   *     false}
   * @return {@code true} si la position courante apparaît au moins {@code times} fois
   */
  public boolean isRepetition(int times) {
    if (times < 1) {
      return false;
    }
    int occurrences = 1;
    if (occurrences >= times) {
      return true;
    }
    long h = currentPosition.zobristHash();
    int hm = currentPosition.halfmoveClock();
    int limit = Math.max(0, historySize - hm);
    for (int i = historySize - 1; i >= limit; i--) {
      if (historyPositions[i].zobristHash() == h) {
        occurrences += 1;
        if (occurrences >= times) {
          return true;
        }
      }
    }
    return false;
  }

  // ---------------------------------------------------------------------------------------------
  // API : génération de coups (délégation à MoveGen)
  // ---------------------------------------------------------------------------------------------

  /**
   * Génère les coups légaux de la position courante dans le buffer fourni à partir de l'offset
   * donné. Délégation directe à {@link MoveGen#generateMoves(Position, int[], int)}.
   *
   * @param buffer buffer de destination, taille &geq; {@code offset + RECOMMENDED_BUFFER_SIZE}
   * @param offset index de départ dans le buffer
   * @return nombre de coups écrits
   */
  public int generateMoves(int[] buffer, int offset) {
    return MoveGen.generateMoves(currentPosition, buffer, offset);
  }

  // ---------------------------------------------------------------------------------------------
  // API : I/O et reset
  // ---------------------------------------------------------------------------------------------

  /**
   * Réinitialise à la position de départ et purge l'historique. Équivalent à {@code
   * setFromFen(Fen.STARTPOS)}.
   */
  public void reset() {
    setFromFen(Fen.STARTPOS);
  }

  /**
   * Reconfigure le GameState à partir d'une FEN et purge l'historique.
   *
   * @param fen FEN à parser
   * @throws IllegalArgumentException si la FEN est mal formée ou la position invalide
   */
  public void setFromFen(String fen) {
    Fen.parse(fen, currentPosition);
    historySize = 0;
  }

  /** Retourne la FEN de la position courante (cf. {@link Position#toFen()}). */
  public String toFen() {
    return currentPosition.toFen();
  }

  // ---------------------------------------------------------------------------------------------
  // Détection terminale (cf. SPEC §6)
  // ---------------------------------------------------------------------------------------------

  /**
   * Indique si la règle des 50 coups est atteinte : 100 demi-coups sans capture, avance de pion ou
   * EP (cf. SPEC §6.3).
   *
   * @return {@code true} si {@code halfmoveClock >= 100}
   */
  public boolean isFiftyMoveRule() {
    return currentPosition.halfmoveClock >= 100;
  }

  /**
   * Indique si la position est en matériel insuffisant selon les règles FIDE strictes (cf. SPEC
   * §6.4).
   *
   * <p>Cas reconnus : K vs K, KN vs K, KB vs K, et toute position ne contenant que rois et fous où
   * tous les fous des deux camps sont sur cases de même couleur.
   *
   * <p>Cas explicitement exclus (théoriquement gagnables) : KNN vs K, KB vs KN, KN vs KN, KBB
   * couleurs différentes, et toute position avec au moins un pion, une tour ou une dame.
   *
   * @return {@code true} si la position est en matériel insuffisant FIDE
   */
  public boolean isInsufficientMaterial() {
    long pawns =
        currentPosition.pieceBB[Piece.WHITE_PAWN] | currentPosition.pieceBB[Piece.BLACK_PAWN];
    long rooks =
        currentPosition.pieceBB[Piece.WHITE_ROOK] | currentPosition.pieceBB[Piece.BLACK_ROOK];
    long queens =
        currentPosition.pieceBB[Piece.WHITE_QUEEN] | currentPosition.pieceBB[Piece.BLACK_QUEEN];
    if ((pawns | rooks | queens) != 0L) {
      return false;
    }

    long knights =
        currentPosition.pieceBB[Piece.WHITE_KNIGHT] | currentPosition.pieceBB[Piece.BLACK_KNIGHT];
    long bishops =
        currentPosition.pieceBB[Piece.WHITE_BISHOP] | currentPosition.pieceBB[Piece.BLACK_BISHOP];

    int nbKnights = Long.bitCount(knights);
    int nbBishops = Long.bitCount(bishops);

    if (nbKnights == 0 && nbBishops == 0) {
      return true;
    }
    if (nbKnights + nbBishops == 1) {
      return true;
    }
    if (nbKnights == 0 && nbBishops >= 2) {
      if ((bishops & Bitboards.LIGHT_SQUARES) == bishops) {
        return true;
      }
      if ((bishops & Bitboards.DARK_SQUARES) == bishops) {
        return true;
      }
    }
    return false;
  }

  /**
   * Indique si la position est un pat : le côté au trait n'est PAS en échec et n'a aucun coup légal
   * (cf. SPEC §6.1).
   *
   * @return {@code true} si pat
   */
  public boolean isStalemate() {
    if (currentPosition.isInCheck()) {
      return false;
    }
    return MoveGen.generateMoves(currentPosition, scratchMoveBuffer, 0) == 0;
  }

  /**
   * Indique si la position est un mat : le côté au trait est en échec et n'a aucun coup légal (cf.
   * SPEC §6.1).
   *
   * @return {@code true} si mat
   */
  public boolean isCheckmate() {
    if (!currentPosition.isInCheck()) {
      return false;
    }
    return MoveGen.generateMoves(currentPosition, scratchMoveBuffer, 0) == 0;
  }

  /**
   * Indique si la position est terminale (mat, pat, triple répétition, 50 coups ou matériel
   * insuffisant). Les tests cheap sont effectués en premier pour éviter la génération de coups
   * lorsque possible (cf. SPEC §6.5).
   *
   * @return {@code true} si la partie est terminée
   */
  public boolean isTerminal() {
    if (isFiftyMoveRule()) {
      return true;
    }
    if (isInsufficientMaterial()) {
      return true;
    }
    if (isRepetition(3)) {
      return true;
    }
    return MoveGen.generateMoves(currentPosition, scratchMoveBuffer, 0) == 0;
  }

  /**
   * Retourne le résultat de la partie (cf. SPEC §6.5).
   *
   * <p>Dispatch sans double génération de coups : les conditions cheap (50 coups, matériel
   * insuffisant, répétition) sont testées en premier, puis une seule génération de coups discrimine
   * mat / pat / partie en cours.
   *
   * @return {@link Result#WIN_WHITE}, {@link Result#WIN_BLACK}, {@link Result#DRAW} ou {@link
   *     Result#IN_PROGRESS}
   */
  public Result getResult() {
    if (isFiftyMoveRule() || isInsufficientMaterial() || isRepetition(3)) {
      return Result.DRAW;
    }
    int count = MoveGen.generateMoves(currentPosition, scratchMoveBuffer, 0);
    if (count > 0) {
      return Result.IN_PROGRESS;
    }
    if (currentPosition.isInCheck()) {
      return currentPosition.sideToMove == Color.WHITE ? Result.WIN_BLACK : Result.WIN_WHITE;
    }
    return Result.DRAW;
  }

  // ---------------------------------------------------------------------------------------------
  // Plans NN AlphaZero (cf. SPEC §7)
  // ---------------------------------------------------------------------------------------------

  /**
   * Encodeur scalaire de référence (cf. SPEC §7.6). Implémentation utilisée par défaut quand aucun
   * foncteur vectorisé n'est injecté. Sa correction sert également d'oracle pour valider les
   * implémentations vectorisées (cohérence bit-à-bit garantie par le contrat de {@link
   * BitboardPlaneEncoder}).
   */
  static void scalarEncode(long bb, float[] dest, int planeOffset) {
    for (int sq = 0; sq < 64; sq++) {
      dest[planeOffset + sq] = ((bb >>> sq) & 1L) == 0L ? 0.0f : 1.0f;
    }
  }

  /**
   * Remplit le buffer fourni avec les 119 plans d'entrée du réseau NN au format AlphaZero (cf. SPEC
   * §7).
   *
   * <p>Utilise l'encodeur scalaire de référence ({@link #scalarEncode(long, float[], int)}). Pour
   * injecter un encodeur vectorisé (Vector API), utiliser {@link #toPlanes(float[], int,
   * BitboardPlaneEncoder)}.
   *
   * @param dest buffer destination, taille &geq; {@code offset + NN_PLANES * 64} (= {@code offset +
   *     7616})
   * @param offset index de départ dans {@code dest}
   */
  public void toPlanes(float[] dest, int offset) {
    toPlanes(dest, offset, GameState::scalarEncode);
  }

  /**
   * Remplit le buffer fourni avec les 119 plans NN AlphaZero, en utilisant l'encodeur fourni pour
   * la conversion bitboard → plan. Conforme à SPEC §7.7.
   *
   * <p>Layout du buffer (cf. §7.5) : 8 timestamps × 14 plans (112 plans temporels) puis 7 plans
   * constants (Color, move count, 4 castling rights, no-progress count). Inversion verticale via
   * {@link Long#reverseBytes(long)} si {@code sideToMove == BLACK} (cf. §7.3).
   *
   * @param dest buffer destination, taille &geq; {@code offset + NN_PLANES * 64} (= {@code offset +
   *     7616})
   * @param offset index de départ dans {@code dest}
   * @param encoder foncteur d'encodage bitboard → plan ; non nul
   */
  public void toPlanes(float[] dest, int offset, BitboardPlaneEncoder encoder) {
    int us = currentPosition.sideToMove;
    int them = us ^ 1;
    boolean needFlip = us == Color.BLACK;

    // 1. Plans temporels (8 timestamps × 14 plans = 112 plans).
    // t == 0 : position courante. t == k > 0 : historyPositions[historySize - k].
    for (int t = 0; t < NN_HISTORY_LENGTH; t++) {
      int tsOffset = offset + t * 14 * 64;
      Position snapshot;
      if (t == 0) {
        snapshot = currentPosition;
      } else {
        int idx = historySize - t;
        if (idx < 0) {
          // Profondeur insuffisante : remplir 14 plans à 0.0 (§7.5).
          Arrays.fill(dest, tsOffset, tsOffset + 14 * 64, 0.0f);
          continue;
        }
        snapshot = historyPositions[idx];
      }
      encodeTimestamp(snapshot, dest, tsOffset, us, them, needFlip, encoder);
    }

    // 2. Plans constants (7 plans, indices 112..118).
    int constOffset = offset + 112 * 64;

    // 2a. Color (plan 112).
    fillPlane(dest, constOffset, us == Color.WHITE ? 1.0f : 0.0f);

    // 2b. Total move count (plan 113), normalisé par 99.
    float moveCountNorm = Math.min(currentPosition.fullmoveNumber, 99) / 99.0f;
    fillPlane(dest, constOffset + 64, moveCountNorm);

    // 2c. Castling rights (plans 114..117), mapping P1/P2 dépendant de sideToMove (§7.4).
    int cr = currentPosition.castlingRights;
    int p1KS = us == Color.WHITE ? Castling.WHITE_KINGSIDE : Castling.BLACK_KINGSIDE;
    int p1QS = us == Color.WHITE ? Castling.WHITE_QUEENSIDE : Castling.BLACK_QUEENSIDE;
    int p2KS = us == Color.WHITE ? Castling.BLACK_KINGSIDE : Castling.WHITE_KINGSIDE;
    int p2QS = us == Color.WHITE ? Castling.BLACK_QUEENSIDE : Castling.WHITE_QUEENSIDE;
    fillPlane(dest, constOffset + 2 * 64, (cr & p1KS) != 0 ? 1.0f : 0.0f);
    fillPlane(dest, constOffset + 3 * 64, (cr & p1QS) != 0 ? 1.0f : 0.0f);
    fillPlane(dest, constOffset + 4 * 64, (cr & p2KS) != 0 ? 1.0f : 0.0f);
    fillPlane(dest, constOffset + 5 * 64, (cr & p2QS) != 0 ? 1.0f : 0.0f);

    // 2d. No-progress count (plan 118), normalisé par 100.
    fillPlane(dest, constOffset + 6 * 64, currentPosition.halfmoveClock / 100.0f);
  }

  /**
   * Encode un timestamp : 12 plans pièces (P1 PAWN..KING puis P2 PAWN..KING) puis 2 plans de
   * répétition (cf. SPEC §7.7).
   */
  private void encodeTimestamp(
      Position snapshot,
      float[] dest,
      int offset,
      int us,
      int them,
      boolean needFlip,
      BitboardPlaneEncoder encoder) {
    // 6 plans P1 (joueur au trait de la position courante, indépendamment du snapshot).
    for (int pieceType = 0; pieceType < PieceType.NB_PIECE_TYPES; pieceType++) {
      long bb = snapshot.pieceBB[Piece.make(us, pieceType)];
      if (needFlip) {
        bb = Long.reverseBytes(bb);
      }
      encoder.encode(bb, dest, offset + pieceType * 64);
    }
    // 6 plans P2.
    for (int pieceType = 0; pieceType < PieceType.NB_PIECE_TYPES; pieceType++) {
      long bb = snapshot.pieceBB[Piece.make(them, pieceType)];
      if (needFlip) {
        bb = Long.reverseBytes(bb);
      }
      encoder.encode(bb, dest, offset + (6 + pieceType) * 64);
    }

    // 2 plans répétition (§7.7) : occurrences strictes dans l'historique (excluant le snapshot
    // lui-même), bornées par halfmoveClock courant.
    int rep = countRepetitions(snapshot.zobristHash);
    fillPlane(dest, offset + 12 * 64, rep >= 1 ? 1.0f : 0.0f);
    fillPlane(dest, offset + 13 * 64, rep >= 2 ? 1.0f : 0.0f);
  }

  /**
   * Compte les occurrences strictes de {@code targetHash} dans l'historique (excluant la position
   * courante). Optimisation par borne basse {@code halfmoveClock} : aucune répétition ne peut
   * traverser un coup irréversible (cf. SPEC §7.7).
   */
  private int countRepetitions(long targetHash) {
    int count = 0;
    int hm = currentPosition.halfmoveClock;
    int limit = Math.max(0, historySize - hm);
    for (int i = historySize - 1; i >= limit; i--) {
      if (historyPositions[i].zobristHash == targetHash) {
        count++;
      }
    }
    return count;
  }

  /** Remplit 64 cases consécutives de {@code dest} avec {@code value}. */
  private static void fillPlane(float[] dest, int offset, float value) {
    Arrays.fill(dest, offset, offset + 64, value);
  }

  // ---------------------------------------------------------------------------------------------
  // Deep copy (pour parallélisation MCTS — ajouté pour engine v1.2.0 batched mode)
  // ---------------------------------------------------------------------------------------------

  /**
   * Construit une copie indépendante de ce {@code GameState}. La copie partage aucun état mutable
   * avec l'original : applyMove sur l'une n'affecte pas l'autre.
   *
   * <p>Cas d'usage : parallélisation MCTS où plusieurs threads doivent descendre l'arbre depuis la
   * même racine (cf. engine v1.2.0 batched, ADR-016). Le thread principal possède le {@code
   * GameState} racine ; chaque worker thread reçoit une copie via {@code rootState.copy()} qu'il
   * mute librement via applyMove/unapplyLastMove pendant sa simulation.
   *
   * <p>Allocation : nouveau {@code GameState} (qui pré-alloue 1024 {@link Position} via le block
   * initializer). Chaque Position de l'historique courant est dup-copiée via {@link
   * Position#copyFrom(Position)}. Les Position au-delà de {@code historySize} restent à leur état
   * pré-alloué (contenu obsolète, sera écrasé par le prochain applyMove). Le scratch buffer n'est
   * pas copié (pas d'état persistant).
   *
   * <p>Performance : ~10-50μs typique selon historySize. Ordre de grandeur : ~1.7 MB d'allocation
   * pour le GameState complet (1024 Positions × ~1.6 KB chacune via leurs bitboards).
   *
   * @return un nouveau {@code GameState} indépendant, à l'état identique
   */
  public GameState copy() {
    GameState clone = new GameState();
    clone.currentPosition.copyFrom(this.currentPosition);
    for (int i = 0; i < this.historySize; i++) {
      clone.historyPositions[i].copyFrom(this.historyPositions[i]);
    }
    clone.historySize = this.historySize;
    return clone;
  }

  /**
   * Reset l'état de ce {@code GameState} à l'état de {@code other} via mutation in-place. Aucune
   * allocation. Cas d'usage : un thread du mode batched MCTS qui réutilise un {@code GameState}
   * pré-alloué entre simulations (cf. ADR-016).
   *
   * <p>Performance : ~5-30μs typique selon {@code other.historySize}. Beaucoup plus rapide que
   * {@link #copy()} car ne fait pas d'allocation.
   *
   * <p>Préconditions : {@code other != null}. Ne JAMAIS appeler avec {@code other == this}
   * (undefined behavior dans ce cas, non testé).
   *
   * @param other état à copier dans {@code this}, non null
   */
  public void copyFrom(GameState other) {
    this.currentPosition.copyFrom(other.currentPosition);
    for (int i = 0; i < other.historySize; i++) {
      this.historyPositions[i].copyFrom(other.historyPositions[i]);
    }
    this.historySize = other.historySize;
  }
}
