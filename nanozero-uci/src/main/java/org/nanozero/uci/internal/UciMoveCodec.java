package org.nanozero.uci.internal;

import java.util.Objects;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.PieceType;
import org.nanozero.board.Square;

/**
 * Codec bidirectionnel UCI long algebraic ↔ {@code Move} int (cf. SPEC §5.4, ADR-005, §12 phase 2).
 *
 * <p>Format UCI long algebraic :
 *
 * <ul>
 *   <li>Mouvement standard : {@code "<from><to>"} (ex. {@code "e2e4"}, {@code "g1f3"}).
 *   <li>Promotion : {@code "<from><to><piece>"} avec {@code piece ∈ {q, r, b, n}} (ex. {@code
 *       "e7e8q"}). Sortie strictement minuscule ; entrée case-insensitive.
 *   <li>Roque : {@code "<kingFrom><kingTo>"} (ex. {@code "e1g1"} pour O-O blanc, {@code "e1c1"}
 *       pour O-O-O blanc). Pas de marqueur spécial.
 *   <li>En passant : {@code "<pawnFrom><epSquare>"} (ex. {@code "e5d6"}). Pas de marqueur spécial ;
 *       le {@link MoveType#EN_PASSANT} est inféré du contexte au decode.
 * </ul>
 *
 * <p><strong>Stratégie d'implémentation</strong> (cf. ADR-005) :
 *
 * <ul>
 *   <li>{@link #encode} : décortique le {@link Move} via les helpers neutres {@code
 *       Move.from}/{@code to}/{@code type}/{@code promo}, puis utilise {@code Square.toAlgebraic}
 *       pour la conversion case → string. L'assemblage UCI 4/5-char est fait localement dans ce
 *       module.
 *   <li>{@link #decode} : itère sur {@code state.generateMoves} et matche by {@code from}/{@code
 *       to} (et promotion si chaîne 5 chars). Plus robuste qu'une reconstruction manuelle qui
 *       devrait dupliquer la classification {@link MoveType} (CASTLING / EN_PASSANT / PROMOTION /
 *       NORMAL).
 * </ul>
 *
 * <p><strong>Note ADR-005 vs board</strong> : {@code board.Move.toUci}/{@code fromUci} existent
 * déjà côté {@code nanozero-board}, contredisant partiellement ADR-005 (« codec dans uci »). Cette
 * classe N'UTILISE PAS ces helpers pour respecter strictement l'ADR. Ambiguïté à documenter en
 * queue d'amendements phase 8.
 *
 * <p>Coût {@code decode} : {@code O(n)} sur les coups légaux ({@code n ≤ 218}). UCI est
 * low-throughput, négligeable.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciMoveCodec {

  private UciMoveCodec() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Encode un {@link Move} int en chaîne UCI long algebraic.
   *
   * @param move coup encodé selon le format {@link Move} (16 bits)
   * @return chaîne UCI 4 ou 5 caractères
   */
  public static String encode(int move) {
    String fromAlg = Square.toAlgebraic(Move.from(move));
    String toAlg = Square.toAlgebraic(Move.to(move));
    if (Move.type(move) != MoveType.PROMOTION) {
      return fromAlg + toAlg;
    }
    char promoChar = promoBitsToUciChar(Move.promo(move));
    return fromAlg + toAlg + promoChar;
  }

  /**
   * Décode une chaîne UCI long algebraic en {@link Move} int dans le contexte de {@code state}.
   *
   * <p>Le contexte est nécessaire pour disambiguer :
   *
   * <ul>
   *   <li>Roi en {@code e1} → {@code g1} : {@link MoveType#CASTLING} (jamais un push de 2 cases qui
   *       serait illégal de toute manière).
   *   <li>Pion vers la case {@link org.nanozero.board.Position#epSquare()} : {@link
   *       MoveType#EN_PASSANT} vs capture standard.
   *   <li>Coup illégal (par exemple syntaxiquement valide mais non présent dans {@code
   *       legalMoves(state)}) : {@link IllegalArgumentException}.
   * </ul>
   *
   * <p>L'entrée est case-insensitive sur le caractère de promotion (tolérance UCI). La sortie de
   * {@link #encode} est toujours strictement minuscule.
   *
   * @param uciMove chaîne UCI 4 ou 5 caractères
   * @param state position courante (l'engine sera dans cet état avant le coup)
   * @return {@link Move} int correspondant au coup légal trouvé
   * @throws NullPointerException si {@code uciMove} ou {@code state} est null
   * @throws IllegalArgumentException si la chaîne est mal formée (longueur, case, promotion) ou si
   *     aucun coup légal de {@code state} ne correspond
   */
  public static int decode(String uciMove, GameState state) {
    Objects.requireNonNull(uciMove, "uciMove must not be null");
    Objects.requireNonNull(state, "state must not be null");
    int len = uciMove.length();
    if (len != 4 && len != 5) {
      throw new IllegalArgumentException(
          "invalid UCI move length (expected 4 or 5): \"" + uciMove + "\"");
    }
    int from = parseSquareOrThrow(uciMove.substring(0, 2), uciMove);
    int to = parseSquareOrThrow(uciMove.substring(2, 4), uciMove);
    int wantedPromoBits = -1;
    if (len == 5) {
      char promoChar = Character.toLowerCase(uciMove.charAt(4));
      wantedPromoBits = uciCharToPromoBitsOrThrow(promoChar, uciMove);
    }
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buffer, 0);
    for (int i = 0; i < n; i++) {
      int candidate = buffer[i];
      if (Move.from(candidate) != from || Move.to(candidate) != to) {
        continue;
      }
      boolean candidateIsPromo = Move.type(candidate) == MoveType.PROMOTION;
      if (wantedPromoBits == -1) {
        if (!candidateIsPromo) {
          return candidate;
        }
      } else {
        if (candidateIsPromo && Move.promo(candidate) == wantedPromoBits) {
          return candidate;
        }
      }
    }
    throw new IllegalArgumentException(
        "UCI move \"" + uciMove + "\" not found among legal moves of position");
  }

  /**
   * Convertit une valeur de promotion 2-bit ({@code 0=N, 1=B, 2=R, 3=Q}) vers son caractère UCI
   * minuscule.
   */
  private static char promoBitsToUciChar(int promoBits) {
    return switch (promoBits) {
      case 0 -> 'n';
      case 1 -> 'b';
      case 2 -> 'r';
      case 3 -> 'q';
      default -> throw new IllegalStateException("invalid promo bits: " + promoBits);
    };
  }

  /**
   * Convertit un caractère UCI de promotion en valeur 2-bit. Note : le {@link PieceType} a {@code
   * KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4} alors que les bits {@code Move} stockent {@code KNIGHT=0,
   * BISHOP=1, ROOK=2, QUEEN=3} (cf. SPEC-board §3.4 et {@link Move#pieceTypeToPromo}).
   */
  private static int uciCharToPromoBitsOrThrow(char c, String uciMove) {
    return switch (c) {
      case 'n' -> 0;
      case 'b' -> 1;
      case 'r' -> 2;
      case 'q' -> 3;
      default ->
          throw new IllegalArgumentException(
              "invalid UCI promotion char '" + c + "' in move \"" + uciMove + "\"");
    };
  }

  /**
   * Wrapper {@link Square#fromAlgebraic} qui re-emballe l'éventuelle exception en {@link
   * IllegalArgumentException} avec le contexte de la chaîne UCI complète, plus parlant pour les
   * messages d'erreur du parser (phase 3).
   */
  private static int parseSquareOrThrow(String squareStr, String fullUciMove) {
    try {
      return Square.fromAlgebraic(squareStr);
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "invalid square \"" + squareStr + "\" in UCI move \"" + fullUciMove + "\"", e);
    }
  }
}
