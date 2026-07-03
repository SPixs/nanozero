// Harness de référence Java pour cross-valider le worker JS.
// Dumpe en JSON-lines (1 objet/ligne) : {label, side, planes[7616], moves:[{uci,idx}]}.
// Mode 1 : positions isolées (historique t1-7 = 0).
// Mode 2 : replay UCI depuis startpos (teste l'historique 8-timestamps + la répétition).
//
// Compile : javac -cp <board>/target/classes:<nn>/target/classes -d <out> CrossValHarness.java
// Run     : java  -cp <board>/target/classes:<nn>/target/classes:<out> CrossValHarness

import org.nanozero.board.Fen;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.Position;
import org.nanozero.nn.MoveEncoding;

public class CrossValHarness {

  static void dump(String label, GameState gs) {
    float[] planes = new float[119 * 64];
    gs.toPlanes(planes, 0);
    Position pos = gs.currentPosition();
    int side = pos.sideToMove();
    int[] moves = new int[256];
    int n = gs.generateMoves(moves, 0);

    StringBuilder sb = new StringBuilder(80000);
    sb.append("{\"label\":\"").append(label).append("\",\"side\":").append(side).append(",\"planes\":[");
    for (int i = 0; i < planes.length; i++) {
      if (i > 0) sb.append(',');
      float v = planes[i];
      if (v == (int) v) sb.append((int) v);
      else sb.append(v);
    }
    sb.append("],\"moves\":[");
    for (int i = 0; i < n; i++) {
      if (i > 0) sb.append(',');
      int idx = MoveEncoding.encode(moves[i], side);
      sb.append("{\"uci\":\"").append(Move.toUci(moves[i])).append("\",\"idx\":").append(idx).append("}");
    }
    sb.append("]}");
    System.out.println(sb);
  }

  static int findUci(GameState gs, String uci) {
    int[] moves = new int[256];
    int n = gs.generateMoves(moves, 0);
    for (int i = 0; i < n; i++) if (Move.toUci(moves[i]).equals(uci)) return moves[i];
    throw new RuntimeException("UCI illégal: " + uci);
  }

  public static void main(String[] args) {
    String[] fens = {
      Fen.STARTPOS,
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
      "rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 5",    // deux roqués
      "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",                                     // promotions push+capture
      "8/8/8/3k4/8/3K4/8/8 w - - 5 30",                                        // finale, compteurs
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",      // trait Noirs
      // --- FENs durs Story 0 (silent-poison) ---
      "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",          // ep réel jouable (e5xd6)
      "4k3/8/8/8/5P2/8/8/4K3 b - f3 0 1",                                      // ep f3 isolé non-capturable
      "r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1",                               // roque-sous-échec (e1 cloué)
      "n1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",                                   // sous-promotion N/B/R + captures
    };
    for (int i = 0; i < fens.length; i++) dump("fen" + i, new GameState(fens[i]));

    String[] ucis = {
      "e2e4", "e7e5", "g1f3", "b8c6",
      "f3g1", "c6b8", "g1f3", "b8c6",   // navette -> rép1
      "f3g1", "c6b8", "g1f3", "b8c6",   // navette -> rép2
    };
    GameState gs = new GameState(Fen.STARTPOS);
    dump("ply0", gs);
    for (int i = 0; i < ucis.length; i++) {
      gs.applyMove(findUci(gs, ucis[i]));
      dump("ply" + (i + 1) + ":" + ucis[i], gs);
    }
  }
}
