package org.nanozero.board;

/**
 * Benchmark simple (chronométrage post-warmup, hors JaCoCo, hors fork) des cibles §9.2.
 *
 * <p>Exécution manuelle :
 *
 * <pre>
 *   mvn -pl nanozero-board test-compile
 *   java -cp nanozero-board/target/classes:nanozero-board/target/test-classes \
 *        org.nanozero.board.PerfMeasurements
 * </pre>
 *
 * <p>Pas de JaCoCo agent (exécution directe), pas de fork Surefire. Les chiffres servent de
 * validation de phase 12. La mesure normative reste réservée à {@code nanozero-bench} (JMH, à
 * venir).
 */
public final class PerfMeasurements {

  private PerfMeasurements() {}

  private static final String STARTPOS_FEN = Fen.STARTPOS;
  private static final String KIWIPETE_FEN =
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

  public static void main(String[] args) {
    Bitboards.warmup();
    Zobrist.warmup();

    System.out.println("=== Cibles §9.2 — mesures simples (post-warmup, hors JaCoCo) ===");
    benchmarkGenerateMoves(STARTPOS_FEN, "generateMoves(startpos)", 200);
    benchmarkGenerateMoves(KIWIPETE_FEN, "generateMoves(kiwipete)", 500);
    benchmarkApplyMove();
    benchmarkCopyFrom();
  }

  private static void benchmarkGenerateMoves(String fen, String label, int targetNs) {
    Position p = Fen.parse(fen);
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    long warmupIters = 5_000_000;
    long measureIters = 10_000_000;

    long acc = 0;
    for (long i = 0; i < warmupIters; i++) {
      acc ^= MoveGen.generateMoves(p, buffer, 0);
    }
    long t0 = System.nanoTime();
    for (long i = 0; i < measureIters; i++) {
      acc ^= MoveGen.generateMoves(p, buffer, 0);
    }
    long t1 = System.nanoTime();
    if (acc == 12345L) {
      System.out.println("(unreachable, side-effect anchor)");
    }
    long nsPerCall = (t1 - t0) / measureIters;
    report(label, nsPerCall, targetNs);
  }

  private static void benchmarkApplyMove() {
    // Coup normal : 1.e2-e4 sur startpos. Sink via zobristHash pour empêcher élision JIT.
    Position template = Fen.parse(STARTPOS_FEN);
    int e2e4 = Move.fromUci("e2e4", template);
    Position scratch = new Position();

    long warmupIters = 5_000_000;
    long measureIters = 10_000_000;

    long acc = 0;
    for (long i = 0; i < warmupIters; i++) {
      scratch.copyFrom(template);
      scratch.applyMove(e2e4);
      acc ^= scratch.zobristHash();
    }
    long t0 = System.nanoTime();
    for (long i = 0; i < measureIters; i++) {
      scratch.copyFrom(template);
      scratch.applyMove(e2e4);
      acc ^= scratch.zobristHash();
    }
    long t1 = System.nanoTime();
    if (acc == 12345L) {
      System.out.println("(unreachable)");
    }
    long pairNs = (t1 - t0) / measureIters;
    long copyFromCost = measureCopyFromCost();
    long applyMoveNs = Math.max(0, pairNs - copyFromCost);
    report("applyMove (coup normal e2e4)", applyMoveNs, 50);
  }

  private static long measureCopyFromCost() {
    // Alterne entre deux templates pour empêcher l'élimination JIT (le résultat de copyFrom doit
    // varier en fonction de l'itération, sinon le JIT peut hisser le calcul hors de la boucle).
    Position templateA = Fen.parse(STARTPOS_FEN);
    Position templateB = Fen.parse(KIWIPETE_FEN);
    Position scratch = new Position();
    long warmupIters = 5_000_000;
    long measureIters = 20_000_000;

    long acc = 0;
    for (long i = 0; i < warmupIters; i++) {
      scratch.copyFrom((i & 1) == 0 ? templateA : templateB);
      acc ^= scratch.zobristHash();
    }
    long t0 = System.nanoTime();
    for (long i = 0; i < measureIters; i++) {
      scratch.copyFrom((i & 1) == 0 ? templateA : templateB);
      acc ^= scratch.zobristHash();
    }
    long t1 = System.nanoTime();
    if (acc == 12345L) {
      System.out.println("(unreachable)");
    }
    return (t1 - t0) / measureIters;
  }

  private static void benchmarkCopyFrom() {
    long ns = measureCopyFromCost();
    report("Position.copyFrom(other)", ns, 30);
  }

  private static void report(String label, long nsPerCall, int targetNs) {
    String mark = nsPerCall <= targetNs ? "OK" : (nsPerCall <= targetNs * 2L ? "WARN" : "FAIL");
    System.out.printf(
        "  %-40s : %5d ns/call  (cible §9.2 < %d ns) %s%n", label, nsPerCall, targetNs, mark);
  }
}
