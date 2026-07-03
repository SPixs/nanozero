package org.nanozero.nn.fixtures;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.nanozero.board.Fen;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.Position;
import org.nanozero.nn.MoveEncoding;

/**
 * Script standalone qui génère des fixtures de parité {@link MoveEncoding#encode(int, int)} pour
 * tests Python <-> Java côté {@code nanozero-training} phase 1.0.0-4-a.
 *
 * <p>Format CSV : {@code fen,uci,index}. Header inclus. {@code index} est dans {@code [0, 4672)}.
 *
 * <p>Usage :
 *
 * <pre>{@code
 * mvn -pl nanozero-nn exec:java \
 *   -Dexec.mainClass=org.nanozero.nn.fixtures.MoveEncodingFixtureGenerator \
 *   -Dexec.classpathScope=test \
 *   -Dexec.args="nanozero-training/tests/fixtures/move_encoding_parity.csv 1000 42"
 * }</pre>
 *
 * <p>Stratégie de génération : random walk depuis {@code startpos}, profondeur uniforme dans {@code
 * [0, 60]}, choix d'1 coup légal aléatoire. Garantit couverture standard moves + promotions + en
 * passant + castling sur 1000 fixtures.
 *
 * <p>Note placement : ce générateur vit dans {@code nanozero-nn/src/test/java/...} car {@code
 * MoveEncoding} est dans le module {@code nanozero-nn}, qui dépend de {@code nanozero-board}.
 * L'inverse aurait créé une dépendance circulaire.
 */
public final class MoveEncodingFixtureGenerator {

  private MoveEncodingFixtureGenerator() {
    throw new AssertionError("Non-instantiable");
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 3) {
      System.err.println("Usage: <output-csv-path> <n-fixtures> <seed>");
      System.exit(1);
    }
    Path outPath = Path.of(args[0]);
    int nFixtures = Integer.parseInt(args[1]);
    long seed = Long.parseLong(args[2]);

    if (nFixtures <= 0) {
      throw new IllegalArgumentException("n-fixtures must be > 0, got " + nFixtures);
    }
    if (outPath.getParent() != null) {
      Files.createDirectories(outPath.getParent());
    }

    Random rng = new Random(seed);
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];

    try (BufferedWriter w = Files.newBufferedWriter(outPath, StandardCharsets.UTF_8)) {
      w.write("fen,uci,index\n");
      int written = 0;
      int attempts = 0;
      int maxAttempts = nFixtures * 20; // safety vs trop-de-positions-terminales
      while (written < nFixtures && attempts < maxAttempts) {
        attempts++;
        GameState gs = new GameState(Fen.STARTPOS);
        int depth = rng.nextInt(61); // [0, 60]
        boolean stuck = false;
        for (int ply = 0; ply < depth; ply++) {
          int n = MoveGen.generateMoves(gs.currentPosition(), buffer, 0);
          if (n == 0) {
            stuck = true;
            break; // position terminale avant la profondeur cible
          }
          int chosen = buffer[rng.nextInt(n)];
          gs.applyMove(chosen);
        }
        if (stuck) {
          continue;
        }
        Position pos = gs.currentPosition();
        int n = MoveGen.generateMoves(pos, buffer, 0);
        if (n == 0) {
          continue; // position cible terminale, pas de coup à encoder
        }
        int move = buffer[rng.nextInt(n)];
        int side = pos.sideToMove();
        int index = MoveEncoding.encode(move, side);
        if (index < 0 || index >= MoveEncoding.POLICY_INDICES) {
          throw new IllegalStateException(
              "MoveEncoding.encode hors plage : index=" + index + " for move=" + Move.toUci(move));
        }
        String fen = Fen.write(pos);
        if (fen.indexOf(',') >= 0) {
          throw new IllegalStateException("FEN contient une virgule inattendue : " + fen);
        }
        w.write(fen);
        w.write(',');
        w.write(Move.toUci(move));
        w.write(',');
        w.write(Integer.toString(index));
        w.write('\n');
        written++;
      }
      if (written < nFixtures) {
        throw new IllegalStateException(
            "Seulement "
                + written
                + " fixtures générées sur "
                + nFixtures
                + " demandées (max attempts atteint).");
      }
    }
    System.out.println("Wrote " + nFixtures + " fixtures to " + outPath);
  }
}
