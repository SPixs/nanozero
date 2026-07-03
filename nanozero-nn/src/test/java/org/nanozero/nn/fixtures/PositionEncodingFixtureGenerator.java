package org.nanozero.nn.fixtures;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.nanozero.board.Fen;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;

/**
 * Script standalone qui génère des fixtures de parité {@link GameState#toPlanes(float[], int)} pour
 * tests Python {@code encode_position} <-> Java côté training phase 1.0.0-4-a.
 *
 * <p>Format binaire big-endian (DataOutputStream Java par défaut) :
 *
 * <pre>{@code
 * [int32 BE: n_fixtures]
 * Pour chaque fixture :
 *   [int32 BE: history_depth] // nombre de coups joués depuis startpos (utile Python pour
 *                              //  reconstruire les snapshots historiques via push)
 *   [int32 BE: fen_len]
 *   [byte[fen_len]: FEN UTF-8]
 *   [int32 BE: n_uci_moves]   // = history_depth, mais explicite pour robustesse format
 *   pour chaque coup joué (dans l'ordre depuis startpos) :
 *     [int32 BE: uci_len]
 *     [byte[uci_len]: UCI UTF-8]
 *   [int32 BE: n_planes=119]
 *   [int32 BE: rows=8]
 *   [int32 BE: cols=8]
 *   [float32 BE × 119*8*8: planes data row-major (plane, rank, file)]
 * }</pre>
 *
 * <p>Le bloc des UCI moves permet au lecteur Python de reproduire l'historique exact via {@code
 * board.push(chess.Move.from_uci(...))} — critique car {@code encode_position} consomme
 * l'historique des 7 plies précédentes pour les plans temporels.
 *
 * <p>Distribution : 1 fixture startpos (depth=0) + N-1 fixtures à profondeurs aléatoires {@code [0,
 * 80]}, dont au moins 10 avec depth ≥ 8 (exerce les plans historiques complets) et quelques
 * positions avec depth ≥ 50 (exerce les répétitions possibles + halfmove clock élevé).
 *
 * <p>Usage :
 *
 * <pre>{@code
 * mvn -pl nanozero-nn exec:java \
 *   -Dexec.mainClass=org.nanozero.nn.fixtures.PositionEncodingFixtureGenerator \
 *   -Dexec.classpathScope=test \
 *   -Dexec.args="nanozero-training/tests/fixtures/position_encoding_parity.bin 100 42"
 * }</pre>
 */
public final class PositionEncodingFixtureGenerator {

  private static final int N_PLANES = 119;
  private static final int ROWS = 8;
  private static final int COLS = 8;
  private static final int PLANES_FLOAT_COUNT = N_PLANES * ROWS * COLS;

  private PositionEncodingFixtureGenerator() {
    throw new AssertionError("Non-instantiable");
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 3) {
      System.err.println("Usage: <output-bin-path> <n-fixtures> <seed>");
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
    int[] moveBuf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    float[] planes = new float[PLANES_FLOAT_COUNT];

    try (DataOutputStream out =
        new DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(outPath)))) {

      out.writeInt(nFixtures);

      int written = 0;
      int attempts = 0;
      int maxAttempts = nFixtures * 20;
      int depthHistGte8 = 0;

      while (written < nFixtures && attempts < maxAttempts) {
        attempts++;
        // 1ʳᵉ fixture : startpos exact (depth=0).
        int depth;
        if (written == 0) {
          depth = 0;
        } else if (depthHistGte8 < Math.max(10, nFixtures / 10)) {
          // Force quelques depths >= 8 pour exercer les plans historiques.
          depth = 8 + rng.nextInt(73); // [8, 80]
        } else {
          depth = rng.nextInt(81); // [0, 80]
        }

        GameState gs = new GameState(Fen.STARTPOS);
        java.util.List<Integer> playedMoves = new java.util.ArrayList<>(depth);

        boolean stuck = false;
        for (int ply = 0; ply < depth; ply++) {
          int n = MoveGen.generateMoves(gs.currentPosition(), moveBuf, 0);
          if (n == 0) {
            stuck = true;
            break;
          }
          int chosen = moveBuf[rng.nextInt(n)];
          playedMoves.add(chosen);
          gs.applyMove(chosen);
        }
        if (stuck) {
          continue;
        }

        // Encode 119 planes (clear buffer first since toPlanes ne touche pas les
        // zones à zéro absolu en cas d'history pads, mais on initialise quand même).
        java.util.Arrays.fill(planes, 0.0f);
        gs.toPlanes(planes, 0);

        // Sanity : tous les floats sont finite + dans [0.0, 1.0].
        for (int i = 0; i < PLANES_FLOAT_COUNT; i++) {
          float v = planes[i];
          if (Float.isNaN(v) || Float.isInfinite(v)) {
            throw new IllegalStateException("NaN/Inf in planes at index " + i);
          }
          if (v < 0.0f || v > 1.0f) {
            throw new IllegalStateException("Plane value out of [0, 1] at index " + i + ": " + v);
          }
        }

        // Header fixture.
        out.writeInt(depth);
        String fen = Fen.write(gs.currentPosition());
        byte[] fenBytes = fen.getBytes(StandardCharsets.UTF_8);
        out.writeInt(fenBytes.length);
        out.write(fenBytes);

        // Bloc UCI moves.
        out.writeInt(playedMoves.size());
        for (int mv : playedMoves) {
          byte[] uciBytes = org.nanozero.board.Move.toUci(mv).getBytes(StandardCharsets.UTF_8);
          out.writeInt(uciBytes.length);
          out.write(uciBytes);
        }

        // Planes meta + data.
        out.writeInt(N_PLANES);
        out.writeInt(ROWS);
        out.writeInt(COLS);
        for (int i = 0; i < PLANES_FLOAT_COUNT; i++) {
          out.writeFloat(planes[i]);
        }

        written++;
        if (depth >= 8) {
          depthHistGte8++;
        }
      }
      if (written < nFixtures) {
        throw new IllegalStateException(
            "Seulement "
                + written
                + " fixtures générées sur "
                + nFixtures
                + " demandées (max attempts atteint).");
      }
      System.out.println(
          "Wrote " + nFixtures + " fixtures to " + outPath + " (depth>=8: " + depthHistGte8 + ")");
    }
  }
}
