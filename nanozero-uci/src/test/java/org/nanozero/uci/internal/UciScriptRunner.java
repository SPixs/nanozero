package org.nanozero.uci.internal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Runner pour les scripts d'intégration {@code .uciscript} (cf. SPEC §8.2, §12 phase 7).
 *
 * <p>Format de fichier :
 *
 * <ul>
 *   <li>{@code > input} : ligne à envoyer au handler (sans le préfixe ni LF terminal).
 *   <li>{@code < pattern} : pattern regex Java attendu pour la prochaine ligne de sortie. Le
 *       pattern doit matcher la ligne entière ({@link Pattern#matches(String, CharSequence)}).
 *   <li>{@code #} : commentaire, ligne ignorée.
 *   <li>Ligne vide : ignorée.
 * </ul>
 *
 * <p>L'exécution est in-process : pas de {@code System.in}/{@code System.out} réel, l'output est
 * capturé via un {@link java.io.ByteArrayOutputStream}. Chaque {@code > input} déclenche un {@code
 * UciCommandParser.parse} + {@code UciCommandHandler.handle}. Si le handler retourne {@code QUIT},
 * le runner sort sans erreur (toute ligne {@code < pattern} restante est ignorée silencieusement).
 *
 * <p><strong>Attente fin-de-recherche</strong> : si une ligne {@code < pattern} demande une sortie
 * (typiquement {@code bestmove}) qui n'est pas encore présente dans le buffer capturé, le runner
 * attend jusqu'à 15 s en pollant le buffer toutes les 50 ms. Ceci couvre les cas où le {@code
 * bestmove} arrive depuis l'InfoReporter de manière asynchrone après un {@code go} + {@code stop},
 * ou via fin naturelle.
 *
 * @apiNote Internal — outil de test uniquement, do not depend on this from outside the {@code
 *     nanozero-uci} module test classpath.
 */
final class UciScriptRunner {

  private static final long PATTERN_WAIT_TIMEOUT_MS = 15_000;
  private static final long POLL_INTERVAL_MS = 50;

  private UciScriptRunner() {}

  /**
   * Exécute le script chargé depuis {@code scriptPath} contre {@code state}. Capture la sortie via
   * le {@code ByteArrayOutputStream} fourni (qui doit avoir été wrappé dans le writer du state au
   * moment de la construction).
   *
   * @param scriptPath chemin vers le fichier .uciscript
   * @param state holder UCI déjà initialisé
   * @param capturedOutput buffer wrappé par le writer du state (pour vérifier les patterns)
   * @throws IOException si la lecture du script échoue
   * @throws AssertionError si un pattern ne match pas dans le timeout
   */
  static void runScript(
      Path scriptPath, UciAdapterState state, java.io.ByteArrayOutputStream capturedOutput)
      throws IOException {
    List<String> lines = Files.readAllLines(scriptPath);
    // Curseur sur les bytes consommés du buffer pour assertExpectedLine.
    int consumedBytes = 0;
    for (int lineNum = 0; lineNum < lines.size(); lineNum++) {
      String raw = lines.get(lineNum);
      String line = raw.trim();
      if (line.isEmpty() || line.startsWith("#")) {
        continue;
      }
      if (line.startsWith("> ")) {
        String input = line.substring(2);
        UciCommand cmd = UciCommandParser.parse(input);
        UciCommandHandler.HandleAction action = UciCommandHandler.handle(cmd, state);
        if (action == UciCommandHandler.HandleAction.QUIT) {
          return;
        }
      } else if (line.equals(">")) {
        // ligne ">" sans contenu : envoyer une ligne vide
        UciCommandHandler.handle(UciCommandParser.parse(""), state);
      } else if (line.startsWith("< ")) {
        String pattern = line.substring(2);
        consumedBytes =
            awaitAndAssertLine(pattern, capturedOutput, consumedBytes, scriptPath, lineNum + 1);
      } else if (line.equals("<")) {
        // ligne "<" sans pattern : attend une ligne vide
        consumedBytes =
            awaitAndAssertLine("", capturedOutput, consumedBytes, scriptPath, lineNum + 1);
      } else {
        throw new AssertionError(
            "Script "
                + scriptPath
                + " line "
                + (lineNum + 1)
                + ": expected '> ', '< ', '#' or blank, got: <<<"
                + raw
                + ">>>");
      }
    }
  }

  /**
   * Skip-then-match : avance le curseur dans {@code captured} jusqu'à trouver une ligne qui matche
   * {@code pattern}. Les lignes intermédiaires non-matchantes sont silencieusement skippées
   * (typiquement des lignes {@code info} émises asynchronement par l'InfoReporter entre un {@code
   * go} et le {@code bestmove} final).
   *
   * <p>Retourne le {@code consumedBytes} pointant après le {@code \n} de la ligne consommée.
   *
   * <p>Timeout : si aucune ligne matchante n'apparaît dans {@link #PATTERN_WAIT_TIMEOUT_MS} ms,
   * lève {@link AssertionError} avec le buffer capturé en contexte.
   */
  private static int awaitAndAssertLine(
      String pattern,
      java.io.ByteArrayOutputStream captured,
      int consumedBytes,
      Path scriptPath,
      int lineNum) {
    Pattern compiled = Pattern.compile(pattern);
    long deadline = System.nanoTime() + PATTERN_WAIT_TIMEOUT_MS * 1_000_000L;
    int cursor = consumedBytes;
    while (System.nanoTime() < deadline) {
      String content = captured.toString(java.nio.charset.StandardCharsets.UTF_8);
      // Consomme toutes les lignes complètes disponibles, jusqu'à trouver un match.
      int newlineIdx;
      while ((newlineIdx = content.indexOf('\n', cursor)) >= 0) {
        String candidate = content.substring(cursor, newlineIdx);
        cursor = newlineIdx + 1;
        if (compiled.matcher(candidate).matches()) {
          return cursor;
        }
        // Sinon : ligne intermédiaire (typiquement info pendant search), skip silencieusement.
      }
      try {
        Thread.sleep(POLL_INTERVAL_MS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new AssertionError("interrupted while waiting for pattern", e);
      }
    }
    throw new AssertionError(
        "Script "
            + scriptPath
            + " line "
            + lineNum
            + ": timeout waiting for pattern <<<"
            + pattern
            + ">>>. Captured so far: <<<"
            + captured.toString(java.nio.charset.StandardCharsets.UTF_8).substring(consumedBytes)
            + ">>>");
  }

  /** Liste tous les fichiers {@code .uciscript} dans {@code dir}. */
  static List<Path> listScripts(Path dir) throws IOException {
    List<Path> scripts = new ArrayList<>();
    try (var stream = Files.list(dir)) {
      stream
          .filter(p -> p.getFileName().toString().endsWith(".uciscript"))
          .sorted()
          .forEach(scripts::add);
    }
    return scripts;
  }
}
