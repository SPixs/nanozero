package org.nanozero.uci.internal;

import java.io.PrintStream;
import java.util.Objects;

/**
 * Writer des {@link UciResponse} vers stdout (cf. SPEC §3.2, §5.6, §7.2, §12 phase 4).
 *
 * <p>Discipline stricte (cf. SPEC §7.2) :
 *
 * <ul>
 *   <li><strong>Émission complète par ligne</strong> : la ligne UCI est construite intégralement en
 *       mémoire via {@link StringBuilder}, puis émise en un seul appel {@code out.print(line +
 *       "\n")}. Pas d'écritures partielles, garantit qu'aucun interleaving avec un autre thread
 *       écrivant sur stdout (typiquement {@code InfoReporter} en phase 6) ne peut produire de torn
 *       lines.
 *   <li><strong>LF strict</strong> : le séparateur de ligne est explicitement {@code "\n"} (pas
 *       {@code System.lineSeparator()} qui serait {@code "\r\n"} sur Windows). Cohérent avec la
 *       canonique UCI multi-plateforme.
 *   <li><strong>Auto-flush responsabilité du caller</strong> : le constructor accepte un {@link
 *       PrintStream} tel quel. Le caller (typiquement {@code UciMain}) DOIT fournir un {@code
 *       PrintStream} auto-flush (ex. {@code new PrintStream(System.out, true,
 *       StandardCharsets.UTF_8)}). Le {@code \n} émis par {@code print} déclenche alors le flush
 *       automatiquement.
 * </ul>
 *
 * <p>La méthode statique {@link #format} est exposée en visibilité package-private pour permettre
 * les tests unitaires directs sans capture stdout.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciResponseWriter {

  private final PrintStream out;

  /**
   * Construit un writer sur le {@link PrintStream} fourni.
   *
   * @param out PrintStream destinataire, non null. <strong>DOIT</strong> être configuré auto-flush
   *     par le caller (cf. classe Javadoc).
   * @throws NullPointerException si {@code out} est null
   */
  public UciResponseWriter(PrintStream out) {
    this.out = Objects.requireNonNull(out, "out must not be null");
  }

  /**
   * Émet la réponse UCI sur stdout en une seule ligne complète. Le caller garantit qu'aucun thread
   * ne fait d'autre écriture concurrente sur le même {@link PrintStream} hors de {@code emit} (cf.
   * SPEC §7.2).
   *
   * <p>Pour {@link UciResponse.Id}, deux lignes UCI distinctes sont émises ({@code id name ...} et
   * {@code id author ...}) via un séparateur {@code "\n"} interne au format ; le {@code "\n"} final
   * est ajouté par {@code emit} à l'écriture.
   *
   * @param response réponse à émettre, non null
   * @throws NullPointerException si {@code response} est null
   */
  public void emit(UciResponse response) {
    Objects.requireNonNull(response, "response must not be null");
    out.print(format(response) + "\n");
    // (v1.2.1) flush explicite — autoflush du PrintStream peut échouer sur certaines
    // configurations stdout (pipe Linux, fastchess) où buffer interne ne se vide pas
    // après le \n. Belt-and-suspenders pour compatibilité Linux SPRT.
    out.flush();
  }

  /**
   * Formate une {@link UciResponse} en chaîne UCI conforme (sans le {@code \n} terminal, ajouté par
   * {@link #emit}). Pour {@link UciResponse.Id}, retourne une chaîne contenant un {@code "\n"}
   * interne séparant les deux lignes UCI conventionnelles.
   *
   * <p>Visibilité package-private pour permettre les tests unitaires sans capture stdout.
   *
   * @param response réponse à formater, non null
   * @return chaîne UCI formatée
   */
  static String format(UciResponse response) {
    return switch (response) {
      case UciResponse.Id id -> formatId(id);
      case UciResponse.UciOk ignored -> "uciok";
      case UciResponse.Option opt -> formatOption(opt);
      case UciResponse.ReadyOk ignored -> "readyok";
      case UciResponse.Info info -> formatInfo(info);
      case UciResponse.InfoString is -> formatInfoString(is);
      case UciResponse.BestMove bm -> formatBestMove(bm);
    };
  }

  /**
   * (v1.2.0) {@code info string <text>} — émission non-standard tolérée par tous les GUIs UCI (cf.
   * SPEC §6.5 amendée v1.2.0, ADR-003 mise à jour v1.2.0). Texte vide produit {@code "info string
   * "} (avec espace trailing, conforme UCI), utile pour le cas position terminale (visits sans
   * pairs).
   */
  private static String formatInfoString(UciResponse.InfoString is) {
    return "info string " + is.text();
  }

  // -------------------------------------------------------------------------------------------
  // Formatters par type de UciResponse
  // -------------------------------------------------------------------------------------------

  /**
   * {@code id name X\nid author Y} — deux lignes UCI conventionnelles séparées par un {@code "\n"}
   * interne. Le {@code "\n"} terminal est ajouté par {@link #emit}.
   */
  private static String formatId(UciResponse.Id id) {
    return "id name " + id.name() + "\n" + "id author " + id.author();
  }

  /**
   * {@code option name X type Y default Z [min A max B]}. Dispatch sur {@link UciOption} sealed
   * (Check / Spin / String_) via switch exhaustif.
   */
  private static String formatOption(UciResponse.Option opt) {
    UciOption o = opt.option();
    StringBuilder sb = new StringBuilder("option name ").append(o.name());
    switch (o) {
      case UciOption.Check c -> sb.append(" type check default ").append(c.defaultValue());
      case UciOption.Spin s ->
          sb.append(" type spin")
              .append(" default ")
              .append(s.defaultValue())
              .append(" min ")
              .append(s.min())
              .append(" max ")
              .append(s.max());
      case UciOption.String_ s -> sb.append(" type string default ").append(s.defaultValue());
    }
    return sb.toString();
  }

  /** {@code bestmove <m> [ponder <m>]} via {@link UciMoveCodec#encode}. */
  private static String formatBestMove(UciResponse.BestMove bm) {
    StringBuilder sb = new StringBuilder("bestmove ").append(UciMoveCodec.encode(bm.move()));
    if (bm.ponderMove().isPresent()) {
      sb.append(" ponder ").append(UciMoveCodec.encode(bm.ponderMove().getAsInt()));
    }
    return sb.toString();
  }

  /**
   * {@code info ...} — champs émis dans l'ordre conventionnel UCI : {@code depth seldepth multipv
   * score nodes nps time pv string}. Seuls les champs présents (Optional non-empty ou pv non vide)
   * sont émis. Si aucun champ n'est présent, retourne {@code "info"} tout court (légal UCI bien que
   * peu utile).
   *
   * <p>Convention {@code score} : si les deux {@code scoreMate} et {@code scoreCp} sont présents
   * (cas dégénéré, peu probable en pratique), {@code score mate} prime sur {@code score cp} —
   * mate-en-N est l'information la plus précise, cp est une approximation.
   */
  private static String formatInfo(UciResponse.Info info) {
    InfoFields f = info.fields();
    StringBuilder sb = new StringBuilder("info");
    if (f.depth().isPresent()) {
      sb.append(" depth ").append(f.depth().getAsInt());
    }
    if (f.seldepth().isPresent()) {
      sb.append(" seldepth ").append(f.seldepth().getAsInt());
    }
    if (f.multipv().isPresent()) {
      sb.append(" multipv ").append(f.multipv().getAsInt());
    }
    if (f.scoreMate().isPresent()) {
      sb.append(" score mate ").append(f.scoreMate().getAsInt());
    } else if (f.scoreCp().isPresent()) {
      sb.append(" score cp ").append(f.scoreCp().getAsInt());
    }
    if (f.nodes().isPresent()) {
      sb.append(" nodes ").append(f.nodes().getAsInt());
    }
    if (f.nps().isPresent()) {
      sb.append(" nps ").append(f.nps().getAsInt());
    }
    if (f.timeMs().isPresent()) {
      sb.append(" time ").append(f.timeMs().getAsLong());
    }
    int[] pv = f.pv();
    if (pv.length > 0) {
      sb.append(" pv");
      for (int m : pv) {
        sb.append(' ').append(UciMoveCodec.encode(m));
      }
    }
    if (f.string().isPresent()) {
      sb.append(" string ").append(f.string().get());
    }
    return sb.toString();
  }
}
