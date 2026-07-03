package org.nanozero.nn.internal;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Parser .npz minimal (cf. SPEC §6.1, §13 phase 2). Lit un fichier .npz produit par {@code
 * numpy.savez_compressed} (ou {@code numpy.savez}) en pur JDK, sans dépendance externe.
 *
 * <p>Format du fichier (documenté par NumPy) :
 *
 * <ul>
 *   <li>ZIP standard (deflate ou stored) contenant des fichiers {@code <tensor_name>.npy}.
 *   <li>Chaque .npy commence par les magic bytes {@code \x93NUMPY}, suivis de la version (2
 *       octets), de la longueur du header (uint16 LE en v1, uint32 LE en v2), du header
 *       (dictionnaire ASCII Python literal), puis des données binaires.
 * </ul>
 *
 * <p>Cette phase NE valide PAS la sémantique du contenu (présence des tenseurs §6.3, header {@code
 * _meta_*}, hash). C'est l'affaire de phase 8 ({@code NetworkLoader}). Le présent parser extrait
 * toutes les entrées qu'il trouve et les classe par dtype.
 *
 * <p>Dtypes supportés :
 *
 * <ul>
 *   <li>{@code <f4} : float32 little-endian, exposé via {@link NpzData#floatTensors()}
 *   <li>{@code <i8} : int64 little-endian, exposé via {@link NpzData#int64Tensors()}
 *   <li>{@code <U*} : Unicode UTF-32 LE strings (scalaires uniquement), via {@link
 *       NpzData#stringScalars()}
 *   <li>{@code |S*} : ASCII byte strings (scalaires uniquement), via {@link
 *       NpzData#stringScalars()}
 * </ul>
 *
 * <p>Tout autre dtype, ou {@code fortran_order: True}, fait échouer la lecture avec une {@link
 * IllegalArgumentException} explicite.
 *
 * <p><strong>Visibilité</strong> : SPEC §12.3 prescrit "package-private" pour le sub-package {@code
 * internal/}. Java ne permet pas d'accès cross-package package-private au sein d'un même module
 * sans {@code module-info.java}. Cette classe est donc publique (class + méthode {@link #read}) à
 * des fins d'usage interne uniquement par {@link org.nanozero.nn.NetworkLoader} (phase 8). Note
 * pour amendement §12.3 : à reformuler en "interne au module, non exporté" quand le module
 * descriptor sera ajouté. Aucune garantie de stabilité hors du module.
 */
public final class NpzReader {

  private NpzReader() {
    throw new AssertionError("Non-instantiable");
  }

  /** Magic bytes du format .npy : {@code \x93NUMPY}. */
  private static final byte[] NPY_MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

  /** Charset Unicode UTF-32 little-endian (utilisé par numpy pour les strings {@code <U*}). */
  private static final Charset UTF_32_LE = Charset.forName("UTF-32LE");

  /** Pattern d'extraction de la valeur entre quotes : {@code 'descr': '<f4'}. */
  private static final Pattern DESCR_PATTERN = Pattern.compile("'descr':\\s*'([^']+)'");

  /** Pattern d'extraction du flag : {@code 'fortran_order': False}. */
  private static final Pattern FORTRAN_PATTERN =
      Pattern.compile("'fortran_order':\\s*(True|False)");

  /** Pattern d'extraction du tuple shape : {@code 'shape': (96, 119, 3, 3)}. */
  private static final Pattern SHAPE_PATTERN = Pattern.compile("'shape':\\s*\\(([^)]*)\\)");

  // ---------------------------------------------------------------------------------------------
  // Structure de retour
  // ---------------------------------------------------------------------------------------------

  /**
   * Résultat de la lecture d'un .npz. Toutes les maps utilisent le nom du tenseur comme clé
   * (extension {@code .npy} retirée). {@link #tensorShapes()} contient l'entrée pour TOUS les
   * tenseurs lus, indépendamment de leur dtype, pour permettre au consommateur de connaître les
   * dimensions de chaque tenseur sans connaître son type.
   */
  public record NpzData(
      Map<String, float[]> floatTensors,
      Map<String, long[]> int64Tensors,
      Map<String, String> stringScalars,
      Map<String, int[]> tensorShapes) {}

  // ---------------------------------------------------------------------------------------------
  // API publique du sub-package
  // ---------------------------------------------------------------------------------------------

  /**
   * Lit un fichier .npz et retourne ses tenseurs classés par dtype.
   *
   * @param path chemin vers le fichier .npz
   * @return données extraites
   * @throws IOException si la lecture échoue (fichier inexistant, ZIP corrompu, etc.)
   * @throws IllegalArgumentException si un .npy contient un dtype non supporté ou un format
   *     malformé
   */
  public static NpzData read(Path path) throws IOException {
    Map<String, float[]> floats = new HashMap<>();
    Map<String, long[]> int64s = new HashMap<>();
    Map<String, String> strings = new HashMap<>();
    Map<String, int[]> shapes = new HashMap<>();

    // ZipFile (random access) plutôt que ZipInputStream : valide la présence de l'end-of-central
    // -directory et lève ZipException pour les fichiers tronqués / non-zip / vides. ZipInputStream
    // se contente de scanner les local file headers et serait trop permissif pour notre besoin de
    // rejet explicite des .npz malformés.
    try (ZipFile zip = new ZipFile(path.toFile())) {
      var entries = zip.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        String entryName = entry.getName();
        if (!entryName.endsWith(".npy")) {
          continue;
        }
        String tensorName = entryName.substring(0, entryName.length() - 4);
        byte[] payload;
        try (InputStream is = zip.getInputStream(entry)) {
          payload = is.readAllBytes();
        }
        parseNpy(tensorName, payload, floats, int64s, strings, shapes);
      }
    }
    return new NpzData(floats, int64s, strings, shapes);
  }

  // ---------------------------------------------------------------------------------------------
  // Parsing .npy
  // ---------------------------------------------------------------------------------------------

  private static void parseNpy(
      String tensorName,
      byte[] raw,
      Map<String, float[]> floats,
      Map<String, long[]> int64s,
      Map<String, String> strings,
      Map<String, int[]> shapes) {
    // 1. Magic bytes + version.
    if (raw.length < 10 || !startsWith(raw, NPY_MAGIC)) {
      throw new IllegalArgumentException("Magic .npy invalide pour " + tensorName);
    }
    int major = raw[6] & 0xFF;
    int minor = raw[7] & 0xFF;
    int headerStart;
    int headerLen;
    if (major == 1) {
      headerLen = (raw[8] & 0xFF) | ((raw[9] & 0xFF) << 8);
      headerStart = 10;
    } else if (major == 2) {
      if (raw.length < 12) {
        throw new IllegalArgumentException("Header .npy v2 tronqué pour " + tensorName);
      }
      headerLen =
          (raw[8] & 0xFF)
              | ((raw[9] & 0xFF) << 8)
              | ((raw[10] & 0xFF) << 16)
              | ((raw[11] & 0xFF) << 24);
      headerStart = 12;
    } else {
      throw new IllegalArgumentException(
          "Version .npy non supportée pour " + tensorName + " : " + major + "." + minor);
    }
    int dataStart = headerStart + headerLen;
    if (raw.length < dataStart) {
      throw new IllegalArgumentException("Header .npy tronqué pour " + tensorName);
    }
    String header = new String(raw, headerStart, headerLen, StandardCharsets.US_ASCII);

    // 2. Parse les 3 champs (descr, fortran_order, shape).
    String descr = matchOne(header, DESCR_PATTERN, tensorName, "descr");
    String fortran = matchOne(header, FORTRAN_PATTERN, tensorName, "fortran_order");
    String shapeContent = matchOne(header, SHAPE_PATTERN, tensorName, "shape");

    if ("True".equals(fortran)) {
      throw new IllegalArgumentException(
          "fortran_order = True non supporté (NCHW row-major requis) pour " + tensorName);
    }
    int[] shape = parseShape(shapeContent);
    shapes.put(tensorName, shape);

    long elementCount = 1;
    for (int d : shape) {
      elementCount *= d;
    }
    if (elementCount > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "Tenseur trop grand pour Java int[] : "
              + tensorName
              + " ("
              + elementCount
              + " éléments)");
    }
    int count = (int) elementCount;

    ByteBuffer dataBuf =
        ByteBuffer.wrap(raw, dataStart, raw.length - dataStart).order(ByteOrder.LITTLE_ENDIAN);

    if ("<f4".equals(descr)) {
      float[] arr = new float[count];
      for (int i = 0; i < count; i++) {
        arr[i] = dataBuf.getFloat();
      }
      floats.put(tensorName, arr);
    } else if ("<i8".equals(descr)) {
      long[] arr = new long[count];
      for (int i = 0; i < count; i++) {
        arr[i] = dataBuf.getLong();
      }
      int64s.put(tensorName, arr);
    } else if (descr.startsWith("<U")) {
      requireScalar(tensorName, shape, descr);
      int charCount = parseStringWidth(descr.substring(2), tensorName);
      byte[] strBytes = new byte[charCount * 4];
      dataBuf.get(strBytes);
      strings.put(tensorName, trimNulls(new String(strBytes, UTF_32_LE)));
    } else if (descr.startsWith("|S")) {
      requireScalar(tensorName, shape, descr);
      int byteCount = parseStringWidth(descr.substring(2), tensorName);
      byte[] strBytes = new byte[byteCount];
      dataBuf.get(strBytes);
      int len = byteCount;
      while (len > 0 && strBytes[len - 1] == 0) {
        len--;
      }
      strings.put(tensorName, new String(strBytes, 0, len, StandardCharsets.US_ASCII));
    } else {
      throw new IllegalArgumentException("Dtype non supporté pour " + tensorName + " : " + descr);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------------------

  private static boolean startsWith(byte[] raw, byte[] prefix) {
    if (raw.length < prefix.length) {
      return false;
    }
    for (int i = 0; i < prefix.length; i++) {
      if (raw[i] != prefix[i]) {
        return false;
      }
    }
    return true;
  }

  private static String matchOne(String header, Pattern pattern, String tensorName, String field) {
    Matcher m = pattern.matcher(header);
    if (!m.find()) {
      throw new IllegalArgumentException(
          "Champ '" + field + "' absent du header .npy pour " + tensorName);
    }
    return m.group(1);
  }

  private static int[] parseShape(String content) {
    String trimmed = content.trim();
    if (trimmed.isEmpty()) {
      return new int[0];
    }
    String[] parts = trimmed.split(",");
    int n = 0;
    for (String p : parts) {
      if (!p.trim().isEmpty()) {
        n++;
      }
    }
    int[] out = new int[n];
    int idx = 0;
    for (String p : parts) {
      String t = p.trim();
      if (!t.isEmpty()) {
        out[idx++] = Integer.parseInt(t);
      }
    }
    return out;
  }

  private static int parseStringWidth(String s, String tensorName) {
    try {
      return Integer.parseInt(s);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(
          "Largeur de string invalide dans le dtype pour " + tensorName + " : " + s, e);
    }
  }

  private static void requireScalar(String tensorName, int[] shape, String descr) {
    if (shape.length != 0) {
      throw new IllegalArgumentException(
          "Tenseur de strings non scalaire non supporté ("
              + tensorName
              + ", dtype="
              + descr
              + ", shape="
              + java.util.Arrays.toString(shape)
              + ")");
    }
  }

  /** Retire les codepoints null de fin (numpy pad les strings <U* à largeur fixe avec \0). */
  private static String trimNulls(String s) {
    int end = s.length();
    while (end > 0 && s.charAt(end - 1) == '\0') {
      end--;
    }
    return s.substring(0, end);
  }
}
