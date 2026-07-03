package org.nanozero.worker;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal hand-rolled JSON serializer + parser for the jobserver wire protocol.
 *
 * <p>Intentionally small : we only need to encode/decode the 5 DTOs of nanozero-jobserver (cf.
 * ADR-014 §Endpoints). No streaming, no big-integer support, no fancy number formats. Zero external
 * dependencies — keeps the worker JAR self-contained and easy to drop on any machine with JDK.
 *
 * <p>API:
 *
 * <ul>
 *   <li>{@link #write(Object)} — serializes a Java {@code Map}, {@code List}, {@code String},
 *       {@code Number}, {@code Boolean}, or {@code null} to a JSON string.
 *   <li>{@link #parseObject(String)} — parses a JSON object string into a {@code Map<String,
 *       Object>}.
 * </ul>
 *
 * <p>Decoded numbers are returned as {@code Long} (integer) or {@code Double} (decimal). Callers
 * cast/coerce as needed.
 */
public final class Json {

  private Json() {}

  // ---------------------------------------------------------------------------------------
  // Writer
  // ---------------------------------------------------------------------------------------

  public static String write(Object value) {
    StringBuilder out = new StringBuilder();
    writeValue(value, out);
    return out.toString();
  }

  private static void writeValue(Object v, StringBuilder out) {
    switch (v) {
      case null -> out.append("null");
      case String s -> writeString(s, out);
      case Boolean b -> out.append(b ? "true" : "false");
      case Long l -> out.append(l);
      case Integer i -> out.append(i);
      case Number n -> writeNumber(n, out);
      case Map<?, ?> m -> writeObject(m, out);
      case List<?> l -> writeArray(l, out);
      default ->
          throw new IllegalArgumentException("Unsupported JSON type: " + v.getClass().getName());
    }
  }

  private static void writeNumber(Number n, StringBuilder out) {
    double d = n.doubleValue();
    if (Double.isNaN(d) || Double.isInfinite(d)) {
      throw new IllegalArgumentException("Cannot encode NaN/Infinity in JSON: " + n);
    }
    out.append(n);
  }

  private static void writeString(String s, StringBuilder out) {
    out.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    out.append('"');
  }

  private static void writeObject(Map<?, ?> m, StringBuilder out) {
    out.append('{');
    boolean first = true;
    for (Map.Entry<?, ?> e : m.entrySet()) {
      if (!first) out.append(',');
      first = false;
      if (!(e.getKey() instanceof String)) {
        throw new IllegalArgumentException("JSON object keys must be String, got " + e.getKey());
      }
      writeString((String) e.getKey(), out);
      out.append(':');
      writeValue(e.getValue(), out);
    }
    out.append('}');
  }

  private static void writeArray(List<?> l, StringBuilder out) {
    out.append('[');
    boolean first = true;
    for (Object v : l) {
      if (!first) out.append(',');
      first = false;
      writeValue(v, out);
    }
    out.append(']');
  }

  // ---------------------------------------------------------------------------------------
  // Parser
  // ---------------------------------------------------------------------------------------

  /**
   * Parse a JSON string into Java objects. Returns {@code Map<String,Object>} for top-level
   * objects, {@code List<Object>} for arrays, or scalars (Long/Double/Boolean/String/null).
   */
  public static Object parse(String text) {
    Parser p = new Parser(text);
    p.skipWhitespace();
    Object value = p.parseValue();
    p.skipWhitespace();
    if (p.pos < p.text.length()) {
      throw new IllegalArgumentException("Trailing content at position " + p.pos);
    }
    return value;
  }

  @SuppressWarnings("unchecked")
  public static Map<String, Object> parseObject(String text) {
    Object v = parse(text);
    if (!(v instanceof Map)) {
      throw new IllegalArgumentException(
          "Expected JSON object, got " + v.getClass().getSimpleName());
    }
    return (Map<String, Object>) v;
  }

  private static final class Parser {
    final String text;
    int pos;

    Parser(String text) {
      this.text = text;
      this.pos = 0;
    }

    Object parseValue() {
      skipWhitespace();
      if (pos >= text.length()) throw error("Unexpected end of input");
      char c = text.charAt(pos);
      return switch (c) {
        case '{' -> parseObjectInternal();
        case '[' -> parseArray();
        case '"' -> parseString();
        case 't', 'f' -> parseBool();
        case 'n' -> parseNull();
        default -> parseNumber();
      };
    }

    Map<String, Object> parseObjectInternal() {
      expect('{');
      Map<String, Object> result = new LinkedHashMap<>();
      skipWhitespace();
      if (peek() == '}') {
        pos++;
        return result;
      }
      while (true) {
        skipWhitespace();
        String key = parseString();
        skipWhitespace();
        expect(':');
        Object value = parseValue();
        result.put(key, value);
        skipWhitespace();
        char next = next();
        if (next == ',') continue;
        if (next == '}') return result;
        throw error("Expected ',' or '}'");
      }
    }

    List<Object> parseArray() {
      expect('[');
      List<Object> result = new java.util.ArrayList<>();
      skipWhitespace();
      if (peek() == ']') {
        pos++;
        return result;
      }
      while (true) {
        result.add(parseValue());
        skipWhitespace();
        char next = next();
        if (next == ',') continue;
        if (next == ']') return result;
        throw error("Expected ',' or ']'");
      }
    }

    String parseString() {
      expect('"');
      StringBuilder sb = new StringBuilder();
      while (pos < text.length()) {
        char c = text.charAt(pos++);
        if (c == '"') return sb.toString();
        if (c == '\\') {
          if (pos >= text.length()) throw error("Bad escape at end");
          char esc = text.charAt(pos++);
          switch (esc) {
            case '"' -> sb.append('"');
            case '\\' -> sb.append('\\');
            case '/' -> sb.append('/');
            case 'n' -> sb.append('\n');
            case 'r' -> sb.append('\r');
            case 't' -> sb.append('\t');
            case 'b' -> sb.append('\b');
            case 'f' -> sb.append('\f');
            case 'u' -> {
              if (pos + 4 > text.length()) throw error("Bad \\u escape");
              sb.append((char) Integer.parseInt(text.substring(pos, pos + 4), 16));
              pos += 4;
            }
            default -> throw error("Unknown escape \\" + esc);
          }
        } else {
          sb.append(c);
        }
      }
      throw error("Unterminated string");
    }

    Object parseNumber() {
      int start = pos;
      if (peek() == '-') pos++;
      while (pos < text.length() && "0123456789.eE+-".indexOf(text.charAt(pos)) >= 0) pos++;
      String literal = text.substring(start, pos);
      if (literal.indexOf('.') >= 0 || literal.indexOf('e') >= 0 || literal.indexOf('E') >= 0) {
        return Double.parseDouble(literal);
      }
      return Long.parseLong(literal);
    }

    Object parseBool() {
      if (text.startsWith("true", pos)) {
        pos += 4;
        return Boolean.TRUE;
      }
      if (text.startsWith("false", pos)) {
        pos += 5;
        return Boolean.FALSE;
      }
      throw error("Expected true/false");
    }

    Object parseNull() {
      if (text.startsWith("null", pos)) {
        pos += 4;
        return null;
      }
      throw error("Expected null");
    }

    void skipWhitespace() {
      while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) pos++;
    }

    char peek() {
      return pos < text.length() ? text.charAt(pos) : '\0';
    }

    char next() {
      if (pos >= text.length()) throw error("Unexpected end of input");
      return text.charAt(pos++);
    }

    void expect(char c) {
      if (next() != c) throw error("Expected '" + c + "'");
    }

    IllegalArgumentException error(String msg) {
      return new IllegalArgumentException(msg + " at position " + pos);
    }
  }
}
