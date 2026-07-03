package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class JsonTest {

  // ---------------------------------------------------------------------------------------
  // Writer
  // ---------------------------------------------------------------------------------------

  @Test
  @DisplayName("write null")
  void writeNull() {
    assertThat(Json.write(null)).isEqualTo("null");
  }

  @Test
  @DisplayName("write booleans")
  void writeBooleans() {
    assertThat(Json.write(true)).isEqualTo("true");
    assertThat(Json.write(false)).isEqualTo("false");
  }

  @Test
  @DisplayName("write integers and doubles")
  void writeNumbers() {
    assertThat(Json.write(42)).isEqualTo("42");
    assertThat(Json.write(-7L)).isEqualTo("-7");
    assertThat(Json.write(3.14)).isEqualTo("3.14");
  }

  @Test
  @DisplayName("write Float and Short via generic Number branch")
  void writeGenericNumbers() {
    assertThat(Json.write(1.5f)).isEqualTo("1.5");
    assertThat(Json.write((short) 9)).isEqualTo("9");
  }

  @Test
  @DisplayName("write strings with escapes")
  void writeStringsWithEscapes() {
    assertThat(Json.write("hello")).isEqualTo("\"hello\"");
    assertThat(Json.write("a\"b")).isEqualTo("\"a\\\"b\"");
    assertThat(Json.write("line1\nline2")).isEqualTo("\"line1\\nline2\"");
    assertThat(Json.write("tab\there")).isEqualTo("\"tab\\there\"");
  }

  @Test
  @DisplayName("write escapes control chars and remaining special characters")
  void writeControlChars() {
    String ctrl = String.valueOf((char) 0x01);
    assertThat(Json.write(ctrl)).isEqualTo("\"\\u0001\"");
    assertThat(Json.write("a\rb")).isEqualTo("\"a\\rb\"");
    assertThat(Json.write("a\bb")).isEqualTo("\"a\\bb\"");
    assertThat(Json.write("a\fb")).isEqualTo("\"a\\fb\"");
    assertThat(Json.write("a\\b")).isEqualTo("\"a\\\\b\"");
  }

  @Test
  @DisplayName("write empty and populated objects")
  void writeObjects() {
    assertThat(Json.write(Map.of())).isEqualTo("{}");

    // LinkedHashMap to guarantee insertion order in the output.
    Map<String, Object> obj = new LinkedHashMap<>();
    obj.put("a", 1);
    obj.put("b", "two");
    assertThat(Json.write(obj)).isEqualTo("{\"a\":1,\"b\":\"two\"}");
  }

  @Test
  @DisplayName("write empty and populated arrays")
  void writeArrays() {
    assertThat(Json.write(List.of())).isEqualTo("[]");
    assertThat(Json.write(List.of(1, 2, 3))).isEqualTo("[1,2,3]");
    assertThat(Json.write(List.of("a", "b"))).isEqualTo("[\"a\",\"b\"]");
  }

  @Test
  @DisplayName("write nested object/array combinations")
  void writeNested() {
    Map<String, Object> root = new LinkedHashMap<>();
    root.put("status", "ok");
    root.put("counts", List.of(1, 2, 3));
    Map<String, Object> nested = new LinkedHashMap<>();
    nested.put("inner", true);
    root.put("nested", nested);
    assertThat(Json.write(root))
        .isEqualTo("{\"status\":\"ok\",\"counts\":[1,2,3],\"nested\":{\"inner\":true}}");
  }

  @Test
  @DisplayName("write rejects NaN/Infinity")
  void writeRejectsNaN() {
    assertThatThrownBy(() -> Json.write(Double.NaN)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.write(Double.POSITIVE_INFINITY))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.write(Float.NEGATIVE_INFINITY))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("write rejects unsupported types")
  void writeRejectsUnsupportedType() {
    assertThatThrownBy(() -> Json.write(new Object()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported JSON type");
  }

  @Test
  @DisplayName("write rejects non-String object keys")
  void writeRejectsNonStringKeys() {
    Map<Object, Object> bad = new LinkedHashMap<>();
    bad.put(1, "v");
    assertThatThrownBy(() -> Json.write(bad))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("keys must be String");
  }

  // ---------------------------------------------------------------------------------------
  // Parser
  // ---------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse scalars")
  void parseScalars() {
    assertThat(Json.parse("null")).isNull();
    assertThat(Json.parse("true")).isEqualTo(Boolean.TRUE);
    assertThat(Json.parse("false")).isEqualTo(Boolean.FALSE);
    assertThat(Json.parse("42")).isEqualTo(42L);
    assertThat(Json.parse("-3.5")).isEqualTo(-3.5);
    assertThat(Json.parse("\"hello\"")).isEqualTo("hello");
  }

  @Test
  @DisplayName("parse numbers : negatives, exponents, decimals")
  void parseNumberVariants() {
    assertThat(Json.parse("-17")).isEqualTo(-17L);
    assertThat(Json.parse("0")).isEqualTo(0L);
    assertThat(Json.parse("1e3")).isEqualTo(1000.0);
    assertThat(Json.parse("1.5E2")).isEqualTo(150.0);
    assertThat(Json.parse("-2.5e-1")).isEqualTo(-0.25);
  }

  @Test
  @DisplayName("parse empty + populated object")
  void parseObject() {
    assertThat(Json.parseObject("{}")).isEmpty();
    assertThat(Json.parseObject("{\"a\":1,\"b\":\"x\"}"))
        .containsEntry("a", 1L)
        .containsEntry("b", "x");
  }

  @Test
  @DisplayName("parse handles whitespace around tokens")
  void parseWhitespace() {
    assertThat(Json.parseObject("  {  \"a\" : 1 , \"b\" : [ 2 , 3 ] }  ")).containsEntry("a", 1L);
  }

  @Test
  @DisplayName("parse nested arrays + objects")
  void parseNested() {
    Map<String, Object> m = Json.parseObject("{\"games\":[{\"id\":1},{\"id\":2}],\"total\":2}");
    assertThat(m.get("total")).isEqualTo(2L);
    @SuppressWarnings("unchecked")
    List<Map<String, Object>> games = (List<Map<String, Object>>) m.get("games");
    assertThat(games).hasSize(2);
    assertThat(games.get(0).get("id")).isEqualTo(1L);
    assertThat(games.get(1).get("id")).isEqualTo(2L);
  }

  @Test
  @DisplayName("parse empty array")
  void parseEmptyArray() {
    assertThat(Json.parse("[]")).isEqualTo(List.of());
  }

  @Test
  @DisplayName("parse string escapes including unicode, slash, and all named escapes")
  void parseStringEscapes() {
    assertThat(Json.parse("\"a\\\"b\"")).isEqualTo("a\"b");
    assertThat(Json.parse("\"a\\nb\"")).isEqualTo("a\nb");
    assertThat(Json.parse("\"\\u0041BC\"")).isEqualTo("ABC");
    assertThat(Json.parse("\"a\\/b\"")).isEqualTo("a/b");
    assertThat(Json.parse("\"a\\\\b\"")).isEqualTo("a\\b");
    assertThat(Json.parse("\"\\r\\t\\b\\f\"")).isEqualTo("\r\t\b\f");
  }

  @Test
  @DisplayName("parse rejects unknown escape")
  void parseRejectsUnknownEscape() {
    assertThatThrownBy(() -> Json.parse("\"a\\xb\""))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unknown escape");
  }

  @Test
  @DisplayName("parse rejects truncated unicode escape")
  void parseRejectsTruncatedUnicode() {
    assertThatThrownBy(() -> Json.parse("\"\\u00\"")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("parse rejects bad escape at end of input")
  void parseRejectsBadEscapeAtEnd() {
    assertThatThrownBy(() -> Json.parse("\"abc\\")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("parse rejects unterminated string")
  void parseRejectsUnterminatedString() {
    assertThatThrownBy(() -> Json.parse("\"abc"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unterminated");
  }

  @Test
  @DisplayName("parse rejects trailing content")
  void parseRejectsTrailing() {
    assertThatThrownBy(() -> Json.parse("42 garbage"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Trailing content");
  }

  @Test
  @DisplayName("parse rejects empty input")
  void parseRejectsEmpty() {
    assertThatThrownBy(() -> Json.parse(""))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unexpected end of input");
  }

  @Test
  @DisplayName("parse rejects malformed JSON")
  void parseRejectsMalformed() {
    assertThatThrownBy(() -> Json.parse("{")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("[1,2")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("{\"a\" 1}")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("{\"a\":1 \"b\":2}"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Expected ',' or '}'");
    assertThatThrownBy(() -> Json.parse("[1 2]"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Expected ',' or ']'");
  }

  @Test
  @DisplayName("parse rejects bad literals")
  void parseRejectsBadLiterals() {
    assertThatThrownBy(() -> Json.parse("tru")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("nul")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("parseObject rejects non-object top-level")
  void parseObjectRejectsNonObject() {
    assertThatThrownBy(() -> Json.parseObject("[1,2,3]"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Expected JSON object");
  }

  // ---------------------------------------------------------------------------------------
  // Round-trip (mirror of the wire DTOs the worker actually sends/receives)
  // ---------------------------------------------------------------------------------------

  @Test
  @DisplayName("round-trip a JobClaim-shaped payload")
  void roundTripJobClaim() {
    Map<String, Object> claim = new LinkedHashMap<>();
    claim.put("job_id", "abc-123");
    claim.put("model_version", 7);
    claim.put("opening_fen", null);
    claim.put("dirichlet_seed", 42);
    claim.put("num_sims", 200);

    String wire = Json.write(claim);
    Map<String, Object> back = Json.parseObject(wire);
    assertThat(back.get("job_id")).isEqualTo("abc-123");
    assertThat(back.get("model_version")).isEqualTo(7L);
    assertThat(back.get("opening_fen")).isNull();
    assertThat(back.get("dirichlet_seed")).isEqualTo(42L);
    assertThat(back.get("num_sims")).isEqualTo(200L);
  }

  @Test
  @DisplayName("round-trip a SubmitRequest with base64 blobs")
  void roundTripSubmitRequest() {
    Map<String, Object> pos = new LinkedHashMap<>();
    pos.put("ply", 0);
    pos.put("fen", "startpos");
    pos.put("input_planes_b64", "AAECAwQ=");
    pos.put("policy_target_b64", "/v8=");
    pos.put("outcome", 0.5);

    Map<String, Object> body = new LinkedHashMap<>();
    body.put("game_id", "uuid-1");
    body.put("model_version", 1);
    body.put("positions", List.of(pos));

    String wire = Json.write(body);
    Map<String, Object> back = Json.parseObject(wire);
    @SuppressWarnings("unchecked")
    List<Map<String, Object>> positions = (List<Map<String, Object>>) back.get("positions");
    assertThat(positions.get(0).get("input_planes_b64")).isEqualTo("AAECAwQ=");
    assertThat(positions.get(0).get("outcome")).isEqualTo(0.5);
  }
}
