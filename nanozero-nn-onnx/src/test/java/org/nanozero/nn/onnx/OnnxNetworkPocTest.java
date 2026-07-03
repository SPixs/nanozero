package org.nanozero.nn.onnx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * Phase 1 PoC test : valide que ORT Java charge le model .onnx exporté depuis PyTorch
 * et produit des outputs numérique-équivalents au reference PyTorch.
 *
 * <p>Workflow :
 *
 * <ol>
 *   <li>Charge {@code poc_model.onnx} (généré par {@code scripts/phase1_export_onnx.py})
 *   <li>Charge {@code poc_reference_board.bin} (input float32 LE, 7616 floats)
 *   <li>Run inference via ORT Java
 *   <li>Charge {@code poc_reference_policy.bin} + {@code poc_reference_value.bin} (outputs ref)
 *   <li>Compare max abs diff < 1e-4 (tolérance noise float32 cross-impl)
 * </ol>
 *
 * <p>Si test PASS : ORT Java backend est numerically équivalent au PyTorch reference,
 * confirmant la viabilité du path pour le drop-in replacement futur.
 */
class OnnxNetworkPocTest {

    private static final Path RESOURCES =
        Paths.get("src/test/resources");
    private static final Path ONNX_MODEL = RESOURCES.resolve("poc_model.onnx");
    private static final Path REF_BOARD = RESOURCES.resolve("poc_reference_board.bin");
    private static final Path REF_POLICY = RESOURCES.resolve("poc_reference_policy.bin");
    private static final Path REF_VALUE = RESOURCES.resolve("poc_reference_value.bin");

    // Tolerance générique pour cross-implementation float32 (PyTorch → ONNX → ORT Java).
    // Le sanity check Python a montré max_diff ~5e-7 ; on prend 1e-4 marge confortable.
    private static final float TOLERANCE = 1e-4f;

    @Test
    void onnxJavaOutputMatchesPyTorchReference() throws Exception {
        // Skip si fixtures absentes (export pas run encore)
        if (!Files.exists(ONNX_MODEL)) {
            throw new AssertionError(
                "Fixture absente: " + ONNX_MODEL
                + " — run d'abord: python scripts/phase1_export_onnx.py");
        }

        float[] inputBoard = loadFloat32LE(REF_BOARD);
        float[] refPolicy = loadFloat32LE(REF_POLICY);
        float[] refValue = loadFloat32LE(REF_VALUE);

        assertEquals(OnnxNetwork.INPUT_FLOATS_PER_POS, inputBoard.length,
            "Input ref board doit avoir 7616 floats (119*8*8)");
        assertEquals(OnnxNetwork.POLICY_LOGITS_PER_POS, refPolicy.length,
            "Ref policy doit avoir 4672 floats (73*8*8)");
        assertEquals(1, refValue.length, "Ref value doit avoir 1 float scalar");

        try (OnnxNetwork net = new OnnxNetwork(ONNX_MODEL)) {
            OnnxNetwork.Result result = net.forwardSingle(inputBoard);
            assertNotNull(result, "Result not null");
            assertEquals(OnnxNetwork.POLICY_LOGITS_PER_POS, result.policyLogits().length,
                "Output policy length");

            float maxPolicyDiff = maxAbsDiff(refPolicy, result.policyLogits());
            float valueDiff = Math.abs(refValue[0] - result.value());

            System.out.printf(
                "PoC parity check (PyTorch vs ORT Java)%n"
                + "  Policy max abs diff: %.6e (tolerance %.0e)%n"
                + "  Value      abs diff: %.6e (tolerance %.0e)%n"
                + "  Ref value          : %.6f%n"
                + "  ORT Java value     : %.6f%n",
                maxPolicyDiff, TOLERANCE, valueDiff, TOLERANCE, refValue[0], result.value());

            assertTrue(maxPolicyDiff < TOLERANCE,
                "Policy diff " + maxPolicyDiff + " >= tolerance " + TOLERANCE);
            assertTrue(valueDiff < TOLERANCE,
                "Value diff " + valueDiff + " >= tolerance " + TOLERANCE);
        }
    }

    private static float[] loadFloat32LE(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        if (raw.length % 4 != 0) {
            throw new IOException("File length not multiple of 4 bytes: " + path);
        }
        int n = raw.length / 4;
        float[] out = new float[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Length mismatch: " + a.length + " vs " + b.length);
        }
        float max = 0f;
        for (int i = 0; i < a.length; i++) {
            float d = Math.abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
