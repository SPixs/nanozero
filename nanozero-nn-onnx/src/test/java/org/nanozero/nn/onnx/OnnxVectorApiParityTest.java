package org.nanozero.nn.onnx;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.nanozero.nn.NNOutput;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * Phase 2a Step B : parité numérique OnnxNetwork (ONNX Runtime Java) vs
 * Network (Vector API SIMD CPU current).
 *
 * <p>Workflow :
 *
 * <ol>
 *   <li>Load {@code deployment_model.onnx} (BN-folded) via OnnxNetwork
 *   <li>Load {@code deployment_model.npz} (mêmes poids, BN-folded) via NetworkLoader
 *   <li>Same input random 7616 floats
 *   <li>Run both backends, compare outputs
 *   <li>Tolerance 1e-3 (Java SIMD vs ONNX Runtime cross-impl, marge confortable)
 * </ol>
 *
 * <p>Si test PASS : ONNX path est drop-in compatible avec Vector API actuel.
 * Permet de switcher backend NN sans changer .npz training pipeline.
 */
class OnnxVectorApiParityTest {

    private static final Path RESOURCES = Paths.get("src/test/resources");
    private static final Path ONNX_MODEL = RESOURCES.resolve("deployment_model.onnx");
    private static final Path NPZ_MODEL = RESOURCES.resolve("deployment_model.npz");
    private static final Path REF_BOARD = RESOURCES.resolve("deployment_reference_board.bin");

    // Tolerance plus large que Phase 1 : SIMD CPU Java vs ONNX Runtime CPU
    // peuvent diverger sur l'ordre d'accumulation des conv/matmul (float32 non
    // associatif). Empiriquement attendu < 1e-3.
    private static final float TOLERANCE = 1e-3f;

    @Test
    void onnxAndVectorApiProduceSameOutputs() throws Exception {
        if (!Files.exists(ONNX_MODEL) || !Files.exists(NPZ_MODEL)) {
            throw new AssertionError(
                "Fixtures absentes — run d'abord: python scripts/phase2a_export_bn_folded_onnx.py");
        }

        float[] inputBoard = loadFloat32LE(REF_BOARD);

        // ---- Backend 1 : ONNX Runtime Java ----
        OnnxNetwork.Result onnxResult;
        try (OnnxNetwork net = new OnnxNetwork(ONNX_MODEL)) {
            onnxResult = net.forwardSingle(inputBoard);
        }
        assertNotNull(onnxResult);
        float[] onnxPolicy = onnxResult.policyLogits();
        float onnxValue = onnxResult.value();

        // ---- Backend 2 : Vector API SIMD via NetworkLoader ----
        Network vectorApi = NetworkLoader.load(NPZ_MODEL);
        // Network.forward expects MAX_BATCH * 119 * 64 = 487424 flat buffer
        int planesLen = Network.MAX_BATCH * 119 * 64;
        float[] planesFull = new float[planesLen];
        System.arraycopy(inputBoard, 0, planesFull, 0, inputBoard.length);
        NNOutput out = new NNOutput();
        vectorApi.forward(planesFull, 1, out);

        float[] vaPolicy = out.logitsOf(0);  // copy de 4672 logits
        float vaValue = out.valueOf(0);

        // ---- Compare ----
        float maxPolicyDiff = maxAbsDiff(onnxPolicy, vaPolicy);
        float valueDiff = Math.abs(onnxValue - vaValue);

        System.out.printf(
            "Parity OnnxNetwork vs Network (Vector API SIMD)%n"
            + "  Policy max abs diff: %.6e (tolerance %.0e)%n"
            + "  Value      abs diff: %.6e (tolerance %.0e)%n"
            + "  ONNX value         : %.6f%n"
            + "  Vector API value   : %.6f%n",
            maxPolicyDiff, TOLERANCE, valueDiff, TOLERANCE, onnxValue, vaValue);

        assertTrue(maxPolicyDiff < TOLERANCE,
            "Policy diff " + maxPolicyDiff + " >= tolerance " + TOLERANCE);
        assertTrue(valueDiff < TOLERANCE,
            "Value diff " + valueDiff + " >= tolerance " + TOLERANCE);
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
