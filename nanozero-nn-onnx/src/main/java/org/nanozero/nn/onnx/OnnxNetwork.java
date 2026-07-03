package org.nanozero.nn.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Phase 1 PoC : NN AlphaZero inference via ONNX Runtime Java.
 *
 * <p>API minimale pour valider la viabilité technique du backend ONNX :
 *
 * <ul>
 *   <li>Charge un model ONNX exporté depuis PyTorch via {@code scripts/phase1_export_onnx.py}.
 *   <li>Single-batch forward pass: float[119*8*8] → policy float[4672] + value float
 *   <li>CPU EP par défaut (Phase 1 DevSrv). CUDA EP / DirectML EP en Phase 3.
 * </ul>
 *
 * <p>Volontairement <em>indépendant</em> du module {@code nanozero-nn} actuel :
 * pas d'implémentation de l'interface {@code Network} (qui est une final class, pas
 * une interface), pas de dépendance sur le format {@code .npz} BN-folded. Le drop-in
 * replacement pour le UCI engine = Phase 2 (refactor Network interface) ou Phase 3
 * (adapter ONNX export pour BN-folded weights).
 *
 * <p>Thread safety : OrtSession est thread-safe pour run() concurrent ; OrtEnvironment
 * est singleton. AutoCloseable pour libération des resources natives.
 */
public final class OnnxNetwork implements AutoCloseable {

    /** AlphaZero input format : 119 plans 8x8 par position. */
    public static final int INPUT_CHANNELS = 119;
    public static final int BOARD_SIZE = 8;
    public static final int INPUT_FLOATS_PER_POS = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

    /** AlphaZero policy format : 73 plans 8x8 = 4672 logits par position. */
    public static final int POLICY_LOGITS_PER_POS = 73 * BOARD_SIZE * BOARD_SIZE;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String policyOutputName;
    private final String valueOutputName;

    /**
     * Charge un model ONNX depuis le filesystem avec CPU EP (default).
     *
     * @param onnxPath chemin vers le fichier .onnx
     * @throws OrtException si le model ne peut pas être chargé
     */
    public OnnxNetwork(Path onnxPath) throws OrtException {
        this(onnxPath, false);
    }

    /**
     * Charge un model ONNX avec choix de l'execution provider.
     *
     * @param onnxPath chemin vers le fichier .onnx
     * @param useCuda  si {@code true}, ajoute CUDA EP (NVIDIA GPU) ; CPU EP reste fallback.
     *                 Requiert que le module soit buildé avec {@code onnxruntime_gpu} (Phase 3b).
     * @throws OrtException si CUDA n'est pas dispo ou le model ne peut pas être chargé
     */
    public OnnxNetwork(Path onnxPath, boolean useCuda) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        if (useCuda) {
            // CUDA EP avec GPU device 0. CPU EP reste auto-fallback.
            opts.addCUDA(0);
        }
        this.session = env.createSession(onnxPath.toString(), opts);

        // Discover input/output names dynamically (PoC robustness)
        this.inputName = session.getInputNames().iterator().next();
        var outputNames = session.getOutputNames().iterator();
        this.policyOutputName = outputNames.next();
        this.valueOutputName = outputNames.next();
    }

    /**
     * Run forward pass sur 1 position. Pas batched (Phase 1 PoC simple).
     *
     * @param boardPlanes input flat shape [119 * 8 * 8] = 7616 floats, row-major NCHW
     * @return Result.policy float[4672] + Result.value float
     * @throws OrtException si l'inference échoue
     */
    public Result forwardSingle(float[] boardPlanes) throws OrtException {
        if (boardPlanes.length != INPUT_FLOATS_PER_POS) {
            throw new IllegalArgumentException(
                "boardPlanes length must be " + INPUT_FLOATS_PER_POS + ", got " + boardPlanes.length);
        }

        // Shape [1, 119, 8, 8]
        long[] shape = {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(boardPlanes), shape)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);
            try (OrtSession.Result results = session.run(inputs)) {
                // policy: float[1][4672]
                float[][] policyOut = (float[][]) results.get(policyOutputName).orElseThrow().getValue();
                // value: float[1]  (shape [batch] après squeeze côté PyTorch)
                float[] valueOut = (float[]) results.get(valueOutputName).orElseThrow().getValue();
                return new Result(policyOut[0].clone(), valueOut[0]);
            }
        }
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        // env est singleton, ne pas close
    }

    /** Output : policy logits (4672 floats) + scalar value (tanh-bounded [-1, 1]). */
    public record Result(float[] policyLogits, float value) {}
}
