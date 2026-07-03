package org.nanozero.nn.onnx;

import org.nanozero.nn.NNOutput;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Phase 3a : Microbenchmark single-inference CPU.
 *
 * <p>Compare latence par forward pass entre :
 *
 * <ul>
 *   <li>{@link OnnxNetwork} via ONNX Runtime Java CPU EP (backend oneDNN/MKL ?)
 *   <li>{@link Network} via Vector API SIMD (Java incubator)
 * </ul>
 *
 * <p>Workload : N forwards single-batch sur same input, warmup pour laisser JIT compiler.
 * Report mean + min + p50 + p99 + max latency. Pas un test pass/fail — toujours pass,
 * print les chiffres pour analyse.
 *
 * <p>Note : Le test mesure CPU EP. La vraie question stratégique pour le projet est
 * "ORT GPU CUDA EP gives big speedup ?". Phase 3b sur W3090 pour cela.
 */
class CpuBenchmarkTest {

    private static final int WARMUP_ITERATIONS = 50;
    private static final int MEASURE_ITERATIONS = 200;

    private static final Path RESOURCES = Paths.get("src/test/resources");
    private static final Path ONNX_MODEL = RESOURCES.resolve("deployment_model.onnx");
    private static final Path NPZ_MODEL = RESOURCES.resolve("deployment_model.npz");
    private static final Path REF_BOARD = RESOURCES.resolve("deployment_reference_board.bin");

    @Test
    void benchSingleInferenceCpu() throws Exception {
        if (!Files.exists(ONNX_MODEL)) {
            throw new AssertionError(
                "Fixtures absentes — run d'abord: python scripts/phase2a_export_bn_folded_onnx.py");
        }

        float[] inputBoard = loadFloat32LE(REF_BOARD);

        // Setup ORT
        OnnxNetwork ort = new OnnxNetwork(ONNX_MODEL);

        // Setup Vector API
        Network va = NetworkLoader.load(NPZ_MODEL);
        int planesLen = Network.MAX_BATCH * 119 * 64;
        float[] planesFull = new float[planesLen];
        System.arraycopy(inputBoard, 0, planesFull, 0, inputBoard.length);
        NNOutput vaOut = new NNOutput();

        // Warmup (JIT compile + ORT first inference cache + Vector API ThreadLocal alloc)
        System.out.println("Warmup " + WARMUP_ITERATIONS + " iterations...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ort.forwardSingle(inputBoard);
            va.forward(planesFull, 1, vaOut);
        }

        // ---- Measure ORT ----
        long[] ortLatencies = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long t0 = System.nanoTime();
            ort.forwardSingle(inputBoard);
            ortLatencies[i] = System.nanoTime() - t0;
        }
        Stats ortStats = computeStats(ortLatencies);

        // ---- Measure Vector API ----
        long[] vaLatencies = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long t0 = System.nanoTime();
            va.forward(planesFull, 1, vaOut);
            vaLatencies[i] = System.nanoTime() - t0;
        }
        Stats vaStats = computeStats(vaLatencies);

        ort.close();

        // ---- Report ----
        double ratio = (double) vaStats.median / ortStats.median;
        System.out.printf("%nCPU Single-Inference Benchmark (%d iterations after %d warmup)%n",
            MEASURE_ITERATIONS, WARMUP_ITERATIONS);
        System.out.printf("  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐%n");
        System.out.printf("  │ Backend             │ Min      │ p50      │ p99      │ Max      │%n");
        System.out.printf("  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤%n");
        System.out.printf("  │ ONNX Runtime Java   │ %7.3fms│ %7.3fms│ %7.3fms│ %7.3fms│%n",
            ortStats.min / 1e6, ortStats.median / 1e6, ortStats.p99 / 1e6, ortStats.max / 1e6);
        System.out.printf("  │ Vector API SIMD     │ %7.3fms│ %7.3fms│ %7.3fms│ %7.3fms│%n",
            vaStats.min / 1e6, vaStats.median / 1e6, vaStats.p99 / 1e6, vaStats.max / 1e6);
        System.out.printf("  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘%n");
        System.out.printf("  Ratio Vector API / ORT = %.2fx %s%n",
            ratio, ratio > 1.0 ? "(ORT faster)" : "(Vector API faster)");

        // Convert to inferences/sec for context
        double ortIps = 1e9 / ortStats.median;
        double vaIps = 1e9 / vaStats.median;
        System.out.printf("  Throughput (single-batch sequential):%n");
        System.out.printf("    ONNX Runtime : %.1f inferences/sec%n", ortIps);
        System.out.printf("    Vector API   : %.1f inferences/sec%n", vaIps);
    }

    private static Stats computeStats(long[] latencies) {
        long[] sorted = latencies.clone();
        Arrays.sort(sorted);
        Stats s = new Stats();
        s.min = sorted[0];
        s.median = sorted[sorted.length / 2];
        s.p99 = sorted[Math.min(sorted.length - 1, (int) (sorted.length * 0.99))];
        s.max = sorted[sorted.length - 1];
        return s;
    }

    private static class Stats {
        long min;
        long median;
        long p99;
        long max;
    }

    private static float[] loadFloat32LE(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        int n = raw.length / 4;
        float[] out = new float[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }
}
