package org.nanozero.nn.onnx;

import ai.onnxruntime.OrtException;
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
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Phase 3b : Benchmark ORT GPU (CUDA EP) vs CPU EP vs Vector API SIMD.
 *
 * <p>Test activé seulement si la system property {@code nano.onnx.gpu=true} (pas dispo
 * par défaut sur DevSrv sans GPU NVIDIA dédié). Sur W3090 RTX 3090 :
 *
 * <pre>{@code
 *   JAVA_HOME=$HOME/jdk-25 mvn -Pgpu test \
 *       -Dtest=GpuBenchmarkTest -Dnano.onnx.gpu=true
 * }</pre>
 *
 * <p>Le profile {@code gpu} dans pom switch la dep onnxruntime → onnxruntime_gpu
 * (bundle CUDA EP). Sans ce profile, l'addCUDA() échouera à charger les natives CUDA.
 *
 * <p>Compare 3 backends side-by-side avec mêmes données :
 *
 * <ol>
 *   <li>ORT GPU CUDA EP (RTX 3090)
 *   <li>ORT CPU EP (CPU host)
 *   <li>Java Vector API SIMD (CPU host, baseline actuel)
 * </ol>
 */
@EnabledIfSystemProperty(named = "nano.onnx.gpu", matches = "true")
class GpuBenchmarkTest {

    private static final int WARMUP_ITERATIONS = 50;
    private static final int MEASURE_ITERATIONS = 300;

    private static final Path RESOURCES = Paths.get("src/test/resources");
    private static final Path ONNX_MODEL = RESOURCES.resolve("deployment_model.onnx");
    private static final Path NPZ_MODEL = RESOURCES.resolve("deployment_model.npz");
    private static final Path REF_BOARD = RESOURCES.resolve("deployment_reference_board.bin");

    @Test
    void benchGpuCpuVectorApiThreeway() throws Exception {
        if (!Files.exists(ONNX_MODEL)) {
            throw new AssertionError(
                "Fixtures absentes — run d'abord: python scripts/phase2a_export_bn_folded_onnx.py");
        }

        float[] inputBoard = loadFloat32LE(REF_BOARD);

        // Setup backends
        OnnxNetwork ortGpu;
        try {
            ortGpu = new OnnxNetwork(ONNX_MODEL, /* useCuda */ true);
        } catch (OrtException e) {
            throw new AssertionError(
                "CUDA EP not available — vérifier driver NVIDIA + dep onnxruntime_gpu (profile -Pgpu)",
                e);
        }
        OnnxNetwork ortCpu = new OnnxNetwork(ONNX_MODEL, /* useCuda */ false);

        Network va = NetworkLoader.load(NPZ_MODEL);
        int planesLen = Network.MAX_BATCH * 119 * 64;
        float[] planesFull = new float[planesLen];
        System.arraycopy(inputBoard, 0, planesFull, 0, inputBoard.length);
        NNOutput vaOut = new NNOutput();

        // Warmup all backends
        System.out.println("Warmup " + WARMUP_ITERATIONS + " iterations on 3 backends...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ortGpu.forwardSingle(inputBoard);
            ortCpu.forwardSingle(inputBoard);
            va.forward(planesFull, 1, vaOut);
        }

        // ---- Measure ORT GPU ----
        long[] gpuLatencies = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long t0 = System.nanoTime();
            ortGpu.forwardSingle(inputBoard);
            gpuLatencies[i] = System.nanoTime() - t0;
        }
        Stats gpu = computeStats(gpuLatencies);

        // ---- Measure ORT CPU ----
        long[] cpuLatencies = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long t0 = System.nanoTime();
            ortCpu.forwardSingle(inputBoard);
            cpuLatencies[i] = System.nanoTime() - t0;
        }
        Stats cpu = computeStats(cpuLatencies);

        // ---- Measure Vector API ----
        long[] vaLatencies = new long[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long t0 = System.nanoTime();
            va.forward(planesFull, 1, vaOut);
            vaLatencies[i] = System.nanoTime() - t0;
        }
        Stats vaStats = computeStats(vaLatencies);

        ortGpu.close();
        ortCpu.close();

        // ---- Report ----
        double speedupVaVsGpu = (double) vaStats.median / gpu.median;
        double speedupVaVsCpu = (double) vaStats.median / cpu.median;
        double speedupCpuVsGpu = (double) cpu.median / gpu.median;

        System.out.printf("%nThree-way Benchmark (%d iterations after %d warmup)%n",
            MEASURE_ITERATIONS, WARMUP_ITERATIONS);
        System.out.printf("  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐%n");
        System.out.printf("  │ Backend             │ Min      │ p50      │ p99      │ Max      │%n");
        System.out.printf("  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤%n");
        System.out.printf("  │ ORT GPU CUDA EP     │ %7.3fms│ %7.3fms│ %7.3fms│ %7.3fms│%n",
            gpu.min / 1e6, gpu.median / 1e6, gpu.p99 / 1e6, gpu.max / 1e6);
        System.out.printf("  │ ORT CPU EP          │ %7.3fms│ %7.3fms│ %7.3fms│ %7.3fms│%n",
            cpu.min / 1e6, cpu.median / 1e6, cpu.p99 / 1e6, cpu.max / 1e6);
        System.out.printf("  │ Vector API SIMD     │ %7.3fms│ %7.3fms│ %7.3fms│ %7.3fms│%n",
            vaStats.min / 1e6, vaStats.median / 1e6, vaStats.p99 / 1e6, vaStats.max / 1e6);
        System.out.printf("  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘%n");
        System.out.printf("  Speedup factors (p50):%n");
        System.out.printf("    Vector API → ORT CPU : %.2fx%n", speedupVaVsCpu);
        System.out.printf("    Vector API → ORT GPU : %.2fx%n", speedupVaVsGpu);
        System.out.printf("    ORT CPU  → ORT GPU   : %.2fx%n", speedupCpuVsGpu);
        System.out.printf("  Throughput (single-batch sequential):%n");
        System.out.printf("    ORT GPU      : %.1f inf/sec%n", 1e9 / gpu.median);
        System.out.printf("    ORT CPU      : %.1f inf/sec%n", 1e9 / cpu.median);
        System.out.printf("    Vector API   : %.1f inf/sec%n", 1e9 / vaStats.median);
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
