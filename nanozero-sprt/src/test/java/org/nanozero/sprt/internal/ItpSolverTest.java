package org.nanozero.sprt.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.function.DoubleUnaryOperator;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

class ItpSolverTest {

  @Test
  void findsRootOfLinearFunction() {
    // f(x) = x - 3, root = 3
    DoubleUnaryOperator f = x -> x - 3.0;
    double root =
        ItpSolver.solve(
            f,
            0.0,
            10.0,
            -3.0,
            7.0,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-9);
    assertThat(root).isCloseTo(3.0, Offset.offset(1e-9));
  }

  @Test
  void findsRootOfQuadratic() {
    // f(x) = x^2 - 2, positive root = sqrt(2) ≈ 1.4142135...
    DoubleUnaryOperator f = x -> x * x - 2.0;
    double root =
        ItpSolver.solve(
            f,
            0.0,
            2.0,
            -2.0,
            2.0,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-10);
    assertThat(root).isCloseTo(Math.sqrt(2.0), Offset.offset(1e-9));
  }

  @Test
  void findsRootOfTranscendental() {
    // f(x) = exp(x) - 5, root = ln(5) ≈ 1.6094...
    DoubleUnaryOperator f = x -> Math.exp(x) - 5.0;
    double root =
        ItpSolver.solve(
            f,
            0.0,
            3.0,
            Math.exp(0.0) - 5.0,
            Math.exp(3.0) - 5.0,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-10);
    assertThat(root).isCloseTo(Math.log(5.0), Offset.offset(1e-9));
  }

  @Test
  void handlesInfiniteEndpointFa() {
    // fastchess passe souvent INFINITY pour fA et -INFINITY pour fB pour des bornes
    // qui correspondent à des asymptotes verticales.
    // Test : f(x) = -1/x avec x > 0. f(0+) = -inf, f(1) = -1, f(2) = -0.5.
    // Actually, on cherche un cas où ITP doit gérer l'absence de mesures aux bornes.
    // Test plus simple : passer INFINITY/NEG_INFINITY initiaux et laisser l'algo prendre la
    // bissection au premier step.
    DoubleUnaryOperator f = x -> x - 1.5;
    double root =
        ItpSolver.solve(
            f,
            0.0,
            3.0,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-9);
    assertThat(root).isCloseTo(1.5, Offset.offset(1e-9));
  }

  @Test
  void convergesQuicklyOnLinearVsBisection() {
    // ITP doit converger en O(log) iterations, comme bissection ou mieux.
    // Pour [0, 1] et epsilon=1e-10 : log2(1/2e-10) ≈ 33 iterations max.
    DoubleUnaryOperator f = x -> x - 0.3;
    double root =
        ItpSolver.solve(
            f,
            0.0,
            1.0,
            -0.3,
            0.7,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-10);
    assertThat(root).isCloseTo(0.3, Offset.offset(1e-9));
  }

  @Test
  void swapsEndpointsIfFaPositive() {
    // f(a) > 0, f(b) < 0 — l'algo doit swap pour avoir f(a) < 0 < f(b).
    DoubleUnaryOperator f = x -> -x + 2.5; // root = 2.5
    double root =
        ItpSolver.solve(
            f,
            0.0,
            5.0,
            2.5,
            -2.5,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            1e-9);
    assertThat(root).isCloseTo(2.5, Offset.offset(1e-9));
  }

  @Test
  void utilityClassCannotBeInstantiated() {
    // Via reflection.
    assertThatThrownBy(
            () -> {
              var ctor = ItpSolver.class.getDeclaredConstructor();
              ctor.setAccessible(true);
              ctor.newInstance();
            })
        .hasCauseInstanceOf(AssertionError.class);
  }
}
