package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires + stress concurrency pour {@link AtomicFloat} (cf. ADR-014, §15.5).
 *
 * <p>Le test {@code stressConcurrentIncrementsPreserveSum} est l'invariant critique : N threads
 * font M ajouts d'un même delta, la somme finale DOIT être exacte ± 1 ULP (tolérance IEEE-754
 * float32 sur l'accumulation d'un million de petites valeurs).
 */
class AtomicFloatTest {

  @Test
  @DisplayName("default constructor initialise à 0.0f")
  void defaultConstructorIsZero() {
    AtomicFloat a = new AtomicFloat();
    assertThat(a.get()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("initial value preserved")
  void initialValuePreserved() {
    AtomicFloat a = new AtomicFloat(3.14f);
    assertThat(a.get()).isEqualTo(3.14f);
  }

  @Test
  @DisplayName("set remplace la valeur")
  void setReplacesValue() {
    AtomicFloat a = new AtomicFloat(1.0f);
    a.set(2.5f);
    assertThat(a.get()).isEqualTo(2.5f);
  }

  @Test
  @DisplayName("atomicAdd returns new value (single-threaded)")
  void atomicAddReturnsNewValue() {
    AtomicFloat a = new AtomicFloat(1.0f);
    float result = a.atomicAdd(0.5f);
    assertThat(result).isEqualTo(1.5f);
    assertThat(a.get()).isEqualTo(1.5f);
  }

  @Test
  @DisplayName("atomicAdd cumule plusieurs incréments séquentiels")
  void sequentialAddsAccumulate() {
    AtomicFloat a = new AtomicFloat(0.0f);
    for (int i = 0; i < 1000; i++) {
      a.atomicAdd(1.0f);
    }
    assertThat(a.get()).isEqualTo(1000.0f);
  }

  @Test
  @DisplayName("atomicAdd avec delta négatif")
  void atomicAddNegative() {
    AtomicFloat a = new AtomicFloat(10.0f);
    a.atomicAdd(-3.0f);
    assertThat(a.get()).isEqualTo(7.0f);
  }

  @Test
  @DisplayName("compareAndSet successful match")
  void compareAndSetSuccess() {
    AtomicFloat a = new AtomicFloat(1.0f);
    assertThat(a.compareAndSet(1.0f, 2.0f)).isTrue();
    assertThat(a.get()).isEqualTo(2.0f);
  }

  @Test
  @DisplayName("compareAndSet échec si valeur ne correspond pas")
  void compareAndSetFailure() {
    AtomicFloat a = new AtomicFloat(1.0f);
    assertThat(a.compareAndSet(99.0f, 2.0f)).isFalse();
    assertThat(a.get()).isEqualTo(1.0f);
  }

  @Test
  @DisplayName("NaN stockable et lisible (bits cohérents)")
  void nanRoundtripBits() {
    AtomicFloat a = new AtomicFloat(Float.NaN);
    assertThat(Float.isNaN(a.get())).isTrue();
  }

  @Test
  @DisplayName("+Infinity stockable")
  void positiveInfinityRoundtrip() {
    AtomicFloat a = new AtomicFloat(Float.POSITIVE_INFINITY);
    assertThat(a.get()).isEqualTo(Float.POSITIVE_INFINITY);
  }

  @Test
  @DisplayName("toString retourne la représentation float standard")
  void toStringIsFloatRepr() {
    AtomicFloat a = new AtomicFloat(2.5f);
    assertThat(a.toString()).isEqualTo("2.5");
  }

  /**
   * Stress test critique (cf. ADR-014) : 4 threads × 250 000 incréments de 1.0f doivent produire
   * exactement 1 000 000 (pas de visite perdue par race).
   *
   * <p>Borne float32 : 1M en float32 est exactement représentable (entier ≤ 2^24 = 16 777 216).
   * Aucune erreur d'arrondi.
   */
  @Test
  @DisplayName("stress : 4 threads × 250 000 incréments, somme finale exactement 1 000 000")
  void stressConcurrentIncrementsPreserveSum() throws InterruptedException {
    final int nThreads = 4;
    final int iterationsPerThread = 250_000;
    AtomicFloat acc = new AtomicFloat(0.0f);

    ExecutorService pool = Executors.newFixedThreadPool(nThreads);
    CountDownLatch ready = new CountDownLatch(nThreads);
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(nThreads);
    for (int t = 0; t < nThreads; t++) {
      pool.submit(
          () -> {
            ready.countDown();
            try {
              start.await();
              for (int i = 0; i < iterationsPerThread; i++) {
                acc.atomicAdd(1.0f);
              }
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            } finally {
              done.countDown();
            }
          });
    }
    ready.await();
    start.countDown();
    boolean finished = done.await(30, TimeUnit.SECONDS);
    pool.shutdown();

    assertThat(finished).as("threads finished within timeout").isTrue();
    assertThat(acc.get())
        .as("somme atomique doit être exacte (pas de visite perdue)")
        .isEqualTo((float) (nThreads * iterationsPerThread));
  }

  /**
   * Stress test secondaire : N threads chacun ajoutant un delta différent ; la somme doit être
   * linéaire en N × delta. Variation pour stress des CAS-loops avec valeurs hétérogènes.
   */
  @Test
  @DisplayName("stress : 8 threads × 50 000 incréments de delta différents, somme exacte")
  void stressHeterogeneousDeltas() throws InterruptedException {
    final int nThreads = 8;
    final int iterationsPerThread = 50_000;
    AtomicFloat acc = new AtomicFloat(0.0f);

    ExecutorService pool = Executors.newFixedThreadPool(nThreads);
    CountDownLatch ready = new CountDownLatch(nThreads);
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(nThreads);
    float expectedSum = 0f;
    for (int t = 0; t < nThreads; t++) {
      final float delta = (t + 1) * 0.5f; // 0.5, 1.0, 1.5, ..., 4.0
      expectedSum += iterationsPerThread * delta;
      pool.submit(
          () -> {
            ready.countDown();
            try {
              start.await();
              for (int i = 0; i < iterationsPerThread; i++) {
                acc.atomicAdd(delta);
              }
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            } finally {
              done.countDown();
            }
          });
    }
    ready.await();
    start.countDown();
    boolean finished = done.await(30, TimeUnit.SECONDS);
    pool.shutdown();

    assertThat(finished).isTrue();
    // Tolérance float32 sur cumul de 400k ajouts non-entiers : 1 ULP suffit largement.
    assertThat(acc.get())
        .as("somme hétérogène doit être exacte (ou ~ 1 ULP en float32)")
        .isCloseTo(expectedSum, org.assertj.core.data.Offset.offset(0.1f));
  }
}
