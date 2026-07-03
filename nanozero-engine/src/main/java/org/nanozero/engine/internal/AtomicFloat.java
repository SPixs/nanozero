package org.nanozero.engine.internal;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Accumulateur {@code float} atomique lock-free, basé sur un {@link AtomicInteger} sur les bits
 * IEEE-754 (cf. ADR-014, §15.5).
 *
 * <p>Le JDK ne fournit pas d'{@code AtomicFloat} natif. Cette classe encapsule un {@link
 * AtomicInteger} et utilise {@link Float#floatToRawIntBits(float)} ↔ {@link
 * Float#intBitsToFloat(int)} pour traiter le float comme un entier 32 bits. Le CAS sur l'entier
 * garantit l'atomicité de la mise à jour.
 *
 * <p>Sémantique de {@code atomicAdd} : CAS-loop qui retry tant que la lecture diverge de l'écriture
 * finale. Sous contention faible (N ≤ 8 threads concurrents sur le même Node, cas typique MCTS
 * multi-thread), une seule itération suffit en moyenne ; sous très forte contention, les retries
 * dégradent le throughput mais préservent la correction.
 *
 * <p>Garanties memory model :
 *
 * <ul>
 *   <li>{@link #get()} : volatile read (sequential consistency).
 *   <li>{@link #set(float)} : volatile write.
 *   <li>{@link #atomicAdd(float)} : compareAndSet jusqu'au succès, équivalent à un CAS
 *       sequentially-consistent.
 * </ul>
 *
 * <p>Particularités IEEE-754 importantes :
 *
 * <ul>
 *   <li>{@code Float.floatToRawIntBits(NaN)} retourne <em>une</em> représentation parmi plusieurs
 *       NaNs possibles (chacun a un bit pattern distinct). Si {@code NaN} est stocké, le retrieval
 *       est cohérent mais l'égalité de bits n'est PAS l'égalité {@code float} (NaN ≠ NaN).
 *   <li>{@code +0.0f} et {@code -0.0f} ont des représentations bits différentes (0x00000000 vs
 *       0x80000000), mais sont égaux comme floats. Les opérations sont cohérentes.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class AtomicFloat {

  private final AtomicInteger bits;

  /**
   * Construit un {@code AtomicFloat} initialisé à la valeur fournie.
   *
   * @param initial valeur initiale (peut être {@code NaN}, {@code ±Infinity}, denormal, etc.)
   */
  public AtomicFloat(float initial) {
    this.bits = new AtomicInteger(Float.floatToRawIntBits(initial));
  }

  /** Construit un {@code AtomicFloat} initialisé à {@code 0.0f}. */
  public AtomicFloat() {
    this(0.0f);
  }

  /**
   * Lecture atomique de la valeur courante.
   *
   * @return la valeur {@code float} actuelle (volatile read)
   */
  public float get() {
    return Float.intBitsToFloat(bits.get());
  }

  /**
   * Écriture atomique (volatile) de la valeur. Remplace la valeur courante par {@code value} sans
   * tenir compte de la valeur précédente.
   *
   * @param value nouvelle valeur
   */
  public void set(float value) {
    bits.set(Float.floatToRawIntBits(value));
  }

  /**
   * Ajoute atomiquement {@code delta} et retourne la nouvelle valeur. Utilise une CAS-loop pour
   * garantir qu'aucun update concurrent n'est perdu.
   *
   * <p>Sous contention élevée, plusieurs retries peuvent survenir. En l'absence de contention,
   * équivalent en perf à un {@code AtomicInteger.addAndGet} (∼1 LOCK XADD x86).
   *
   * @param delta valeur à ajouter (peut être négative ; NaN/Infinity propagent comme attendu pour
   *     l'arithmétique IEEE-754)
   * @return la nouvelle valeur après ajout
   */
  public float atomicAdd(float delta) {
    while (true) {
      int currentBits = bits.get();
      float currentValue = Float.intBitsToFloat(currentBits);
      float newValue = currentValue + delta;
      int newBits = Float.floatToRawIntBits(newValue);
      if (bits.compareAndSet(currentBits, newBits)) {
        return newValue;
      }
    }
  }

  /**
   * CAS conditionnel sur les bits float. Met à jour de {@code expected} à {@code update} uniquement
   * si la valeur courante (en bits raw) est identique à {@code expected}.
   *
   * <p>Note : la comparaison est sur les bits raw, donc {@code compareAndSet(NaN, x)} ne retournera
   * vrai que si la représentation bits du NaN stocké est exactement la même que celle de {@code
   * Float.floatToRawIntBits(NaN)}.
   *
   * @param expected valeur attendue
   * @param update valeur à écrire si la valeur courante est égale (en bits) à {@code expected}
   * @return {@code true} si le CAS a réussi
   */
  public boolean compareAndSet(float expected, float update) {
    return bits.compareAndSet(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(update));
  }

  @Override
  public String toString() {
    return Float.toString(get());
  }
}
