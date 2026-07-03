package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.engine.EngineConfig;

/**
 * Tests {@link PUCTSelector} (cf. SPEC §5.2, §12 phase 3).
 *
 * <p>Critère §12 phase 3 : 1000 cas random vs implémentation oracle scalaire indépendante, 0
 * divergence. L'oracle est volontairement écrit dans un style différent (boucle if/else explicite,
 * accumulation séparée) pour qu'une erreur de formule ait peu de chances d'apparaître
 * symétriquement dans les deux versions.
 */
class PUCTSelectorTest {

  // -------------------------------------------------------------------------------------------
  // Oracle scalaire indépendant (formule §5.2 réécrite explicitement)
  // -------------------------------------------------------------------------------------------

  /**
   * Implémentation oracle de {@link PUCTSelector#argmax}. Volontairement plus verbeuse pour
   * minimiser le risque que la même erreur de formule contamine les deux versions.
   */
  private static int oracle(Node node, EngineConfig config) {
    int n = node.childMoves.length;
    double sqrtN = Math.sqrt((double) node.totalVisits.get());
    double bestScore = Double.NEGATIVE_INFINITY;
    int bestIdx = 0;
    for (int i = 0; i < n; i++) {
      Node ch = node.children[i];
      double q;
      double nVisits;
      if (ch == null) {
        q = config.fpuValue();
        nVisits = 0;
      } else if (ch.totalVisits.get() == 0) {
        q = config.fpuValue();
        nVisits = 0;
      } else {
        // Convention zero-sum cf. SPEC §5.4 amendé : child.totalValueSum stocké du POV du child,
        // négation pour obtenir Q du POV du parent (= côté à jouer à `node`).
        q = -(double) ch.totalValueSum.get() / (double) ch.totalVisits.get();
        nVisits = ch.totalVisits.get();
      }
      double explorationNumerator = (double) config.cPuct() * (double) node.childPriors[i] * sqrtN;
      double u = explorationNumerator / (1.0 + nVisits);
      double score = q + u;
      // Comparaison stricte pour tie-breaking par premier index (cohérent avec §5.2 + ADR-010).
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // -------------------------------------------------------------------------------------------
  // Helpers : construction de Node manipulés à la main (pas de NN, pas de SearcherCore)
  // -------------------------------------------------------------------------------------------

  private static Node createExpandedNode(int[] moves, float[] priors, int parentVisits) {
    Node n = new Node();
    n.expanded = true;
    n.childMoves = moves.clone();
    n.childPriors = priors.clone();
    n.children = new Node[moves.length];
    n.totalVisits.set(parentVisits);
    return n;
  }

  private static Node createVisitedChild(Node parent, int childIdx, int visits, float valueSum) {
    Node ch = new Node(parent, parent.childMoves[childIdx]);
    ch.totalVisits.set(visits);
    ch.totalValueSum.set(valueSum);
    parent.children[childIdx] = ch;
    return ch;
  }

  // -------------------------------------------------------------------------------------------
  // Edge cases
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Single child : argmax retourne toujours 0 (seul choix)")
  void singleChildAlwaysReturnsZero() {
    Node node = createExpandedNode(new int[] {0xAB}, new float[] {1.0f}, 5);
    EngineConfig cfg = EngineConfig.defaults();
    assertThat(PUCTSelector.argmax(node, cfg)).isZero();
  }

  @Test
  @DisplayName("Fresh expansion (totalVisits=0) : tous les U=0, argmax = 0 par tie-breaking")
  void freshExpansionReturnsFirstIndex() {
    Node node =
        createExpandedNode(
            new int[] {0xA, 0xB, 0xC}, new float[] {0.1f, 0.5f, 0.4f}, /* parentVisits */ 0);
    EngineConfig cfg = EngineConfig.defaults();
    // Tous children null. sqrtN = 0 => U = 0. Tous Q = fpuValue = 0.0f.
    // Tous scores égaux à 0.0f, argmax = 0 (premier index).
    assertThat(PUCTSelector.argmax(node, cfg)).isZero();
    assertThat(oracle(node, cfg)).isZero();
  }

  @Test
  @DisplayName("One visited child + 2 null : exploration domine pour les non-visités")
  void oneVisitedChildExplorationDominates() {
    Node node =
        createExpandedNode(
            new int[] {0xA, 0xB, 0xC}, new float[] {0.1f, 0.5f, 0.4f}, /* parentVisits */ 10);
    // children[0] visité 5 fois, Q = 4.0/5 = 0.8
    createVisitedChild(node, 0, 5, 4.0f);
    EngineConfig cfg = EngineConfig.defaults(); // cPuct=2.5, fpuValue=0.0

    // sqrtN = sqrt(10) ≈ 3.162
    // score[0] = 0.8 + 2.5 * 0.1 * 3.162 / 6 ≈ 0.932
    // score[1] = 0.0 + 2.5 * 0.5 * 3.162 / 1 ≈ 3.953
    // score[2] = 0.0 + 2.5 * 0.4 * 3.162 / 1 ≈ 3.162
    // argmax = 1.
    assertThat(PUCTSelector.argmax(node, cfg)).isEqualTo(1);
    assertThat(oracle(node, cfg)).isEqualTo(1);
  }

  @Test
  @DisplayName(
      "Exploitation domine quand cPuct est petit et le child visité a un Q élevé pour le parent")
  void exploitationWinsWhenCPuctSmall() {
    Node node =
        createExpandedNode(new int[] {0xA, 0xB}, new float[] {0.5f, 0.5f}, /* parentVisits */ 50);
    // Convention zero-sum (cf. SPEC §5.4 amendé) : un Q élevé POUR LE PARENT correspond à un
    // child.totalValueSum NÉGATIF (le child est dans une mauvaise position de SON POV).
    // Ici children[0] visité 50 fois avec totalValueSum = -45 (Q child POV = -0.9 ; Q parent POV
    // = +0.9 = bon coup pour le parent).
    createVisitedChild(node, 0, 50, -45.0f);
    // children[1] null → Q parent POV = fpuValue = 0

    EngineConfig small = new EngineConfig(0.1f, 0.0f, 1024, 0.0f, 0.0f, 0L);
    // sqrtN = sqrt(50) ≈ 7.07
    // score[0] = -(-0.9) + 0.1 * 0.5 * 7.07 / 51 ≈ 0.907
    // score[1] = 0.0 + 0.1 * 0.5 * 7.07 / 1 ≈ 0.354
    // argmax = 0 (exploitation : le parent veut le coup qui rend le child mauvais pour le child).
    assertThat(PUCTSelector.argmax(node, small)).isZero();
    assertThat(oracle(node, small)).isZero();
  }

  @Test
  @DisplayName("Tie-breaking : 2 scores strictement égaux → premier index l'emporte")
  void tieBreakingFirstIndexWins() {
    Node node =
        createExpandedNode(new int[] {0xA, 0xB}, new float[] {0.5f, 0.5f}, /* parentVisits */ 4);
    // 2 children null, priors identiques, sqrtN = 2.
    // U[0] = U[1] = cPuct * 0.5 * 2 / 1 (identique).
    // Q[0] = Q[1] = fpuValue.
    // scores identiques → argmax = 0 (premier index).
    EngineConfig cfg = EngineConfig.defaults();
    assertThat(PUCTSelector.argmax(node, cfg)).isZero();
    assertThat(oracle(node, cfg)).isZero();
  }

  @Test
  @DisplayName("Child instancié mais visits=0 : traité comme null (Q=fpu, pas de NaN)")
  void instantiatedChildWithZeroVisits() {
    Node node =
        createExpandedNode(new int[] {0xA, 0xB}, new float[] {0.5f, 0.5f}, /* parentVisits */ 4);
    // children[0] instancié mais totalVisits=0 (transition lazy / pré-backup).
    Node child = new Node(node, node.childMoves[0]);
    node.children[0] = child;
    // Pas de createVisitedChild ici : visits restent 0.
    // children[1] null.

    EngineConfig cfg = EngineConfig.defaults();
    int picked = PUCTSelector.argmax(node, cfg);
    // Aucun NaN attendu. Tie attendu (tous les Q=fpuValue, tous les U identiques).
    assertThat(picked).isZero();
    assertThat(oracle(node, cfg)).isZero();
  }

  @Test
  @DisplayName("Q négatif : un autre child non-visité (Q=fpuValue=0) peut le supplanter")
  void negativeQMakesUnvisitedChildPreferred() {
    Node node =
        createExpandedNode(new int[] {0xA, 0xB}, new float[] {0.5f, 0.5f}, /* parentVisits */ 5);
    // children[0] visité 3 fois avec valueSum = -2.4, Q = -0.8.
    createVisitedChild(node, 0, 3, -2.4f);
    // children[1] null, Q = fpuValue = 0.0.

    EngineConfig cfg = EngineConfig.defaults();
    // sqrtN = sqrt(5) ≈ 2.236
    // score[0] = -0.8 + 2.5 * 0.5 * 2.236 / 4 ≈ -0.8 + 0.699 = -0.101
    // score[1] = 0.0 + 2.5 * 0.5 * 2.236 / 1 ≈ 2.795
    // argmax = 1.
    assertThat(PUCTSelector.argmax(node, cfg)).isEqualTo(1);
    assertThat(oracle(node, cfg)).isEqualTo(1);
  }

  @Test
  @DisplayName("Prior écrasant (0.96 vs 0.01 × 4) sur tous null : argmax = index du gros prior")
  void largePriorDominates() {
    Node node =
        createExpandedNode(
            new int[] {0xA, 0xB, 0xC, 0xD, 0xE},
            new float[] {0.01f, 0.01f, 0.96f, 0.01f, 0.01f},
            /* parentVisits */ 4);
    EngineConfig cfg = EngineConfig.defaults();
    assertThat(PUCTSelector.argmax(node, cfg)).isEqualTo(2);
    assertThat(oracle(node, cfg)).isEqualTo(2);
  }

  @Test
  @DisplayName("FPU value > 0 augmente le score des children non-visités (sanity)")
  void fpuValueAffectsUnvisitedChildren() {
    Node node =
        createExpandedNode(new int[] {0xA, 0xB}, new float[] {0.1f, 0.5f}, /* parentVisits */ 10);
    createVisitedChild(node, 0, 4, 2.0f); // Q = 0.5

    // FPU = 0 : score[1] = 0 + cPuct*0.5*sqrt(10)/1
    //          score[0] = 0.5 + cPuct*0.1*sqrt(10)/5
    EngineConfig fpuZero = new EngineConfig(1.0f, 0.0f, 1024, 0.0f, 0.0f, 0L);
    int pickedZero = PUCTSelector.argmax(node, fpuZero);
    assertThat(pickedZero).isEqualTo(oracle(node, fpuZero));

    // FPU = 0.6 : score[1] = 0.6 + cPuct*0.5*sqrt(10)/1 (gros boost de score[1]).
    //            score[0] inchangé.
    EngineConfig fpuHigh = new EngineConfig(1.0f, 0.6f, 1024, 0.0f, 0.0f, 0L);
    int pickedHigh = PUCTSelector.argmax(node, fpuHigh);
    assertThat(pickedHigh).isEqualTo(oracle(node, fpuHigh));
  }

  // -------------------------------------------------------------------------------------------
  // Critère §12 phase 3 : 1000 cas random vs oracle (0 divergence)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Critère §12 phase 3 : 1000 cas random vs oracle, 0 divergence")
  void thousandRandomCasesVsOracle() {
    Random rng = new Random(0xC0FFEE12L);
    int divergences = 0;
    int firstDivergenceCase = -1;
    for (int trial = 0; trial < 1000; trial++) {
      int n = 1 + rng.nextInt(30); // [1, 30] enfants
      int[] moves = new int[n];
      for (int i = 0; i < n; i++) {
        moves[i] = rng.nextInt(0xFFFF) + 1; // moves quelconques (non utilisés par l'algo)
      }
      // Priors random + normalisés (sum = 1).
      float[] priors = new float[n];
      double sum = 0;
      for (int i = 0; i < n; i++) {
        priors[i] = rng.nextFloat();
        sum += priors[i];
      }
      for (int i = 0; i < n; i++) {
        priors[i] = (float) (priors[i] / sum);
      }
      int parentVisits = rng.nextInt(1000); // peut être 0
      Node node = createExpandedNode(moves, priors, parentVisits);

      // Fraction d'enfants visités (uniforme [0, 1]).
      double visitedFraction = rng.nextDouble();
      for (int i = 0; i < n; i++) {
        if (rng.nextDouble() < visitedFraction) {
          int childVisits = 1 + rng.nextInt(parentVisits + 1);
          // Q ∈ [-1, 1] => valueSum = childVisits * Q
          float q = (rng.nextFloat() * 2f) - 1f;
          createVisitedChild(node, i, childVisits, childVisits * q);
        }
      }

      float cPuct = 0.5f + rng.nextFloat() * 4.5f; // [0.5, 5.0]
      float fpu = (rng.nextFloat() * 1f) - 0.5f; // [-0.5, 0.5]
      EngineConfig cfg = new EngineConfig(cPuct, fpu, 1024, 0.0f, 0.0f, 0L);

      int impl = PUCTSelector.argmax(node, cfg);
      int ref = oracle(node, cfg);
      if (impl != ref) {
        divergences++;
        if (firstDivergenceCase < 0) {
          firstDivergenceCase = trial;
        }
      }
    }
    assertThat(divergences)
        .as(
            "PUCTSelector vs oracle : %d/1000 divergences ; première au cas %d",
            divergences, firstDivergenceCase)
        .isZero();
  }

  // -------------------------------------------------------------------------------------------
  // Validation : utility class non instanciable
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Constructeur privé (utility class), instanciation par réflexion lève AssertionError")
  void utilityClassNotInstantiable() throws NoSuchMethodException {
    Constructor<PUCTSelector> ctor = PUCTSelector.class.getDeclaredConstructor();
    assertThat(Modifier.isPrivate(ctor.getModifiers())).isTrue();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance)
        .isInstanceOf(InvocationTargetException.class)
        .cause()
        .isInstanceOf(AssertionError.class);
  }

  // -------------------------------------------------------------------------------------------
  // v1.1.0 — Injection Dirichlet au root (cf. SPEC §5.2 amendement, ADR-012)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("dirichletNoise == null → priors purs NN (comportement v1.0.0 strictement préservé)")
  void argmax_dirichletNoiseNull_usesPureNNPriors() {
    // Position arbitraire : 3 coups, priors [0.5, 0.3, 0.2], jamais visités, fpu=0
    Node node = createExpandedNode(new int[] {1, 2, 3}, new float[] {0.5f, 0.3f, 0.2f}, 10);
    // Pas de noise
    assertThat(node.dirichletNoise).isNull();
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 0L);
    int idxWithoutNoise = PUCTSelector.argmax(node, cfg);
    // Oracle = formule pure (mêmes priors car noise null → epsilon ignorée)
    assertThat(idxWithoutNoise).isEqualTo(oracle(node, cfg));
    // Coup 0 le plus prior → choisi car tous fpu=0 et sqrtN constant : argmax sur P
    assertThat(idxWithoutNoise).isZero();
  }

  @Test
  @DisplayName("dirichletNoise non-null + epsilon=0.25 → P_eff = 0.75*P_nn + 0.25*noise")
  void argmax_dirichletNoiseNonNull_mixesPriors() {
    // 3 coups, P_nn = [0.7, 0.2, 0.1] : sans noise, le coup 0 domine.
    // Noise [0.0, 0.0, 1.0] : avec epsilon=0.25, P_eff = [0.525, 0.15, 0.325]
    // → coup 0 reste argmax mais l'écart se réduit. On vérifie le mix sur un cas plus extrême.
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.5f, 0.3f, 0.2f}, 1);
    node.dirichletNoise = new float[] {0.0f, 0.0f, 1.0f};
    // P_eff[0] = 0.75 * 0.5 + 0.25 * 0.0 = 0.375
    // P_eff[1] = 0.75 * 0.3 + 0.25 * 0.0 = 0.225
    // P_eff[2] = 0.75 * 0.2 + 0.25 * 1.0 = 0.4 → ARGMAX
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 0L);
    int picked = PUCTSelector.argmax(node, cfg);
    assertThat(picked).isEqualTo(2); // noise booste coup 2 au-dessus de coup 0
  }

  @Test
  @DisplayName("dirichletNoise non-null + epsilon=0 → priors purs NN (cas pathologique)")
  void argmax_dirichletEpsilonZeroWithNonNullNoise_noEffect() {
    // Cas pathologique : noise présent mais epsilon=0 (peut arriver si Dirichlet activé puis
    // désactivé à mi-recherche, ou bug). La formule (1-0)*P + 0*noise = P doit donner v1.0.0.
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.5f, 0.3f, 0.2f}, 1);
    node.dirichletNoise = new float[] {0.0f, 0.0f, 1.0f};
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L);
    // Doit être strictement équivalent au comportement noise=null car epsilon=0 annule le mix.
    int picked = PUCTSelector.argmax(node, cfg);
    Node nodeNoNoise =
        createExpandedNode(new int[] {10, 20, 30}, new float[] {0.5f, 0.3f, 0.2f}, 1);
    int pickedNoNoise = PUCTSelector.argmax(nodeNoNoise, cfg);
    assertThat(picked).isEqualTo(pickedNoNoise);
    assertThat(picked).isZero(); // P_nn[0] = 0.5 domine
  }

  @Test
  @DisplayName("dirichletNoise non-null + epsilon=1.0 → priors purs noise (extrême)")
  void argmax_dirichletEpsilonOneWithNoise_fullNoise() {
    // Cas extrême : epsilon=1.0 → P_eff = noise complètement.
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.5f, 0.3f, 0.2f}, 1);
    node.dirichletNoise = new float[] {0.0f, 0.0f, 1.0f};
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 1.0f, 0L);
    // P_eff = [0, 0, 1] → coup 2 doit gagner même si P_nn favorise coup 0.
    int picked = PUCTSelector.argmax(node, cfg);
    assertThat(picked).isEqualTo(2);
  }

  // -------------------------------------------------------------------------------------------
  // v1.2.0 — Virtual loss (cf. ADR-013, SPEC §15.3)
  // -------------------------------------------------------------------------------------------

  /** Config v1.2.0 avec virtual loss configurable, autres champs au défaut. */
  private static EngineConfig configWithVloss(float vloss) {
    return new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, vloss);
  }

  @Test
  @DisplayName("v1.2.0 régression : childInFlight=null + vloss=0 → comportement v1.1.2 strict")
  void vlossRegressionNullInFlightAndZeroVloss() {
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.4f, 0.3f, 0.3f}, 100);
    createVisitedChild(node, 0, 50, -25.0f);
    createVisitedChild(node, 1, 30, 0.0f);
    createVisitedChild(node, 2, 20, 5.0f);
    EngineConfig cfg = EngineConfig.defaults(); // vloss=0
    assertThat(node.childInFlight).as("default node has null childInFlight").isNull();
    int picked = PUCTSelector.argmax(node, cfg);
    int oraclePicked = oracle(node, cfg);
    assertThat(picked)
        .as("vloss=0 + childInFlight=null doit donner identique oracle v1.1.2")
        .isEqualTo(oraclePicked);
  }

  @Test
  @DisplayName(
      "v1.2.0 régression : childInFlight=null + vloss>0 → vloss skipé (pas d'inFlight à pénaliser)")
  void vlossRegressionVlossActiveButNullInFlight() {
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.4f, 0.3f, 0.3f}, 100);
    createVisitedChild(node, 0, 50, -25.0f);
    createVisitedChild(node, 1, 30, 0.0f);
    createVisitedChild(node, 2, 20, 5.0f);
    EngineConfig cfg = configWithVloss(3.0f); // vloss=3 mais childInFlight=null
    // Pas d'allocation de childInFlight → vloss ignoré → comportement strict v1.1.2.
    int picked = PUCTSelector.argmax(node, cfg);
    int oraclePicked = oracle(node, configWithVloss(0.0f)); // oracle ignore vloss
    assertThat(picked)
        .as("vloss ne s'applique que si childInFlight non-null")
        .isEqualTo(oraclePicked);
  }

  @Test
  @DisplayName(
      "v1.2.0 régression : childInFlight alloué + tous inFlight[i]=0 → comportement v1.1.2 strict")
  void vlossRegressionAllocatedButAllZero() {
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.4f, 0.3f, 0.3f}, 100);
    createVisitedChild(node, 0, 50, -25.0f);
    createVisitedChild(node, 1, 30, 0.0f);
    createVisitedChild(node, 2, 20, 5.0f);
    node.ensureChildInFlightAllocated();
    EngineConfig cfg = configWithVloss(3.0f);
    // Tous inFlight[i] = 0 par défaut → vlossN = 3 * 0 = 0 → équivalent v1.1.2.
    int picked = PUCTSelector.argmax(node, cfg);
    int oraclePicked = oracle(node, configWithVloss(0.0f));
    assertThat(picked).isEqualTo(oraclePicked);
  }

  @Test
  @DisplayName(
      "v1.2.0 : 2 children identiques, child 0 in-flight → PUCT préfère child 1 (vloss diverge)")
  void vlossSteersAwayFromInFlightChild() {
    // 2 children identiques en stats : N=10, W=0 → Q POV parent = 0 partout, U identique
    // → tie-break favorise child[0]. Si on met inFlight[0]=1 avec vloss=3 → child[0] pénalisé,
    // child[1] doit être choisi.
    Node node = createExpandedNode(new int[] {10, 20}, new float[] {0.5f, 0.5f}, 20);
    createVisitedChild(node, 0, 10, 0.0f);
    createVisitedChild(node, 1, 10, 0.0f);
    node.ensureChildInFlightAllocated();
    EngineConfig cfg = configWithVloss(3.0f);
    // Sans vloss, tie-break = child[0].
    assertThat(PUCTSelector.argmax(node, cfg)).isZero();
    // Avec inFlight[0] = 1 → child[0] pénalisé, child[1] gagne.
    node.childInFlight.incrementAndGet(0);
    int picked = PUCTSelector.argmax(node, cfg);
    assertThat(picked).as("PUCT doit éviter le chemin in-flight").isEqualTo(1);
  }

  @Test
  @DisplayName(
      "v1.2.0 : child non-visité + in-flight → Q_eff = -1 (très mauvais POV parent), évité")
  void vlossUnvisitedInFlightYieldsWorstQ() {
    // child[0] : non-visité (N=0, W=0) mais in-flight. Avec vloss=3 + inFlight=1 :
    //   effN = 0 + 3*1 = 3 ; wRaw + vlossN = 0 + 3 = 3 ; qSa = -3/3 = -1.0
    // child[1] : non-visité, pas in-flight → qSa = fpu (= 0.0f par défaut).
    // U identique → child[1] gagne (qSa=0 > -1).
    Node node = createExpandedNode(new int[] {10, 20}, new float[] {0.5f, 0.5f}, 0);
    node.ensureChildInFlightAllocated();
    node.childInFlight.set(0, 1);
    EngineConfig cfg = configWithVloss(3.0f);
    int picked = PUCTSelector.argmax(node, cfg);
    assertThat(picked).isEqualTo(1);
  }

  @Test
  @DisplayName("v1.2.0 : inFlight multiple amplifie la pénalité (vloss cumulé)")
  void vlossPenaltyScalesWithInFlight() {
    // child[0] : N=10, W=0, inFlight=5. effN = 10 + 3*5 = 25 ; wEff = 0 + 15 = 15.
    //   qSa = -15/25 = -0.6
    // child[1] : N=10, W=0, inFlight=0. qSa = -0/10 = 0.0
    // child[1] gagne nettement.
    Node node = createExpandedNode(new int[] {10, 20}, new float[] {0.5f, 0.5f}, 20);
    createVisitedChild(node, 0, 10, 0.0f);
    createVisitedChild(node, 1, 10, 0.0f);
    node.ensureChildInFlightAllocated();
    node.childInFlight.set(0, 5); // 5 threads in-flight sur child 0
    EngineConfig cfg = configWithVloss(3.0f);
    assertThat(PUCTSelector.argmax(node, cfg)).isEqualTo(1);
  }

  @Test
  @DisplayName("v1.2.0 : ensureChildInFlightAllocated idempotent")
  void ensureChildInFlightIdempotent() {
    Node node = createExpandedNode(new int[] {10, 20, 30}, new float[] {0.5f, 0.3f, 0.2f}, 0);
    node.ensureChildInFlightAllocated();
    java.util.concurrent.atomic.AtomicIntegerArray first = node.childInFlight;
    assertThat(first).isNotNull();
    assertThat(first.length()).isEqualTo(3);
    // Re-appel : même instance, pas de re-alloc.
    node.ensureChildInFlightAllocated();
    assertThat(node.childInFlight).isSameAs(first);
  }

  @Test
  @DisplayName("v1.2.0 : ensureChildInFlightAllocated avant expand → IllegalStateException")
  void ensureChildInFlightBeforeExpandRejected() {
    Node node = new Node();
    // childMoves == null (non expandé)
    assertThatThrownBy(node::ensureChildInFlightAllocated)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("childMoves");
  }
}
