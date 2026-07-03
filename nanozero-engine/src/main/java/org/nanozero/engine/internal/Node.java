package org.nanozero.engine.internal;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

/**
 * Nœud de l'arbre MCTS représentant une position d'échecs visitée pendant la recherche (cf. SPEC
 * §3.1, mise à jour v1.2.0 §15.5).
 *
 * <p>(v1.2.0) Les compteurs muables {@link #totalVisits} et {@link #totalValueSum} sont des
 * primitives atomiques (cf. ADR-014) ; {@link #expanded} est {@code volatile} pour garantir la
 * visibilité cross-thread. En mode mono-thread ({@code searchThreads=1}), le surcoût atomique est
 * négligeable (∼5-10 ns par accès vs MOV simple) et la sémantique est strictement préservée
 * bit-pour-bit.
 *
 * <p>La structure utilise une <strong>lazy expansion</strong> : un nœud enfant n'est instancié
 * comme objet {@code Node} qu'à la première fois qu'il est sélectionné via PUCT (avant cela, ses
 * informations — coup, prior — vivent uniquement dans les tableaux du parent).
 *
 * <p><strong>Invariants normatifs §3.1</strong> (maintenus par l'usage MCTS, non par validation
 * interne — {@code Node} est une struct mutable manipulée par un code de confiance) :
 *
 * <ul>
 *   <li><strong>I-Node-1</strong> : si {@code expanded == true}, alors {@link #childMoves}, {@link
 *       #childPriors}, {@link #children} sont non-null et ont la même longueur (= nombre de coups
 *       légaux depuis cette position).
 *   <li><strong>I-Node-2</strong> : si {@code expanded == false}, alors {@link #childMoves}, {@link
 *       #childPriors}, {@link #children} sont {@code null} (feuille pure).
 *   <li><strong>I-Node-3</strong> : si {@code terminal == true}, alors {@code expanded == true} et
 *       {@code childMoves.length == 0}. {@link #terminalValue} ∈ {-1.0f, 0.0f} selon §5.3 (-1.0 =
 *       mat du côté au trait ; 0.0 = pat / nul).
 *   <li><strong>I-Node-4</strong> : {@link #totalVisits} ≥ 0. Pour tout nœud non-racine ayant
 *       {@code totalVisits > 0}, son {@link #parent} a un {@code child[i] == this} dans son tableau
 *       {@link #children}.
 *   <li><strong>I-Node-5</strong> : pour la racine après re-rooting, {@code parent == null}. Les
 *       autres nœuds peuvent avoir leur {@code parent} réinitialisé suite à re-rooting (orphelins →
 *       garbage collection).
 * </ul>
 *
 * <p>Visibilité : {@code package-private} dans {@code engine.internal}, accès direct aux champs
 * autorisé pour les classes du sub-package (PUCTSelector, Backup, LeafEvaluator, SearchTree).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class Node {

  /**
   * Lien vers le parent. {@code null} pour la racine et après re-rooting (cf. I-Node-5).
   * Réinitialisé à {@code null} par {@code SearchTree.rerootOnMove} lors de la promotion d'un child
   * en racine.
   */
  public Node parent;

  /**
   * Coup ayant mené du parent à ce nœud (encodage {@code Move} 16-bit, cf. SPEC-board §3.4). Vaut
   * {@code 0} pour la racine et après re-rooting.
   */
  public int moveFromParent;

  /**
   * Nombre total de visites {@code N(s)}, mis à jour pendant BACKUP (cf. §5.4, §15.5). {@code ≥ 0}.
   *
   * <p>(v1.2.0) {@link AtomicInteger} pour permettre BACKUP concurrent en mode batched (ADR-014).
   * En mode mono-thread, l'overhead vs {@code int} plain est de quelques ns par accès — négligeable
   * face au coût d'une simulation MCTS.
   */
  public final AtomicInteger totalVisits = new AtomicInteger(0);

  /**
   * Somme des valeurs propagées {@code W(s)}. {@code Q(s) = W(s) / N(s)} pour {@code N(s) > 0},
   * sinon {@link org.nanozero.engine.EngineConfig#fpuValue} (FPU). Mis à jour pendant BACKUP.
   *
   * <p>(v1.2.0) {@link AtomicFloat} pour permettre BACKUP concurrent en mode batched. Wrapper
   * lock-free CAS sur bits IEEE-754 (cf. ADR-014).
   */
  public final AtomicFloat totalValueSum = new AtomicFloat(0.0f);

  /**
   * {@code true} après EVALUATE (NN appelé, priors et children attribués). Cf. I-Node-1, I-Node-2.
   *
   * <p>(v1.2.0) {@code volatile} pour garantir la visibilité cross-thread de la transition false→
   * true. ⚠️ (Amendement v1.3.0) L'écriture est un <strong>plain write</strong>, PAS un CAS : deux
   * threads peuvent expanser concurremment la même feuille (<strong>expansion redondante
   * tolérée</strong> — stats initiales identiques, arbre cohérent, seul un forward NN redondant
   * gaspillé ; cf. ADR-013/ADR-014 amendés). Le « CAS en phase 1.2.0-4 » initialement prévu n'a PAS
   * été implémenté. En mode mono-thread la sémantique est inchangée.
   */
  public volatile boolean expanded;

  /**
   * {@code true} si la position est terminale (mat ou pat). Si {@code true}, alors {@code expanded
   * == true} et {@code childMoves.length == 0} (cf. I-Node-3).
   */
  public boolean terminal;

  /**
   * Valeur fixe pour les nœuds terminaux uniquement. Significatif ssi {@code terminal == true} :
   * vaut {@code -1.0f} en cas de mat (perte côté au trait) ou {@code 0.0f} en cas de pat / nul.
   */
  public float terminalValue;

  /**
   * Coups légaux depuis cette position, encodés au format {@code Move} 16-bit. Non-null ssi {@code
   * expanded == true} (cf. I-Node-1, I-Node-2). Longueur = nombre de coups légaux.
   *
   * <p>(v1.2.0) {@code volatile} pour publication safe cross-thread après expand : un thread B qui
   * voit {@code expanded == true} (volatile read = acquire barrier) est garanti de voir aussi la
   * dernière version de ce champ écrite avant le {@code expanded = true} (volatile write = release
   * barrier, cf. JMM happens-before).
   */
  public volatile int[] childMoves;

  /**
   * Priors {@code P(s, a)} issus du NN après {@code MoveEncoding.decodePolicy} (softmax masqué sur
   * les coups légaux). Non-null ssi {@code expanded == true}, longueur identique à {@link
   * #childMoves}.
   *
   * <p>(v1.2.0) {@code volatile} pour publication safe cross-thread (cf. {@link #childMoves}).
   */
  public volatile float[] childPriors;

  /**
   * Children matérialisés (lazy expansion : {@code children[i] == null} jusqu'au premier
   * SELECT-EXPAND ciblant {@code childMoves[i]}). Non-null ssi {@code expanded == true}, longueur
   * identique à {@link #childMoves}.
   *
   * <p>(v1.2.0) {@code volatile} pour publication safe cross-thread (cf. {@link #childMoves}). Note
   * : les <em>éléments</em> du tableau ne sont pas volatile per-se ; la lazy assignation d'un slot
   * {@code children[i] = new Node(...)} reste une race coopérative (deux threads peuvent alors
   * créer chacun un Node ; le dernier write "gagne", l'autre devient orphelin — tolérance acceptée
   * car les statistiques sur l'orphelin sont perdues mais l'arbre reste cohérent).
   */
  public volatile Node[] children;

  /**
   * (v1.1.0) Dirichlet noise vector for AlphaZero-style root exploration during self-play (cf. SPEC
   * §3.1 amendement, §5.2 injection, ADR-012).
   *
   * <p>Non-null seulement quand ce nœud est le root d'une SearchTree pendant une recherche avec
   * Dirichlet activé ({@code EngineConfig.dirichletEpsilon() > 0}). Quand non-null, le tableau a la
   * même longueur que {@link #childMoves} (un poids par coup légal), valeurs {@code >= 0}, somme =
   * 1.0.
   *
   * <p>Sampled une seule fois par recherche par {@code SearcherCore.runOneSimulation} (cf. §5.3) au
   * début de chaque sim, tant que la node est expandée et que ce champ est encore null. Idempotent
   * : pas de re-sampling si déjà non-null.
   *
   * <p>Pour les nodes non-root, ce champ reste null et le comportement v1.0.0 est strictement
   * préservé. La détection root par PUCTSelector se fait via {@code dirichletNoise != null} sans
   * test additionnel "is this the root?".
   *
   * <p>(v1.3.0 fix P2.3) {@code volatile} pour publication safe en mode batched : un thread sample
   * et écrit, les autres threads doivent voir la référence non-null sans cache stale. L'idempotence
   * stricte (jamais 2 samples concurrents) est assurée via {@link #DIRICHLET_NOISE_UPDATER}
   * compareAndSet dans SearcherCore.runOneSimulation.
   */
  public volatile float[] dirichletNoise;

  /**
   * (v1.3.0 fix P2.3) Updater atomique sur le champ {@link #dirichletNoise} pour garantir un seul
   * sample par recherche même en mode batched (N threads peuvent rentrer dans le bloc {@code if
   * (dirichletNoise == null)} simultanément). Le perdant du CAS abandonne son sample.
   */
  public static final java.util.concurrent.atomic.AtomicReferenceFieldUpdater<Node, float[]>
      DIRICHLET_NOISE_UPDATER =
          java.util.concurrent.atomic.AtomicReferenceFieldUpdater.newUpdater(
              Node.class, float[].class, "dirichletNoise");

  /**
   * (v1.2.0) Compteur de threads "in-flight" par child, pour virtual loss (cf. ADR-013 §15.5).
   * {@code childInFlight[i]} = nombre de threads ayant pris le chemin vers {@code children[i]} et
   * n'ayant pas encore terminé leur BACKUP. Utilisé par {@link PUCTSelector#argmax} pour pénaliser
   * temporairement les chemins déjà explorés et diversifier les K threads concurrents.
   *
   * <p>Lazy alloc : reste {@code null} jusqu'à ce qu'un thread {@link
   * #ensureChildInFlightAllocated()} en réponse à un premier SELECT en mode multi-thread. En mode
   * mono-thread ({@code searchThreads=1}), ce champ reste toujours {@code null} et le code path
   * PUCT court-circuite la branche virtual loss → comportement v1.1.2 strict bit-pour-bit.
   *
   * <p>Visibilité {@code volatile} car alloué cross-thread (un thread alloue, d'autres lisent).
   * L'array {@link AtomicIntegerArray} interne fournit les opérations atomiques par index.
   *
   * @apiNote Internal — réservé aux helpers MCTS du mode batched (cf. ADR-013).
   */
  public volatile AtomicIntegerArray childInFlight;

  /**
   * Construit un nœud enfant. Les champs statistiques restent à zéro / état non-expandé. Usage
   * typique : appelé par PUCTSelector / SearcherCore lors de la première sélection d'un coup
   * (matérialisation lazy).
   *
   * @param parent parent du nœud créé ({@code null} pour la racine)
   * @param moveFromParent coup ayant mené du parent à ce nœud ({@code 0} pour la racine)
   */
  public Node(Node parent, int moveFromParent) {
    this.parent = parent;
    this.moveFromParent = moveFromParent;
  }

  /**
   * (v1.2.0) Alloue {@link #childInFlight} si non-null déjà. Idempotent et thread-safe via
   * double-checked locking sur l'instance.
   *
   * <p>Doit être appelé après l'expansion (quand {@link #childMoves} est non-null), avant le
   * premier SELECT-concurrent vers un child. La taille du tableau est calquée sur {@code
   * childMoves.length}.
   *
   * @throws IllegalStateException si {@code childMoves == null} (nœud non-expandé)
   */
  public void ensureChildInFlightAllocated() {
    if (childInFlight != null) {
      return;
    }
    if (childMoves == null) {
      throw new IllegalStateException(
          "Cannot allocate childInFlight before expansion (childMoves == null)");
    }
    synchronized (this) {
      if (childInFlight == null) {
        childInFlight = new AtomicIntegerArray(childMoves.length);
      }
    }
  }

  /** Constructeur racine (équivalent à {@code new Node(null, 0)}). */
  public Node() {
    this(null, 0);
  }
}
