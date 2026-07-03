package org.nanozero.engine.internal;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.internal.batched.LeafSubmission;
import org.nanozero.engine.internal.batched.LeafSubmissionQueue;
import org.nanozero.nn.BitboardPlaneEncoderVector;
import org.nanozero.nn.MoveEncoding;

/**
 * Boucle MCTS mono-thread synchrone (cf. SPEC §5.1, §12 phase 6). Assemble les briques :
 *
 * <ul>
 *   <li>{@link PUCTSelector} pour la phase SELECT (descente d'arbre, §5.2).
 *   <li>{@link LeafEvaluator} pour la phase EXPAND/EVALUATE (call NN mocké, §5.3).
 *   <li>{@link Backup} pour la phase BACKUP (propagation alternée, §5.4).
 * </ul>
 *
 * <p><strong>Stratégie d'état</strong> : make-undo sur {@link GameState} (cohérent avec l'API
 * mutable de {@code nanozero-board}). Pendant la descente, on appelle {@code applyMove} sur {@code
 * rootState} ; à la fin de la simulation, on appelle {@code unapplyLastMove} {@code depth} fois
 * pour restaurer l'état initial. Évite les allocations {@code new GameState} à chaque descente.
 *
 * <p><strong>Terminal handling minimal</strong> : si la feuille est terminale, on stocke {@code
 * terminalValue} (-1 pour mat, 0 pour autres cas de nul) sans appeler le NN. Le terminal handling
 * complet (avec distinction mat / pat / répétition / 50 coups / matériel insuffisant dans des
 * chemins de code dédiés) est phase 8.
 *
 * <p><strong>Concurrence</strong> (mise à jour v1.2.0/v1.3.0) : {@link #runOneSimulation} est
 * <strong>thread-safe</strong> et s'exécute en parallèle sur N threads en mode multi-thread
 * (ADR-013, ADR-016) — chaque thread a son propre {@code SearcherCore} + {@code GameState}, l'arbre
 * est partagé via atomics ({@link Node}, {@link Backup}). En mono-thread (défaut {@code
 * searchThreads=1}, {@code virtualLoss=0}) le comportement v1.1.2 bit-pour-bit est préservé. La
 * Javadoc « aucune concurrence, mono-thread v1.0.0 » d'origine est obsolète depuis v1.2.0.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class SearcherCore {

  private static final int[] EMPTY_INT_ARRAY = new int[0];
  private static final float[] EMPTY_FLOAT_ARRAY = new float[0];
  private static final Node[] EMPTY_NODE_ARRAY = new Node[0];

  private final EngineConfig config;
  private final LeafEvaluator leafEvaluator;

  /**
   * (v1.2.0-fix2) Queue partagée des soumissions NN en mode batched. {@code null} en mode mono
   * (comportement v1.1.2 bit-pour-bit via {@link #leafEvaluator}). En mode batched, chaque thread
   * MCTS soumet sa feuille à cette queue et attend la complétion via la future — le {@code
   * NNEvalThread} dédié drain la queue par batch K et appelle {@code Network.forward(planes, K,
   * output)} une seule fois. Cf. SPEC §15.3, ADR-013, ADR-015.
   */
  private final LeafSubmissionQueue batchedQueue;

  /**
   * (v1.2.0-fix2) Buffer planes single-position (119 × 64 = 7616 floats) utilisé pour encoder la
   * position avant submission à la queue. ThreadLocal de facto (1 SearcherCore par thread en mode
   * batched, cf. ThreadController.threadLocalSearcher). {@code null} en mode mono.
   */
  private final float[] batchedPlanesBuffer;

  /** Buffer pré-alloué pour {@code state.generateMoves} à chaque expansion. */
  private final int[] legalMovesBuffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];

  /** Buffer pré-alloué pour les priors normalisés retournés par {@link LeafEvaluator}. */
  private final float[] priorsBuffer = new float[MoveGen.RECOMMENDED_BUFFER_SIZE];

  /**
   * (v1.2.0-fix) Buffer pré-alloué pour mémoriser l'index sélectionné à chaque niveau de la
   * descente SELECT, requis pour décrémenter {@code childInFlight} sur le path remonté lors du
   * BACKUP. Taille 256 suffit largement (profondeur MCTS typique ≤ 30, parties ≤ 200 plies). Le
   * garde-fou M4 (cf. {@link #runOneSimulation}) borne la descente à {@code .length} pour éviter
   * tout {@code ArrayIndexOutOfBoundsException} dans le hot path.
   *
   * <p>Visibilité <strong>package-private non-final</strong> (et non {@code private final}) : seam
   * de test permettant à {@code SearcherCoreTest} de réduire la capacité (ex {@code new int[1]})
   * pour exercer le garde-fou M4 sans construire un arbre de profondeur 256. Aucun code de
   * production ne réaffecte ce champ.
   */
  int[] pathIndicesBuffer = new int[256];

  /**
   * (v1.1.0) Random instance unique pour le sampling Dirichlet (cf. SPEC §5.3 amendement, ADR-012).
   * Instancié une seule fois au constructor avec {@code new Random(config.randomSeed())} et
   * réutilisé entre toutes les simulations.
   *
   * <p><strong>Piège déterminisme</strong> : ne JAMAIS instancier un nouveau {@link Random} à
   * chaque sim — cela produirait des samples identiques (même seed → même séquence). La Javadoc de
   * {@code DirichletSampler.sample} documente ce piège.
   */
  private final Random rng;

  /**
   * Construit un {@code SearcherCore}. Le {@link LeafEvaluator} interne est composé sur le {@link
   * NetworkProvider} fourni.
   *
   * @param networkProvider provider NN mockable, non null
   * @param config hyperparamètres ({@code cPuct}, {@code fpuValue}, et v1.1.0 {@code
   *     dirichletAlpha}, {@code dirichletEpsilon}, {@code randomSeed}), non null
   * @throws NullPointerException si un argument est null
   */
  public SearcherCore(NetworkProvider networkProvider, EngineConfig config) {
    this(networkProvider, config, null, null);
  }

  /**
   * (v1.2.0-fix2) Construit un {@code SearcherCore} avec une queue batched optionnelle. Si {@code
   * batchedQueue != null}, les évaluations NN passent par la queue (submit + await future) au lieu
   * du forward direct via {@link LeafEvaluator} — permet 1 forward(K) batché au lieu de N forwards
   * séquentiels. Mode prévu pour le multi-thread (cf. ADR-013).
   *
   * @param networkProvider provider NN (utilisé en mode mono, ignoré dans le path batched), non
   *     null
   * @param config hyperparamètres, non null
   * @param batchedQueue queue partagée pour mode batched, {@code null} pour mode mono v1.1.2
   * @throws NullPointerException si {@code networkProvider} ou {@code config} est null
   */
  public SearcherCore(
      NetworkProvider networkProvider, EngineConfig config, LeafSubmissionQueue batchedQueue) {
    this(networkProvider, config, batchedQueue, null);
  }

  /**
   * (ADR-018) Construit un {@code SearcherCore} avec une queue batched optionnelle ET un {@link
   * NNEvalCache} optionnel partagé. Le cache est passé au {@link LeafEvaluator} interne (mode A).
   * Le path batched (mode B) ne passe pas par le {@code LeafEvaluator} : le cache y est inactif
   * (câblage mode B = follow-up, cf. ADR-018). {@code cache == null} ⇒ comportement v1.x
   * bit-pour-bit.
   *
   * @param networkProvider provider NN (utilisé en mode mono, ignoré dans le path batched), non
   *     null
   * @param config hyperparamètres, non null
   * @param batchedQueue queue partagée pour mode batched, {@code null} pour mode mono v1.1.2
   * @param cache cache d'évaluation NN partagé (ADR-018), ou {@code null} pour désactiver
   * @throws NullPointerException si {@code networkProvider} ou {@code config} est null
   */
  public SearcherCore(
      NetworkProvider networkProvider,
      EngineConfig config,
      LeafSubmissionQueue batchedQueue,
      NNEvalCache cache) {
    Objects.requireNonNull(networkProvider, "networkProvider must not be null");
    this.config = Objects.requireNonNull(config, "config must not be null");
    this.leafEvaluator = new LeafEvaluator(networkProvider, cache);
    // (v1.3.0 fix P2.3) Seed per-thread en mode batched uniquement. Sans cet offset, en mode
    // batched chaque thread créerait un Random avec la MÊME seed → séquences identiques sur N
    // threads → diversification Dirichlet neutralisée (tous les threads produiraient les mêmes
    // samples corrélés). En mode mono-thread (searchThreads<=1) la seed reste pure pour préserver
    // le déterminisme bit-pour-bit garanti par ADR-010 (1000 runs identiques → résultats
    // bit-pour-bit identiques).
    long rngSeed =
        config.searchThreads() > 1
            ? (config.randomSeed() ^ Thread.currentThread().threadId())
            : config.randomSeed();
    this.rng = new Random(rngSeed);
    this.batchedQueue = batchedQueue;
    this.batchedPlanesBuffer = (batchedQueue != null) ? new float[GameState.NN_PLANES * 64] : null;
  }

  /**
   * Exécute une recherche MCTS jusqu'à épuisement du budget. Mute {@code tree} en place : nœuds
   * expandés, statistiques propagées. Restaure {@code tree.rootState()} à son état initial à la fin
   * (make-undo strict).
   *
   * @param tree arbre à explorer, non null
   * @param budget critère d'arrêt, non null
   * @return nombre total de simulations effectuées (≥ 0)
   * @throws NullPointerException si {@code tree} ou {@code budget} sont null
   */
  public int runSearch(SearchTree tree, SearchBudget budget) {
    Objects.requireNonNull(tree, "tree must not be null");
    Objects.requireNonNull(budget, "budget must not be null");

    long start = System.nanoTime();
    int simulations = 0;
    while (!budget.shouldStop(simulations, System.nanoTime() - start)) {
      runOneSimulation(tree, tree.rootState());
      simulations++;
    }
    return simulations;
  }

  /**
   * Une simulation MCTS complète : SELECT → EXPAND/EVALUATE → BACKUP. Mute l'arbre en place et
   * laisse {@code state} dans son état initial via make-undo.
   *
   * <p>Visibilité {@code package-private} : exposée pour {@code ThreadController} (phase 10) qui
   * boucle lui-même avec check d'état entre sims, au lieu de passer par {@link #runSearch} qui
   * boucle en interne.
   *
   * <p>(v1.2.0) Le {@link GameState} est passé en argument plutôt que lu via {@code
   * tree.rootState()} : permet le mode batched multi-thread (ADR-016) où chaque thread possède sa
   * propre copie clonée du root state. En mode mono-thread, le caller passe {@code
   * tree.rootState()} et le comportement v1.1.2 est strictement préservé bit-pour-bit.
   *
   * @param tree arbre MCTS (partagé, atomic compteurs)
   * @param state {@link GameState} mutable sur lequel descendre — DOIT initialement matcher {@code
   *     tree.rootState()} ; sera restauré à cet état initial à la fin via make-undo
   */
  void runOneSimulation(SearchTree tree, GameState state) {
    Node root = tree.root();

    // v1.1.0 — Sampling Dirichlet conditionnel au root (cf. SPEC §5.3 amendement, ADR-012).
    // Position : AVANT la boucle SELECT pour que le premier PUCTSelector.argmax(root) voie déjà
    // le noise. Idempotent (check dirichletNoise == null) : un seul sample par root pour toute
    // la durée d'une recherche. Sur la 1ère sim, root.expanded est false → skip ; au plus tard
    // sur la 2e sim, root est expandé et le sampling s'opère.
    // v1.1.2 — Condition `!root.terminal` : protège contre une recherche démarrée par erreur sur
    // position terminale (mat / pat) où childMoves.length == 0, qui ferait lever IAE à
    // DirichletSampler.sample(alpha, 0, rng). Sampling silencieusement skipé sur root terminal.
    // (v1.3.0 review) La condition redondante `root.childMoves.length > 0` a été RETIRÉE : quand
    // `root.expanded && !root.terminal`, l'expansion non-terminale garantit toujours
    // `childMoves.length >= 1` (une position non-terminale a >= 1 coup légal) → c'était une branche
    // morte inatteignable. La protection NPE sur `root.childMoves` vient de `root.expanded` (qui
    // court-circuite avant tout accès), pas de cette condition.
    if (config.dirichletEpsilon() > 0f
        && root.expanded
        && !root.terminal
        && root.dirichletNoise == null) {
      // (v1.3.0 fix P2.3) Compare-and-set : en mode batched, N threads peuvent évaluer
      // dirichletNoise==null simultanément. Sans CAS, tous samplent en parallèle (gaspillage
      // + samples corrélés car tous threads partagent même seed). Avec CAS, seul le 1er gagnant
      // écrit, les autres abandonnent leur sample. Idempotence stricte.
      float[] sampled =
          DirichletSampler.sample(config.dirichletAlpha(), root.childMoves.length, this.rng);
      Node.DIRICHLET_NOISE_UPDATER.compareAndSet(root, null, sampled);
    }

    Node leaf = root;
    int depth = 0;
    // (v1.2.0-fix) Virtual loss actif si vloss>0 (mode multi-thread). En mono (vloss==0)
    // toute la mécanique childInFlight est court-circuitée — comportement v1.1.2 strict.
    final float vloss = config.virtualLoss();
    final boolean vlossActive = vloss > 0f;

    // (v1.3.0 fix P1.1) try/finally pour garantir le release des effets de bord (vloss decrement +
    // state make-undo restoration) même si une exception remonte du NN (e.g. CompletableFuture
    // .join() lance CompletionException, batchedQueue.submit InterruptedException, ou
    // leafEvaluator.evaluate RuntimeException). Sans ce try/finally, sur erreur NN :
    //   - childInFlight reste incrémenté → virtual loss persistante sur la branche → tous les
    //     threads suivants la pénalisent perpétuellement (bug détecté lors de la review).
    //   - state reste avec depth coups appliqués → la prochaine sim copie un GameState corrompu.
    // Raisonnement détaillé dans ce commentaire + l'amendement v1.3.0 d'ADR-013 (pas d'ADR dédié).
    try {
      // SELECT : descendre tant que la feuille courante est expandée non terminale.
      while (leaf.expanded && !leaf.terminal) {
        int bestIdx = PUCTSelector.argmax(leaf, config);
        // (v1.2.0-fix) Virtual loss wiring : penalize the branche choisie avant de descendre, pour
        // que les autres threads qui sélectionnent en concurrence sur ce même nœud voient un Q
        // pénalisé et choisissent un autre child. Sans cet incrément, PUCTSelector lit toujours
        // inFlight=0 et la formule virtual loss est inopérante (4 threads convergent sur le même
        // chemin). Le décrément a lieu après BACKUP sur le path remonté.
        // (v1.3.0 fix M4) Garde-fou de profondeur : rien ne borne structurellement la descente
        // MCTS (typiquement ≤ 30, mais une branche pathologique pourrait dépasser). Au-delà de la
        // capacité du buffer on continue à descendre SANS enregistrer le virtual loss (dégradation
        // gracieuse) plutôt que de lever ArrayIndexOutOfBoundsException dans le hot path.
        if (vlossActive && depth < pathIndicesBuffer.length) {
          leaf.ensureChildInFlightAllocated();
          leaf.childInFlight.getAndIncrement(bestIdx);
          pathIndicesBuffer[depth] = bestIdx;
        }
        int move = leaf.childMoves[bestIdx];
        state.applyMove(move);
        depth++;
        // (v1.2.0) Variable locale pour éviter relire le slot children[bestIdx] après assignation.
        // En mode batched, un autre thread peut écrire le slot entre temps (race coopérative —
        // tolérée car les 2 Node créés ont les mêmes stats initiales 0). On garde la référence
        // locale non-null pour continuer la descente.
        Node next = leaf.children[bestIdx];
        if (next == null) {
          next = new Node(leaf, move);
          leaf.children[bestIdx] = next;
        }
        leaf = next;
      }

      // EXPAND + EVALUATE.
      float value;
      if (leaf.expanded) {
        // Sortie de la boucle SELECT avec expanded=true : nécessairement terminal (cas re-visite
        // d'un nœud terminal détecté précédemment). Re-utilise terminalValue déjà stocké.
        value = leaf.terminalValue;
      } else if (state.isTerminal()) {
        value = state.isCheckmate() ? -1.0f : 0.0f;
        leaf.terminal = true;
        leaf.terminalValue = value;
        leaf.childMoves = EMPTY_INT_ARRAY;
        leaf.childPriors = EMPTY_FLOAT_ARRAY;
        leaf.children = EMPTY_NODE_ARRAY;
        leaf.expanded = true;
      } else {
        int n = state.generateMoves(legalMovesBuffer, 0);
        if (batchedQueue != null) {
          // (v1.2.0-fix2) Mode batched : encode planes localement, submit à la queue, await la
          // future. Le NNEvalThread dédié batche K submissions et fait 1 forward(K) — c'est ce qui
          // transforme le parallélisme CPU en vrai gain Elo (sinon les N threads font N forwards
          // séquentiels d'1 plane chacun, perdant l'avantage du batched). Cf. SPEC §15.3, ADR-013.
          state.toPlanes(batchedPlanesBuffer, 0, BitboardPlaneEncoderVector.INSTANCE);
          CompletableFuture<LeafSubmission.Result> future = new CompletableFuture<>();
          try {
            batchedQueue.submit(new LeafSubmission(batchedPlanesBuffer, future));
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while submitting NN leaf", e);
          }
          LeafSubmission.Result result = future.join();
          MoveEncoding.decodePolicy(
              result.policyLogits(),
              legalMovesBuffer,
              n,
              state.currentPosition().sideToMove(),
              priorsBuffer);
          value = result.value();
        } else {
          value = leafEvaluator.evaluate(state, legalMovesBuffer, n, priorsBuffer);
        }
        // Copies obligatoires : les buffers sont réutilisés à la prochaine simulation.
        leaf.childMoves = Arrays.copyOf(legalMovesBuffer, n);
        leaf.childPriors = Arrays.copyOf(priorsBuffer, n);
        leaf.children = new Node[n];
        leaf.expanded = true;
      }

      // BACKUP : propagation alternée jusqu'à la racine.
      Backup.backup(leaf, value);
    } finally {
      // (v1.3.0 fix P1.1) Release garantis même sur exception. La valeur `depth` est l'indice
      // jusqu'auquel on a appliqué un move + (si vlossActive) incrémenté childInFlight. Les
      // décrémenter en ordre inverse via pathIndicesBuffer.
      if (vlossActive) {
        Node n = leaf;
        for (int i = depth - 1; i >= 0; i--) {
          n = n.parent;
          // (v1.3.0 fix M4) Seuls les niveaux dans la capacité du buffer ont été pénalisés en
          // SELECT — on ne décrémente que ceux-là (symétrie stricte apply/release).
          if (i < pathIndicesBuffer.length) {
            n.childInFlight.decrementAndGet(pathIndicesBuffer[i]);
          }
        }
      }
      // Restaure l'état initial via make-undo (contrat strict GameState).
      for (int i = 0; i < depth; i++) {
        state.unapplyLastMove();
      }
    }
  }
}
