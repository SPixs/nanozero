package org.nanozero.engine.internal;

import java.util.Objects;
import org.nanozero.board.GameState;

/**
 * Encapsule la racine courante de l'arbre MCTS et la logique de re-rooting (cf. SPEC §3.2, §5.5).
 *
 * <p><strong>Stateless en lecture, mutable en écriture</strong> par le seul thread de recherche
 * (mono-thread search v1.0.0, cf. ADR-002). {@code SearchTree} ne tient aucun verrou ; toute
 * synchronisation cross-thread est portée par le {@code ThreadController} en phase 9-10.
 *
 * <p>Re-rooting : tente de promouvoir un sous-arbre déjà calculé en nouvelle racine pour économiser
 * le travail des recherches précédentes. Si le coup demandé n'est pas matérialisable (racine non
 * expandée, terminale, coup absent, child non instancié), {@code SearchTree} retombe sur un
 * <strong>fresh tree</strong> (nouvelle racine vierge).
 *
 * <p><strong>Responsabilité du caller</strong> : {@code SearchTree} ne vérifie PAS que {@code
 * newState} est cohérent avec l'application de {@code move} sur {@code rootState}. C'est l'engine
 * (phase 9) qui maintient cet invariant en passant des états cohérents — un mismatch silencieux
 * corromprait les statistiques de l'arbre re-rooté.
 *
 * <p><strong>Invariants normatifs §3.2</strong> :
 *
 * <ul>
 *   <li><strong>I-Tree-1</strong> : {@link #rootState()} correspond toujours exactement à la
 *       position que {@link #root()} représente.
 *   <li><strong>I-Tree-2</strong> : après re-rooting réussi, {@code root.parent == null} et tous
 *       les nœuds non-descendants de la nouvelle racine deviennent orphelins (donc
 *       GC-collectables).
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class SearchTree {

  private Node root;
  private GameState rootState;

  /**
   * Construit un nouveau {@code SearchTree} avec une racine fraîche (non expandée).
   *
   * @param initialState position initiale de la recherche (non null)
   * @throws NullPointerException si {@code initialState} est null
   */
  public SearchTree(GameState initialState) {
    Objects.requireNonNull(initialState, "initialState must not be null");
    this.root = new Node();
    this.rootState = initialState;
  }

  /** Racine courante. Jamais null. */
  public Node root() {
    return root;
  }

  /** État correspondant à la racine courante (cf. I-Tree-1). Jamais null. */
  public GameState rootState() {
    return rootState;
  }

  /**
   * Tente de re-rooter sur un coup joué. Si le coup correspond à un enfant déjà matérialisé, cet
   * enfant devient nouvelle racine ; sinon, fresh tree.
   *
   * <p>Algorithme strict §3.2 :
   *
   * <ol>
   *   <li>Si {@code root} non expandé OU {@code root.terminal} : fresh tree.
   *   <li>Sinon, parcourir {@code root.childMoves} :
   *       <ul>
   *         <li>si match {@code childMoves[i] == move} ET {@code children[i] != null} : promotion
   *             ({@code children[i].parent = null}, devient racine).
   *       </ul>
   *   <li>Si rien trouvé (coup absent ou child non instancié) : fresh tree.
   * </ol>
   *
   * <p><strong>Note</strong> : un coup présent dans {@code childMoves} mais avec {@code children[i]
   * == null} signifie qu'il a été énuméré lors de l'expansion (NN a fourni un prior) mais jamais
   * sélectionné par PUCT. Aucun sous-arbre à réutiliser → fresh tree.
   *
   * @param move coup ayant été joué (encodage {@code Move} 16-bit)
   * @param newState position résultante (non null) ; cohérence avec {@code move} à la charge du
   *     caller
   * @throws NullPointerException si {@code newState} est null
   */
  public void rerootOnMove(int move, GameState newState) {
    Objects.requireNonNull(newState, "newState must not be null");
    if (!root.expanded || root.terminal) {
      freshTree(newState);
      return;
    }
    for (int i = 0; i < root.childMoves.length; i++) {
      if (root.childMoves[i] == move && root.children[i] != null) {
        promote(root.children[i], newState);
        return;
      }
    }
    // Coup absent ou child non instancié.
    freshTree(newState);
  }

  /**
   * Tente de re-rooter sur deux coups consécutifs (cas standard : tour engine + tour adversaire
   * entre deux searches). Sémantiquement équivalent à un double saut sans état intermédiaire : la
   * fonction tente le double match (firstMove sur la racine, puis secondMove sur le child), sans
   * jamais matérialiser un état intermédiaire.
   *
   * <p>Tous les cas d'échec retombent sur un fresh tree :
   *
   * <ul>
   *   <li>racine non expandée ou terminale ;
   *   <li>{@code firstMove} absent de {@code root.childMoves} ;
   *   <li>{@code root.children[i]} pour ce coup est null (jamais sélectionné) ;
   *   <li>l'intermédiaire trouvé n'est pas expandé ou est terminal ;
   *   <li>{@code secondMove} absent des childMoves de l'intermédiaire ;
   *   <li>l'enfant correspondant à {@code secondMove} est null.
   * </ul>
   *
   * @param firstMove premier coup (engine) ayant été joué
   * @param secondMove deuxième coup (adversaire) ayant été joué
   * @param newState position résultante après les deux coups (non null) ; cohérence avec {@code
   *     firstMove} puis {@code secondMove} à la charge du caller
   * @throws NullPointerException si {@code newState} est null
   */
  public void rerootOnMoves(int firstMove, int secondMove, GameState newState) {
    Objects.requireNonNull(newState, "newState must not be null");
    if (!root.expanded || root.terminal) {
      freshTree(newState);
      return;
    }
    for (int i = 0; i < root.childMoves.length; i++) {
      if (root.childMoves[i] != firstMove || root.children[i] == null) {
        continue;
      }
      Node intermediate = root.children[i];
      if (!intermediate.expanded || intermediate.terminal) {
        freshTree(newState);
        return;
      }
      for (int j = 0; j < intermediate.childMoves.length; j++) {
        if (intermediate.childMoves[j] == secondMove && intermediate.children[j] != null) {
          promote(intermediate.children[j], newState);
          return;
        }
      }
      // Intermédiaire trouvé mais secondMove introuvable.
      freshTree(newState);
      return;
    }
    // firstMove introuvable.
    freshTree(newState);
  }

  // -------------------------------------------------------------------------------------------
  // Helpers privés
  // -------------------------------------------------------------------------------------------

  private void freshTree(GameState newState) {
    this.root = new Node();
    this.rootState = newState;
  }

  private void promote(Node newRoot, GameState newState) {
    newRoot.parent = null;
    newRoot.moveFromParent = 0;
    this.root = newRoot;
    this.rootState = newState;
  }
}
