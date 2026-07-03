/**
 * API publique du module {@code nanozero-engine} : recherche MCTS PUCT (variante AlphaZero) guidée
 * par réseau de neurones (cf. SPEC §11.1).
 *
 * <p>Classes exposées hors du module :
 *
 * <ul>
 *   <li>{@code Engine} — point d'entrée principal, API asynchrone (startSearch / startPonder /
 *       ponderhit / stop / currentBest)
 *   <li>{@code EngineConfig} — record d'hyperparamètres (cPuct, fpuValue, treeInitialCapacity)
 *   <li>{@code EngineState} — enum (IDLE, SEARCHING, PONDERING, STOPPING, DONE, CLOSED)
 *   <li>{@code SearchBudget} — interface composable + factories (nodes, deadline, duration,
 *       untilStopped, firstOf, UNLIMITED)
 *   <li>{@code SearchResult} — record immutable (bestMove, principalVariation, value,
 *       simulationsCount, elapsedNanos, childVisits, childMoves, terminated)
 * </ul>
 *
 * <p>Le sub-package {@link org.nanozero.engine.internal} contient la mécanique MCTS interne (Node,
 * SearchTree, PUCTSelector, Backup, LeafEvaluator, SearcherCore, ThreadController,
 * EngineStateMachine), marquée {@code @apiNote Internal}.
 */
package org.nanozero.engine;
