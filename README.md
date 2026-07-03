# NanoZero

**Un moteur d'échecs AlphaZero, entraîné depuis zéro — par la communauté.**

🌐 **[nanozero.org](https://nanozero.org)** — jouer contre Nano, analyser ses parties, et contribuer à
l'entraînement directement depuis le navigateur, sans rien installer.

NanoZero implémente le paradigme AlphaZero : recherche MCTS (PUCT) guidée par un réseau de neurones
ResNet, entraîné **from scratch en pur self-play** — zéro connaissance humaine, pas de livre d'ouverture.
L'inférence tourne en **pur Java** (Vector API SIMD) ou via ONNX Runtime (CPU/GPU/navigateur).

## Le site

| Surface | Description |
|---|---|
| [Contribuer](https://nanozero.org/worker/) | Prête ton GPU/CPU le temps d'un onglet : le navigateur joue des parties d'entraînement (ONNX Runtime Web, WebGPU/WASM) et les envoie — anonymement, sans compte |
| [Analyser](https://nanozero.org/analyse/) | Colle un PGN/FEN : graphe W/D/L de la partie, meilleures variantes, indice de criticité — 100 % dans le navigateur |
| [Jouer](https://nanozero.org/play/) | Trois niveaux calibrés en Elo, du mode tranquille à la pleine force |

Aucun cookie, aucun traceur, aucune requête tierce — tout est auto-hébergé.

## Architecture (monorepo)

| Module | Rôle |
|---|---|
| `nanozero-board` | Représentation, génération de coups, plans d'entrée NN (validée par perft + cross-validation) |
| `nanozero-nn` | Inférence ResNet en pur Java Vector API SIMD (parité PyTorch ~1e-6) + backend ONNX Runtime. Value head WDL |
| `nanozero-engine` | MCTS PUCT AlphaZero : API asynchrone, tree reuse Zobrist, multi-thread, batching GPU |
| `nanozero-uci` | Adaptateur protocole UCI (exécutable `UciMain`) |
| `nanozero-worker` | Worker self-play distribué (claim → play → submit) |
| `nanozero-sprt` | Tests SPRT en Java (statistiques portées de fastchess) |
| `nanozero-jobserver` | Serveur de jobs FastAPI + SQLite WAL : replay buffer, distribution des modèles, observabilité |
| `nanozero-training` | Pipeline d'entraînement PyTorch : datasets self-play → training → export ONNX/NPZ |
| `nanozero-browser-worker` | Worker navigateur (page volontaire) : MCTS JavaScript + ONNX Runtime Web |
| `nanozero-web` | Le site : landing, Jouer, Analyser (vanilla JS, esbuild) + déploiement Caddy |
| `nanozero-human-fidelity` | Recherche : indicateurs de criticité et de jouabilité calibrés sur des parties humaines |

## Build

Prérequis : **JDK 25**, **Maven 3.9+** (modules Java) ; **Python 3.11+** (training/jobserver) ; **Node 20+** (web).

```bash
mvn -B verify                    # build complet : compile + tests + coverage + format
mvn -pl nanozero-board verify    # un seul module
```

Conventions : Google Java Format (Spotless, build-break), Javadoc sur l'API publique, JUnit 5 + AssertJ,
couverture cible 90 % par module, TDD sur les algorithmes critiques (movegen, kernels NN, MCTS).

## Comment ça apprend

Cycle continu : les workers (flotte + navigateurs volontaires) jouent des parties en self-play →
le jobserver collecte les positions → entraînement PyTorch d'une nouvelle génération → **SPRT**
(test séquentiel de force, seul arbitre) → si la nouvelle génération est plus forte, elle est promue
et la flotte bascule dessus. Chaque génération est le professeur de la suivante.

## Licence

[GPL-3.0](LICENSE) — comme Stockfish et Leela Chess Zero : le moteur et ses dérivés restent libres,
et l'effort d'entraînement communautaire reste au bénéfice de la communauté.

## Contact

Questions, suggestions : **contact@nanozero.org** · [Mentions légales](https://nanozero.org/mentions-legales/)
