# ⚠️ Cœur copié — risque de déphasage (Phase 0)

Les fichiers `.mjs` de ce dossier sont des **copies** du cœur moteur du worker de contribution
(`nanozero-browser-worker/worker/src/` + `ui/board-render.mjs`), introduites en **Phase 0** (story F1)
pour que les surfaces **Jouer** (Lot P) et **Analyse** (Lot A) du site disposent d'un cœur JS réutilisable
sans modifier — ni risquer de régresser — le self-play en production.

## Règle absolue jusqu'à la Phase 5 (migration source unique, `architecture.md §9`)

> **Toute modification d'un fichier de `nanozero-web/core/` DOIT être appliquée à l'identique dans
> `nanozero-browser-worker/worker/src/` (ou `ui/board-render.mjs`), et inversement.**

Sinon le worker et le site divergent silencieusement (risque moyen / probabilité haute,
`architecture.md §10`). À rappeler dans `CONTRIBUTING.md`.

## Fichiers copiés et leur source

| Copie (`nanozero-web/core/`) | Source (`nanozero-browser-worker/`) |
|---|---|
| `fast-board.mjs` | `worker/src/fast-board.mjs` |
| `game-state.mjs` | `worker/src/game-state.mjs` |
| `mcts.mjs` | `worker/src/mcts.mjs` |
| `batched-evaluator.mjs` | `worker/src/batched-evaluator.mjs` |
| `nn.mjs` | `worker/src/nn.mjs` |
| `encoding/plane-encoding.mjs` | `worker/src/encoding/plane-encoding.mjs` |
| `encoding/move-encoding.mjs` | `worker/src/encoding/move-encoding.mjs` |
| `encoding/fen.mjs` | `worker/src/encoding/fen.mjs` |
| `board-render.mjs` | `ui/board-render.mjs` |

NON copiés (spécifiques worker, hors cœur réutilisable) : `nn-node.mjs`, `browser-api.mjs`,
`run-loop.mjs`, `self-play.mjs`, `jobserver-client.mjs`, `submit-codec.mjs`.

## Sortie de Phase 0

La Phase 5 remplace ces copies par une source unique (package partagé ou import direct). Tant que
ce dossier existe, considérer les copies comme **en lecture seule conceptuelle** : on n'édite le cœur
que des deux côtés à la fois.
