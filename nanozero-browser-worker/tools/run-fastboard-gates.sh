#!/usr/bin/env bash
# run-fastboard-gates.sh — Lance TOUS les gates du filet FastBoard (Story 0 baseline + Stories 1-4).
#   1. (option --java) régénère /tmp/reference.jsonl depuis CrossValHarness.java (JDK 25)
#   2. Story 0 (baseline GameState/chess.js) : perft, invariants, cross-validate Java↔JS.
#   3. Stories 1-4 (FastBoard) : perft, invariants make/unmake, zobrist, encoder-state,
#      cross-validate Java↔FastBoard, legalMoves {from,to,type,promo}==GameState, game-over.
# Usage : tools/run-fastboard-gates.sh [--java]
set -euo pipefail

REPO=$HOME/dev/nanozero
TOOLS="$REPO/nanozero-browser-worker/tools"
WORKER="$REPO/nanozero-browser-worker/worker"
JDK=/usr/lib/jvm/java-25-openjdk-amd64
BOARD="$REPO/nanozero-board/target/classes"
NN="$REPO/nanozero-nn/target/classes"
OUT="${TMPDIR:-/tmp}/fastboard-harness-out"

if [[ "${1:-}" == "--java" ]]; then
  echo "### Régénération du dump Java (CrossValHarness) ###"
  if [[ ! -d "$BOARD" || ! -d "$NN" ]]; then
    echo "  target/classes manquant -> compilation des modules board+nn..."
    (cd "$REPO" && mvn -q -pl nanozero-board,nanozero-nn -am compile)
  fi
  mkdir -p "$OUT"
  "$JDK/bin/javac" -cp "$BOARD:$NN" -d "$OUT" "$TOOLS/CrossValHarness.java"
  "$JDK/bin/java" -cp "$BOARD:$NN:$OUT" CrossValHarness > /tmp/reference.jsonl
  echo "  -> /tmp/reference.jsonl ($(wc -l < /tmp/reference.jsonl) lignes)"
fi

cd "$WORKER"
echo; echo "############## STORY 0 — baseline GameState (chess.js) ##############"
echo; echo "### GATE 1+2 : PERFT ###";          node test/fast-board-perft.mjs
echo; echo "### GATE 3 : INVARIANTS ###";       node test/fast-board-invariants.mjs
echo; echo "### GATE 4 : CROSS-VALIDATE ###";   node test/cross-validate.mjs

echo; echo "############## STORIES 1-4 — FastBoard (chessops) ##############"
echo; echo "### S1 : PERFT FastBoard (197281 / 97862) ###";        node test/fast-board-perft-fb.mjs
echo; echo "### S1 : INVARIANT make/unmake FastBoard ###";         node test/fast-board-invariants-fb.mjs
echo; echo "### S2 : ZOBRIST incrémental + ep convention-Java ###"; node test/fast-board-zobrist-fb.mjs
echo; echo "### S3 : ENCODER-STATE == GameState ###";              node test/encoder-state-equality-fb.mjs
echo; echo "### S3 : CROSS-VALIDATE Java↔FastBoard (561 coups) ###"; node test/cross-validate-fb.mjs
echo; echo "### S4 : legalMoves {from,to,type,promo} == GameState ###"; node test/legal-moves-equality-fb.mjs
echo; echo "### S4 : isGameOver/isCheckmate/isDraw == GameState ###"; node test/game-over-equality-fb.mjs

echo; echo "✅✅✅ TOUS LES GATES FASTBOARD (Story 0 + Stories 1-4) SONT VERTS"
