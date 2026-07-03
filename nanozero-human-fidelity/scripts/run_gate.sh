#!/usr/bin/env bash
# Orchestre le gate β : label (Stockfish shardé, nice-19) → compute_beta → validate.
# DevSrv partage le self-play → nice-19 + nb de shards capé pour ne pas voler de cœurs aux workers.
#
# Usage : run_gate.sh POSITIONS.csv [ONNX] [DEPTH] [SHARDS] [TAU_CP]
set -euo pipefail

POS=${1:?usage: run_gate.sh positions.csv [onnx] [depth] [shards] [tau_cp]}
ONNX=${2:-"$HOME/nanozero-night/models/gen-031-promoted.onnx"}
DEPTH=${3:-16}
SHARDS=${4:-6}
TAU=${5:-100}

SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.gatevenv/bin/python"
DIR=$(dirname "$POS"); BASE=$(basename "$POS" .csv)
LABELED="$DIR/$BASE.labeled.csv"; BETA="$DIR/$BASE.beta.csv"; REPORT="$DIR/$BASE.report.json"

N=$(($(wc -l < "$POS") - 1))
CHUNK=$(( (N + SHARDS - 1) / SHARDS ))
echo "[1/3] label : N=$N positions, $SHARDS shards × ~$CHUNK, Stockfish depth=$DEPTH, nice-19"
pids=()
for s in $(seq 0 $((SHARDS - 1))); do
  start=$((s * CHUNK))
  nice -n 19 "$PY" "$SC/label_blunders.py" "$POS" "$DIR/$BASE.lab.$s.csv" \
    --depth "$DEPTH" --tau-cp "$TAU" --start "$start" --count "$CHUNK" --threads 1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

head -1 "$DIR/$BASE.lab.0.csv" > "$LABELED"
for s in $(seq 0 $((SHARDS - 1))); do tail -n +2 "$DIR/$BASE.lab.$s.csv" >> "$LABELED"; done
rm -f "$DIR/$BASE".lab.*.csv
echo "      labeled → $LABELED ($(($(wc -l < "$LABELED") - 1)) lignes)"

echo "[2/3] compute_beta (champion = $ONNX)"
"$PY" "$SC/compute_beta.py" "$LABELED" "$BETA" --onnx "$ONNX"

echo "[3/3] validate"
"$PY" "$SC/validate_beta.py" "$BETA" --out "$REPORT"
echo "rapport → $REPORT"
