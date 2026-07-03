#!/usr/bin/env bash
# calibrate.sh — calibration Elo des niveaux de /play/ via fastchess.
# Méthode : échelle par VOISINS (sims adjacents, temp 0) chaînée + 1 ancrage TSCP (sd6 ≈ 1700) +
# coût de la température à 2 sims représentatifs. Moteur = bridge UCI Node (= moteur exact du navigateur).
# Sorties structurées dans $OUT/results.tsv → post-traitées par calibrate-report.mjs.
#
# Usage :  GAMES=150 CC=12 bash bench/calibrate.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FC=${FASTCHESS:-"$HOME/dev/fastchess/fastchess"}
BRIDGE=$HERE/uci-bridge.mjs
TSCP=${TSCP_CMD:-/tmp/tscp_uci.sh}
BOOK=${BOOK:-$HERE/book.pgn}
GAMES=${GAMES:-150}
CC=${CC:-12}
TC=${TC:-300+3}          # généreux : moteurs à force fixe (sims/sd), jamais de perte au temps
OUT=${OUT:-/tmp/calib}
export NANOZERO_THREADS=1
mkdir -p "$OUT"
: > "$OUT/results.tsv"   # type  Asims Atemp  Bsims Btemp  elo  err   (B vide = TSCP)
echo "calibration → $OUT  (GAMES=$GAMES CC=$CC TC=$TC)"

san() { echo "$1" | tr -d '.'; }   # 0.15 -> 015 (noms d'engine sans point)

# nano vs nano : record "LADDER/TEMP  Asims Atemp Bsims Btemp elo(A-B) err"
nn() {
  local s1=$1 t1=$2 s2=$3 t2=$4 typ=$5 tag="${5}_${1}_$(san $2)_vs_${3}_$(san $4)"
  echo "  match $tag …"
  "$FC" -engine cmd=node "args=$BRIDGE" name="A_${s1}_$(san $t1)" option.Sims=$s1 option.Temperature=$t1 \
        -engine cmd=node "args=$BRIDGE" name="B_${s2}_$(san $t2)" option.Sims=$s2 option.Temperature=$t2 \
        -each tc=$TC -rounds $((GAMES/2)) -games 2 -repeat -concurrency $CC \
        -openings file="$BOOK" format=pgn order=random > "$OUT/$tag.log" 2>&1
  # Valeur FINALE : dernière ligne "Elo:" (head=intervalle précoce, bug) ; ancrée sur ^Elo: pour NE PAS
  # matcher "nElo:". Champs awk : $2=Elo, $4=±err (virgule de fin retirée).
  local line=$(grep -E '^Elo:' "$OUT/$tag.log" | tail -1)
  local elo=$(echo "$line" | awk '{print $2}')
  local err=$(echo "$line" | awk '{gsub(/,/,"",$4); print $4}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$typ" "$s1" "$t1" "$s2" "$t2" "${elo:-NA}" "${err:-NA}" >> "$OUT/results.tsv"
  echo "    → Elo(A-B)=${elo:-NA} ±${err:-NA}"
}

# nano vs tscp (ancrage)
nt() {
  local s1=$1 t1=$2 tag="ANCHOR_${1}_$(san $2)"
  echo "  match $tag (vs TSCP) …"
  "$FC" -engine cmd=node "args=$BRIDGE" name="Nano_${s1}" option.Sims=$s1 option.Temperature=$t1 \
        -engine cmd="$TSCP" name=TSCP_sd6 \
        -each tc=$TC -rounds $((GAMES/2)) -games 2 -repeat -concurrency $CC \
        -openings file="$BOOK" format=pgn order=random > "$OUT/$tag.log" 2>&1
  # Valeur FINALE : dernière ligne "Elo:" (head=intervalle précoce, bug) ; ancrée sur ^Elo: pour NE PAS
  # matcher "nElo:". Champs awk : $2=Elo, $4=±err (virgule de fin retirée).
  local line=$(grep -E '^Elo:' "$OUT/$tag.log" | tail -1)
  local elo=$(echo "$line" | awk '{print $2}')
  local err=$(echo "$line" | awk '{gsub(/,/,"",$4); print $4}')
  printf "ANCHOR\t%s\t%s\tTSCP\t-\t%s\t%s\n" "$s1" "$t1" "${elo:-NA}" "${err:-NA}" >> "$OUT/results.tsv"
  echo "    → Elo(Nano-TSCP)=${elo:-NA} ±${err:-NA}"
}

echo "== Ancrage (sims 256, temp 0) vs TSCP sd6 =="
nt 256 0
echo "== Échelle par voisins (temp 0) =="
nn 8 0 16 0 LADDER
nn 16 0 32 0 LADDER
nn 32 0 64 0 LADDER
nn 64 0 128 0 LADDER
nn 128 0 256 0 LADDER
nn 256 0 512 0 LADDER
nn 512 0 1024 0 LADDER
echo "== Coût de la température (à 64 et 256 sims) =="
nn 64 0 64 0.15 TEMP
nn 64 0 64 0.30 TEMP
nn 256 0 256 0.15 TEMP
nn 256 0 256 0.30 TEMP

echo "=== terminé. Post-traitement : ==="
node "$HERE/calibrate-report.mjs" "$OUT/results.tsv" 1700
