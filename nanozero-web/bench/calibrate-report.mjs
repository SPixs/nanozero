// calibrate-report.mjs — post-traitement de la calibration (cf. calibrate.sh).
// Reconstruit la courbe Elo absolue : échelle de sims chaînée par voisins (temp 0) + ancrage TSCP.
// Usage : node calibrate-report.mjs <results.tsv> [tscpElo=1700]
import { readFileSync } from 'node:fs';

const file = process.argv[2] || '/tmp/calib/results.tsv';
const TSCP_ELO = Number(process.argv[3] || 1700);
const rows = readFileSync(file, 'utf8').trim().split('\n').filter(Boolean)
  .map((l) => l.split('\t'))
  .map(([type, As, At, Bs, Bt, elo, err]) => ({ type, As: +As, At: +At, Bs, Bt, elo: parseFloat(elo), err: parseFloat(err) }));

const num = (x) => (Number.isFinite(x) ? x : null);

// 1) Échelle de sims (LADDER, temp 0) : elo = Elo(As) - Elo(Bs), As < Bs.
const ladder = rows.filter((r) => r.type === 'LADDER' && r.Bt !== undefined && +r.Bt === 0 && r.At === 0);
const simsSet = new Set();
ladder.forEach((r) => { simsSet.add(r.As); simsSet.add(+r.Bs); });
const sims = [...simsSet].sort((a, b) => a - b);
const rel = new Map(); const relErr = new Map();
if (sims.length) { rel.set(sims[0], 0); relErr.set(sims[0], 0); }
for (let i = 0; i < sims.length - 1; i++) {
  const lo = sims[i], hi = sims[i + 1];
  const m = ladder.find((r) => r.As === lo && +r.Bs === hi);
  if (!m || num(m.elo) === null) { rel.set(hi, null); relErr.set(hi, null); continue; }
  const baseR = rel.get(lo), baseE = relErr.get(lo) ?? 0;
  // elo = rel(lo) - rel(hi)  →  rel(hi) = rel(lo) - elo
  rel.set(hi, baseR === null ? null : baseR - m.elo);
  relErr.set(hi, baseR === null ? null : Math.hypot(baseE, num(m.err) ?? 0));
}

// 2) Ancrage : abs(256) = TSCP_ELO + Elo(Nano256 - TSCP). offset = abs(anchorSims) - rel(anchorSims).
const anchor = rows.find((r) => r.type === 'ANCHOR');
let offset = null, anchorAbs = null, anchorErr = 0;
if (anchor && num(anchor.elo) !== null && rel.get(anchor.As) != null) {
  anchorAbs = TSCP_ELO + anchor.elo;
  offset = anchorAbs - rel.get(anchor.As);
  anchorErr = num(anchor.err) ?? 0;
}

const absElo = (s) => (offset === null || rel.get(s) == null ? null : Math.round(rel.get(s) + offset));
const absErr = (s) => (relErr.get(s) == null ? null : Math.round(Math.hypot(relErr.get(s), anchorErr)));

console.log('\n=== Courbe Elo = f(sims), temp 0 (ancrée sur TSCP=' + TSCP_ELO + ') ===');
console.log('sims\tEloΔvoisin\tElo abs\t±');
for (const s of sims) {
  const a = absElo(s);
  console.log(`${s}\t${rel.get(s) == null ? 'NA' : Math.round(rel.get(s))}\t${a == null ? 'NA' : a}\t${absErr(s) ?? ''}`);
}
if (anchorAbs != null) console.log(`(ancrage: Nano@${anchor.As} sims = ${Math.round(anchorAbs)} Elo, soit ${anchor.elo >= 0 ? '+' : ''}${anchor.elo} vs TSCP)`);

// 3) Coût de la température (TEMP : Elo(temp0) - Elo(tempX) à sims fixe).
const temps = rows.filter((r) => r.type === 'TEMP');
if (temps.length) {
  console.log('\n=== Coût de la température (Elo perdu vs temp 0) ===');
  for (const t of temps) console.log(`${t.As} sims, temp ${t.Bt} : ${num(t.elo) === null ? 'NA' : (t.elo >= 0 ? '-' + t.elo : '+' + (-t.elo))} Elo`);
}

// 4) Reco sims par cible (interpolation en log2(sims)).
function simsForElo(target) {
  const pts = sims.map((s) => [Math.log2(s), absElo(s)]).filter((p) => p[1] != null);
  if (pts.length < 2) return null;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
    if ((target >= y0 && target <= y1) || (target <= y0 && target >= y1)) {
      const x = x0 + (target - y0) * (x1 - x0) / (y1 - y0);
      return Math.round(2 ** x);
    }
  }
  // hors gamme : extrapole avec la pente globale
  const [xa, ya] = pts[0], [xb, yb] = pts[pts.length - 1];
  const slope = (yb - ya) / (xb - xa);
  const x = xa + (target - ya) / slope;
  return { sims: Math.round(2 ** x), extrapolated: true };
}
console.log('\n=== Reco sims par cible (temp 0 ; ajouter le coût température si temp>0) ===');
for (const [label, target] of [['Tranquille', 1200], ['Joueur de club', 1600]]) {
  const r = simsForElo(target);
  const txt = r == null ? 'indéterminé' : (typeof r === 'object' ? `~${r.sims} sims (EXTRAPOLÉ, hors gamme mesurée)` : `~${r} sims`);
  console.log(`${label} (${target}) → ${txt}`);
}
const top = sims[sims.length - 1];
console.log(`Pleine force (max) → sims max testé = ${top} → ${absElo(top) ?? 'NA'} Elo (plafond 8×96 ; au-delà = latence)`);
