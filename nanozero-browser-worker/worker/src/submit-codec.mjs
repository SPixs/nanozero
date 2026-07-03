// submit-codec.mjs — codec binaire de soumission self-play (Story A.1).
//
// Format compact pour le submit navigateur (alternative au JSON base64 du worker Java).
// Layout LITTLE-ENDIAN (cf. architecture-jobserver-connection.md, §Implementation Patterns) :
//   uint32 num_positions
//   par position :
//     float32       outcome
//     float32[7616] planes        (slice EXACT du Float32Array, pas .buffer brut)
//     uint16        nnz           (nb de coups non-nuls dans la policy)
//     nnz × { uint16 index<4672 ; float32 value }   (policy CREUSE)
//
// Le décodeur serveur (submit_codec.py) densifie la policy et reconstruit des bytes denses
// identiques au chemin JSON. Codec PUR : aucune I/O, testable en isolation.

export const PLANES_LEN = 7616;
export const POLICY_LEN = 4672;
const PLANES_BYTES = PLANES_LEN * 4;

// La parité bit-à-bit suppose une plateforme little-endian (tous les navigateurs le sont).
function assertLittleEndian() {
  if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) {
    throw new Error('submit-codec : plateforme non little-endian non supportée');
  }
}

// Extrait les (index, value) non-nuls d'une policy dense.
function sparsify(policy) {
  const idx = [];
  const val = [];
  for (let i = 0; i < POLICY_LEN; i++) {
    if (policy[i] !== 0) { idx.push(i); val.push(policy[i]); }
  }
  return { idx, val };
}

/**
 * Encode une liste de samples en ArrayBuffer binaire.
 * @param {Array<{planes:Float32Array, policy:Float32Array, outcome:number}>} samples
 * @returns {ArrayBuffer}
 */
export function encode(samples) {
  assertLittleEndian();
  // Valide les entrées AVANT d'encoder : une mauvaise taille de planes ferait lire hors de la
  // vue (corruption silencieuse / RangeError), un outcome non-fini ferait rejeter tout le batch
  // côté serveur. Mieux vaut lever ici, à la source.
  for (const s of samples) {
    if (s.planes.length !== PLANES_LEN) {
      throw new Error(`submit-codec : planes.length=${s.planes.length}, attendu ${PLANES_LEN}`);
    }
    if (s.policy.length !== POLICY_LEN) {
      throw new Error(`submit-codec : policy.length=${s.policy.length}, attendu ${POLICY_LEN}`);
    }
    if (!Number.isFinite(s.outcome)) {
      throw new Error('submit-codec : outcome non-fini');
    }
  }
  const sparse = samples.map((s) => sparsify(s.policy));
  let total = 4; // num_positions
  for (const sp of sparse) total += 4 + PLANES_BYTES + 2 + sp.idx.length * 6;

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let off = 0;
  dv.setUint32(off, samples.length, true); off += 4;

  for (let k = 0; k < samples.length; k++) {
    const s = samples[k];
    dv.setFloat32(off, s.outcome, true); off += 4;
    // planes : slice EXACT (gère byteOffset si Float32Array est une vue) ; bytes f32 LE.
    const p = s.planes;
    u8.set(new Uint8Array(p.buffer, p.byteOffset, PLANES_BYTES), off); off += PLANES_BYTES;
    const { idx, val } = sparse[k];
    dv.setUint16(off, idx.length, true); off += 2;
    for (let j = 0; j < idx.length; j++) {
      dv.setUint16(off, idx[j], true); off += 2;
      dv.setFloat32(off, val[j], true); off += 4;
    }
  }
  return buf;
}

/**
 * Décode un ArrayBuffer binaire en liste de samples (sert au round-trip de test).
 * @param {ArrayBuffer} buf
 * @returns {Array<{planes:Float32Array, policy:Float32Array, outcome:number}>}
 */
export function decode(buf) {
  assertLittleEndian();
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let off = 0;
  const n = dv.getUint32(off, true); off += 4;
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    const outcome = dv.getFloat32(off, true); off += 4;
    const planes = new Float32Array(PLANES_LEN);
    new Uint8Array(planes.buffer).set(u8.subarray(off, off + PLANES_BYTES)); off += PLANES_BYTES;
    const nnz = dv.getUint16(off, true); off += 2;
    const policy = new Float32Array(POLICY_LEN);
    for (let j = 0; j < nnz; j++) {
      const index = dv.getUint16(off, true); off += 2;
      const value = dv.getFloat32(off, true); off += 4;
      policy[index] = value;
    }
    out[k] = { planes, policy, outcome };
  }
  return out;
}
