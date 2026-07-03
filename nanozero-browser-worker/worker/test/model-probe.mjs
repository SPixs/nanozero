// Sonde le modèle gen-027 via onnxruntime-node : noms/shapes I/O + inférence sur zéros.
import * as ort from 'onnxruntime-node';

const session = await ort.InferenceSession.create('./gen-027-promoted.onnx');
console.log('inputs :', session.inputNames);
console.log('outputs:', session.outputNames);

const input = new ort.Tensor('float32', new Float32Array(119 * 8 * 8), [1, 119, 8, 8]);
const feeds = {}; feeds[session.inputNames[0]] = input;
const out = await session.run(feeds);
for (const name of session.outputNames) {
  const t = out[name];
  console.log(`  ${name} : dims=${JSON.stringify(t.dims)} dtype=${t.type} head=[${Array.from(t.data.slice(0, 6)).map((x) => x.toFixed(3)).join(', ')}]`);
}
