// nn-node.mjs — chargeur NeuralNet via onnxruntime-node (tests/dev hors navigateur).
// Au runtime navigateur, on créera la session via onnxruntime-web (WebGPU) avec la même interface.
import * as ort from 'onnxruntime-node';
import { NeuralNet } from './nn.mjs';

export async function loadNet(modelPath) {
  const session = await ort.InferenceSession.create(modelPath, { graphOptimizationLevel: 'all' });
  return new NeuralNet(session, { Tensor: ort.Tensor });
}
