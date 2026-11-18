import * as ort from 'onnxruntime-node';
import { runModel } from '../dsp.js';
const tensor = (t, d, s) => new ort.Tensor(t, d, s);
function signal(SEG) {
  const x = new Float32Array(2 * SEG);
  for (let i = 0; i < SEG; i++) {
    const t = i / 44100;
    const bass = 0.4 * Math.sin(2 * Math.PI * 55 * t) * (Math.floor(t * 2) % 2 ? 1 : 0.6);
    const lead = 0.25 * Math.sin(2 * Math.PI * (440 + 110 * Math.floor(t * 3 % 4)) * t) * Math.exp(-((t * 3) % 1) * 2);
    const kick = (i % 22050) < 2000 ? Math.sin(2 * Math.PI * 60 * (i % 22050) / 44100) * Math.exp(-(i % 22050) / 600) : 0;
    const hat = (i % 11025) < 300 ? (Math.random() - .5) * 0.3 : 0;
    x[i] = bass + lead + kick + hat; x[SEG + i] = bass + 0.7 * lead + kick + hat * 0.5;
  }
  let m = 0, v = 0; for (let i = 0; i < SEG; i++) m += (x[i] + x[SEG + i]) / 2; m /= SEG;
  for (let i = 0; i < SEG; i++) v += ((x[i] + x[SEG + i]) / 2 - m) ** 2; const sd = Math.sqrt(v / SEG);
  for (let i = 0; i < 2 * SEG; i++) x[i] = (x[i] - m) / sd;
  return x;
}
const names = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
const SEG = 343980, x = signal(SEG);
const k = await ort.InferenceSession.create('m/kramp.onnx');
const ref = (await k.run({ mix: tensor('float32', x, [1, 2, SEG]) })).stems.data;
const a78 = await ort.InferenceSession.create('m/aw_s78.onnx');
const t0 = Date.now(); const got = await runModel(a78, x, SEG, tensor); console.log('s78 pipeline ms', Date.now() - t0);
for (let s = 0; s < 6; s++) {
  let e = 0, p = 0, eMid = 0, pMid = 0;
  for (let i = 0; i < 2 * SEG; i++) { const j = s * 2 * SEG + i, d = got[j] - ref[j]; e += d * d; p += ref[j] * ref[j]; const ii = i % SEG; if (ii > 20000 && ii < SEG - 20000) { eMid += d * d; pMid += ref[j] * ref[j]; } }
  console.log(names[s].padEnd(7), 'SNR vs official', (10 * Math.log10(p / e)).toFixed(1), 'dB   (middle', (10 * Math.log10(pMid / eMid)).toFixed(1), 'dB)   energy', p.toExponential(2));
}
const a39 = await ort.InferenceSession.create('m/aw_s39.onnx');
const S2 = 171990, x2 = signal(S2), g2 = await runModel(a39, x2, S2, tensor);
let e = 0, p = 0; for (let c = 0; c < 2; c++) for (let i = 0; i < S2; i++) { let y = 0; for (let s = 0; s < 6; s++) y += g2[(s * 2 + c) * S2 + i]; const d = y - x2[c * S2 + i]; e += d * d; p += x2[c * S2 + i] ** 2; }
console.log('s39: stems sum vs input SNR', (10 * Math.log10(p / e)).toFixed(1), 'dB');
