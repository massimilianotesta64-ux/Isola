import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.wasm.min.mjs';
import { runModel } from './dsp.js';

const CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
// HT-Demucs 6-source (Meta, MIT) exported for browsers by AudioWeb: fp16 weights, chunked attention.
const MODELS = {
  light: { seg: 171990, file: 'demucs_htdemucs_6s_s39_fp16w_ca.onnx', mb: 57 },
  full:  { seg: 343980, file: 'demucs_htdemucs_6s_s78_fp16w_ca.onnx', mb: 55 }
};
const BASE = 'https://huggingface.co/Andremi/audioweb-models/resolve/main/';
const CACHE_NAME = 'isola-model-v2';
const post = (m, t) => self.postMessage(m, t || []);

// ---- shared overlap-add core (also unit-tested) ----
export async function separate(L, R, SEG, runChunk, emit, progress) {
  const N = L.length;
  const OVERLAP = Math.floor(SEG / 4), STRIDE = SEG - OVERLAP, ROWS = 12;
  // global normalisation, as in the official Demucs pipeline
  let sum = 0;
  for (let i = 0; i < N; i++) sum += (L[i] + R[i]) * 0.5;
  const mean = sum / N;
  let vs = 0;
  for (let i = 0; i < N; i++) { const d = (L[i] + R[i]) * 0.5 - mean; vs += d * d; }
  let std = Math.sqrt(vs / Math.max(1, N - 1));
  if (!(std > 1e-8)) std = 1;
  // triangular transition window
  const W = new Float32Array(SEG), half = SEG >> 1;
  for (let i = 0; i < SEG; i++) W[i] = i < half ? (i + 1) / half : (SEG - i) / half;
  const offs = [];
  for (let off = 0; ; off += STRIDE) { offs.push(off); if (off + SEG >= N) break; }
  const acc = new Float32Array(ROWS * SEG), wsum = new Float32Array(SEG), input = new Float32Array(2 * SEG);
  for (let k = 0; k < offs.length; k++) {
    const off = offs[k], len = Math.min(SEG, N - off), last = k === offs.length - 1;
    input.fill(0);
    for (let i = 0; i < len; i++) { input[i] = (L[off + i] - mean) / std; input[SEG + i] = (R[off + i] - mean) / std; }
    const out = await runChunk(input, k);
    for (let r = 0; r < ROWS; r++) {
      const b = r * SEG;
      for (let i = 0; i < len; i++) acc[b + i] += out[b + i] * W[i];
    }
    for (let i = 0; i < len; i++) wsum[i] += W[i];
    const emitLen = last ? len : STRIDE;
    const rows = [];
    for (let r = 0; r < ROWS; r++) {
      const b = r * SEG, o = new Float32Array(emitLen);
      for (let i = 0; i < emitLen; i++) o[i] = acc[b + i] / Math.max(wsum[i], 1e-8) * std + mean;
      rows.push(o);
    }
    emit(rows, k);
    if (!last) {
      for (let r = 0; r < ROWS; r++) {
        const b = r * SEG;
        acc.copyWithin(b, b + STRIDE, b + SEG);
        acc.fill(0, b + SEG - STRIDE, b + SEG);
      }
      wsum.copyWithin(0, STRIDE, SEG);
      wsum.fill(0, SEG - STRIDE, SEG);
    }
    progress(k + 1, offs.length);
  }
}

async function getModel(m) {
  const key = self.location.origin + '/__model__/' + m.file;
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    caches.delete('isola-model-v1').catch(() => {}); // old, browser-incompatible model
  } catch (e) {}
  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) { post({ type: 'status', text: 'Loading the AI model…' }); return new Uint8Array(await hit.arrayBuffer()); }
    } catch (e) {}
  }
  post({ type: 'status', text: 'Downloading the AI model (first time only)…' });
  const res = await fetch(BASE + m.file, { mode: 'cors', credentials: 'omit' });
  if (!res.ok) throw new Error('Could not download the AI model (HTTP ' + res.status + '). Check your internet connection.');
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const parts = []; let got = 0, t0 = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    const now = Date.now();
    if (now - t0 > 250) { t0 = now; post({ type: 'download', got, total }); }
  }
  const bytes = new Uint8Array(got); let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  if (bytes.length < 40e6) throw new Error('The AI model download looks incomplete. Please try again.');
  if (cache) { try { await cache.put(key, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) {} }
  return bytes;
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type !== 'run') return;
  let stage = 'starting';
  try {
    ort.env.wasm.wasmPaths = CDN;
    const cores = (self.navigator && navigator.hardwareConcurrency) || 2;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
    const model = MODELS[m.model] || MODELS.light;
    stage = 'loading the AI model';
    let bytes = await getModel(model);
    stage = 'starting the splitter';
    post({ type: 'status', text: 'Starting the splitter…' });
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false
    });
    bytes = null;
    const th = ort.env.wasm.numThreads;
    post({ type: 'engine', ep: 'wasm', label: (m.model === 'full' ? 'Full model' : 'Light model') + ' · ' + th + (th > 1 ? ' threads' : ' thread') });
    stage = 'splitting';
    const tensor = (t, d, s) => new ort.Tensor(t, d, s);
    await separate(m.L, m.R, model.seg, async (input) => {
      const t = performance.now();
      const out = await runModel(session, input, model.seg, tensor);
      post({ type: 'chunktime', ms: performance.now() - t });
      return out;
    },
      (rows, k) => post({ type: 'block', k, rows }, rows.map(r => r.buffer)),
      (done, total) => post({ type: 'progress', done, total }));
    try { await session.release(); } catch (e) {}
    post({ type: 'done', ep: 'wasm' });
  } catch (e) {
    const msg = (e && e.message) || String(e);
    post({ type: 'error', message: 'Error while ' + stage + ': ' + msg });
  }
};
