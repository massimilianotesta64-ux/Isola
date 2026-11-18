import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.webgpu.min.mjs';
const CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
const MODEL_URLS = [
  'https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx',
  'https://huggingface.co/adowu/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx'
];
const CACHE_NAME = 'isola-model-v1';
const CACHE_KEY = self.location.origin + '/__model__/htdemucs_6s_fp16weights.onnx';
const SEG = 343980;
const post = (m, t) => self.postMessage(m, t || []);

// ---- shared overlap-add core (also unit-tested) ----
async function separate(L, R, runChunk, emit, progress) {
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

async function getModel() {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch (e) {}
  if (cache) {
    try {
      const hit = await cache.match(CACHE_KEY);
      if (hit) { post({ type: 'status', text: 'Loading the AI model…' }); return new Uint8Array(await hit.arrayBuffer()); }
    } catch (e) {}
  }
  let lastErr = null;
  for (const url of MODEL_URLS) {
    try {
      post({ type: 'status', text: 'Downloading the AI model (first time only)…' });
      const res = await fetch(url, { mode: 'cors', credentials: 'omit' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const total = +res.headers.get('content-length') || 0;
      const reader = res.body.getReader();
      let buf = total ? new Uint8Array(total) : null, parts = [], got = 0, t0 = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (buf && got + value.length <= total) buf.set(value, got);
        else { if (buf) { parts.push(buf.slice(0, got)); buf = null; } parts.push(value); }
        got += value.length;
        const now = Date.now();
        if (now - t0 > 250) { t0 = now; post({ type: 'download', got, total }); }
      }
      let bytes;
      if (buf && got === total) bytes = buf;
      else {
        if (buf) parts.unshift(buf.slice(0, got));
        bytes = new Uint8Array(got); let o = 0;
        for (const p of parts) { bytes.set(p, o); o += p.length; }
      }
      if (bytes.length < 50e6) throw new Error('Model download looks incomplete');
      if (cache) { try { await cache.put(CACHE_KEY, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) {} }
      return bytes;
    } catch (e) { lastErr = e; }
  }
  throw new Error('Could not download the AI model. Check your internet connection. (' + (lastErr && lastErr.message) + ')');
}

async function makeSession(bytes, ep) {
  const opts = ep === 'webgpu'
    ? { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' }
    : { executionProviders: ['wasm'], graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false };
  return ort.InferenceSession.create(bytes, opts);
}

async function runWith(session, input) {
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor('float32', input, [1, 2, SEG]);
  const res = await session.run(feeds);
  const t = res[session.outputNames[0]];
  let d = t.data;
  if (t.location && t.location !== 'cpu') d = await t.getData(true);
  return d;
}

function looksValid(input, out) {
  // the six stems should add up (roughly) to the input mix
  let e = 0, s = 0;
  for (let c = 0; c < 2; c++) for (let i = 0; i < SEG; i += 7) {
    let y = 0;
    for (let st = 0; st < 6; st++) y += out[(st * 2 + c) * SEG + i];
    if (!Number.isFinite(y)) return false;
    const x = input[c * SEG + i]; e += (y - x) * (y - x); s += x * x;
  }
  return s < 1e-6 || e / s < 0.5;
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type !== 'run') return;
  let stage = 'starting';
  try {
    ort.env.wasm.wasmPaths = CDN;
    const cores = (self.navigator && navigator.hardwareConcurrency) || 2;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
    stage = 'loading the AI model';
    const bytes = await getModel();
    stage = 'starting the splitter';
    post({ type: 'status', text: 'Starting the splitter…' });
    let session = null, ep = 'wasm';
    // The 6-stem model contains an op the WebGPU backend can't run, and a failed
    // WebGPU attempt can crash the whole engine, so only the processor route is used.
    if (false) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) { session = await makeSession(bytes, 'webgpu'); ep = 'webgpu'; }
      } catch (e) { session = null; }
    }
    if (!session) { session = await makeSession(bytes, 'wasm'); ep = 'wasm'; }
    const engineLabel = () => ep === 'webgpu' ? 'Graphics chip (WebGPU)' : ('Processor · ' + ort.env.wasm.numThreads + (ort.env.wasm.numThreads > 1 ? ' threads' : ' thread'));
    post({ type: 'engine', ep, label: engineLabel() });
    let modelBytes = bytes;
    stage = 'splitting';
    const runChunk = async (input, k) => {
      const t = performance.now();
      let out;
      try { out = await runWith(session, input); }
      catch (e) {
        if (ep !== 'webgpu') throw e;
        out = null;
      }
      if (k === 0 && ep === 'webgpu' && (!out || !looksValid(input, out))) {
        try { await session.release(); } catch (e) {}
        session = await makeSession(modelBytes, 'wasm'); ep = 'wasm';
        post({ type: 'engine', ep, label: engineLabel() });
        out = await runWith(session, input);
      }
      if (k === 0) modelBytes = null;
      post({ type: 'chunktime', ms: performance.now() - t });
      return out;
    };
    await separate(m.L, m.R, runChunk,
      (rows, k) => post({ type: 'block', k, rows }, rows.map(r => r.buffer)),
      (done, total) => post({ type: 'progress', done, total }));
    try { await session.release(); } catch (e) {}
    post({ type: 'done', ep });
  } catch (e) {
    const msg = (e && e.message) || String(e);
    post({ type: 'error', message: 'Error while ' + (typeof stage === 'string' ? stage : 'starting') + ': ' + msg + ' [threads: ' + ort.env.wasm.numThreads + ', isolated: ' + self.crossOriginIsolated + ']' });
  }
};
