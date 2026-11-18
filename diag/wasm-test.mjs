import * as ort from 'onnxruntime-web';
const f = process.argv[2];
const rss = () => (process.memoryUsage().rss / 1048576).toFixed(0) + 'MB';
ort.env.wasm.numThreads = 1;
for (const opt of ['all', 'disabled']) {
  try {
    const t0 = Date.now();
    const s = await ort.InferenceSession.create(f, { executionProviders: ['wasm'], graphOptimizationLevel: opt, enableCpuMemArena: false, enableMemPattern: false });
    console.log(`[${opt}] session ok ${Date.now() - t0}ms rss=${rss()} in=${s.inputNames} out=${s.outputNames}`);
    const meta = s.inputMetadata || [];
    const feeds = {};
    for (let i = 0; i < s.inputNames.length; i++) {
      const md = meta[i] || {};
      let shape = (md.shape || []).map(d => typeof d === 'number' && d > 0 ? d : 1);
      if (!shape.length) { console.log('  no shape metadata; skipping run'); shape = null; }
      if (shape) { const n = shape.reduce((a, b) => a * b, 1); const a = new Float32Array(n); for (let k = 0; k < n; k++) a[k] = Math.sin(k * 0.01) * 0.3; feeds[s.inputNames[i]] = new ort.Tensor(md.type || 'float32', a, shape); console.log('  input', s.inputNames[i], shape, md.type); }
    }
    if (Object.keys(feeds).length === s.inputNames.length) {
      const t1 = Date.now();
      const r = await s.run(feeds);
      for (const k of s.outputNames) console.log(`  out ${k} dims=${r[k].dims}`);
      console.log(`[${opt}] RUN OK ${Date.now() - t1}ms rss=${rss()}`);
    }
    await s.release();
    break;
  } catch (e) { console.log(`[${opt}] FAILED ${String(e.message || e).slice(0, 200)} rss=${rss()}`); }
}
process.exit(0);
