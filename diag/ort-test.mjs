const SEG = 343980;
const mem = () => (process.memoryUsage().rss / 1048576).toFixed(0) + ' MB rss';
const input = new Float32Array(2 * SEG);
for (let i = 0; i < SEG; i++) { input[i] = Math.sin(i * 0.01) * 0.5 + (Math.random() - .5) * 0.3; input[SEG + i] = Math.sin(i * 0.011) * 0.5; }
async function tryOrt(label, ort, opts) {
  console.log('=== ' + label);
  try {
    const t0 = Date.now();
    const s = await ort.InferenceSession.create('model.onnx', opts);
    console.log('session ok', Date.now() - t0, 'ms', mem(), 'in', s.inputNames, 'out', s.outputNames);
    const t1 = Date.now();
    const r = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, [1, 2, SEG]) });
    const o = r[s.outputNames[0]];
    let e = 0, x2 = 0;
    for (let c = 0; c < 2; c++) for (let i = 0; i < SEG; i += 5) { let y = 0; for (let st = 0; st < 6; st++) y += o.data[(st * 2 + c) * SEG + i]; const x = input[c * SEG + i]; e += (y - x) ** 2; x2 += x * x; }
    console.log('run ok', Date.now() - t1, 'ms dims', o.dims, 'recon err', (e / x2).toFixed(4), mem());
  } catch (err) { console.log('FAILED', err && err.stack || err); }
}
const node = await import('onnxruntime-node');
await tryOrt('onnxruntime-node cpu', node, { executionProviders: ['cpu'] });
const web = await import('onnxruntime-web');
web.env.wasm.numThreads = 1;
await tryOrt('onnxruntime-web wasm 1 thread, app options', web, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false });
await tryOrt('onnxruntime-web wasm 1 thread, basic opt', web, { executionProviders: ['wasm'], graphOptimizationLevel: 'basic' });
