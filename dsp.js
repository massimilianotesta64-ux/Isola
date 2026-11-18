// STFT / inverse STFT matching HT-Demucs (_spec / _ispec):
// n_fft 4096, hop 1024, periodic Hann window, normalized, centre + reflect padding.
const N = 4096, HOP = 1024, BINS = 2048, LOG2N = 12;
const WIN = new Float32Array(N);
for (let i = 0; i < N; i++) WIN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
const COS = new Float64Array(N / 2), SIN = new Float64Array(N / 2);
for (let i = 0; i < N / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / N); SIN[i] = -Math.sin(2 * Math.PI * i / N); }
const REV = new Uint32Array(N);
for (let i = 0; i < N; i++) { let r = 0, x = i; for (let b = 0; b < LOG2N; b++) { r = (r << 1) | (x & 1); x >>= 1; } REV[i] = r; }

// in-place complex FFT (sign -1 forward, +1 inverse, unscaled)
function fft(re, im, inverse) {
  for (let i = 0; i < N; i++) { const j = REV[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, step = N / size;
    for (let st = 0; st < N; st += size) {
      for (let k = 0; k < half; k++) {
        const c = COS[k * step], s = inverse ? -SIN[k * step] : SIN[k * step];
        const a = st + k, b = a + half;
        const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

function reflectPad(x, l, r) {
  const n = x.length, out = new Float32Array(n + l + r);
  for (let i = 0; i < l; i++) out[i] = x[l - i];
  out.set(x, l);
  for (let i = 0; i < r; i++) out[l + n + i] = x[n - 2 - i];
  return out;
}

export const framesFor = len => Math.ceil(len / HOP);

// x: one channel, length L. Writes spec for channel c into out laid out as [2, 2048, T, 2].
export function stftInto(x, out, c, T) {
  const L = x.length, le = framesFor(L);
  if (le !== T) throw new Error('frame count mismatch');
  const pad = (HOP >> 1) * 3;
  const p1 = reflectPad(x, pad, pad + le * HOP - L);
  const p2 = reflectPad(p1, N >> 1, N >> 1);
  const re = new Float64Array(N), im = new Float64Array(N), scale = 1 / Math.sqrt(N);
  for (let t = 0; t < le; t++) {
    const s = (t + 2) * HOP;
    for (let i = 0; i < N; i++) { re[i] = p2[s + i] * WIN[i]; im[i] = 0; }
    fft(re, im, false);
    for (let f = 0; f < BINS; f++) {
      const o = ((c * BINS + f) * T + t) * 2;
      out[o] = re[f] * scale; out[o + 1] = im[f] * scale;
    }
  }
}

// freq: model output [6, 4, 2048, T] (4 = channel*2 + re/im). Returns time signal of length L
// for stem s, channel c.
export function istft(freq, s, c, T, L) {
  const frames = T + 4, outLen = N + HOP * (frames - 1);
  const y = new Float64Array(outLen), env = new Float64Array(outLen);
  const re = new Float64Array(N), im = new Float64Array(N), scale = Math.sqrt(N) / N;
  const bR = ((s * 4 + c * 2) * BINS) * T, bI = ((s * 4 + c * 2 + 1) * BINS) * T;
  for (let fr = 0; fr < frames; fr++) {
    const t = fr - 2, off = fr * HOP;
    for (let i = 0; i < N; i++) env[off + i] += WIN[i] * WIN[i];
    if (t < 0 || t >= T) continue;
    re.fill(0); im.fill(0);
    for (let f = 0; f < BINS; f++) { re[f] = freq[bR + f * T + t]; im[f] = freq[bI + f * T + t]; }
    // bin 2048 (Nyquist) is zero; mirror for a real signal
    for (let f = 1; f < BINS; f++) { re[N - f] = re[f]; im[N - f] = -im[f]; }
    im[0] = 0;
    fft(re, im, true);
    for (let i = 0; i < N; i++) y[off + i] += re[i] * scale * WIN[i];
  }
  const pad = (HOP >> 1) * 3, start = (N >> 1) + pad, out = new Float32Array(L);
  for (let i = 0; i < L; i++) { const k = start + i, e = env[k]; out[i] = e > 1e-11 ? y[k] / e : 0; }
  return out;
}

// One model call: spectrogram in, six stems (spectral branch + time branch) out.
export async function runModel(session, input, SEG, tensor) {
  const T = framesFor(SEG);
  const spec = new Float32Array(2 * 2048 * T * 2);
  stftInto(input.subarray(0, SEG), spec, 0, T);
  stftInto(input.subarray(SEG, 2 * SEG), spec, 1, T);
  const res = await session.run({
    mix: tensor('float32', input, [1, 2, SEG]),
    spec: tensor('float32', spec, [1, 2, 2048, T, 2])
  });
  const freq = res.freq.data, time = res.time.data;
  const out = new Float32Array(12 * SEG);
  for (let s = 0; s < 6; s++) for (let c = 0; c < 2; c++) {
    const y = istft(freq, s, c, T, SEG), b = (s * 2 + c) * SEG;
    for (let i = 0; i < SEG; i++) out[b + i] = y[i] + time[b + i];
  }
  return out;
}

export const DSP = { N, HOP, BINS };

