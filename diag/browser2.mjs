import { chromium } from 'playwright';
import fs from 'fs';
const sr = 44100, n = sr * 40, b = Buffer.alloc(44 + n * 4);
b.write('RIFF', 0); b.writeUInt32LE(36 + n * 4, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(2, 22);
b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 4, 40);
for (let i = 0; i < n; i++) { const t = i / sr; const v = 0.3 * Math.sin(2 * Math.PI * 55 * t) + 0.2 * Math.sin(2 * Math.PI * 440 * t) * (Math.floor(t * 4) % 2) + ((i % 22050) < 2000 ? Math.sin(2 * Math.PI * 60 * (i % 22050) / sr) * Math.exp(-(i % 22050) / 600) * 0.6 : 0); b.writeInt16LE(Math.round(v * 20000), 44 + i * 4); b.writeInt16LE(Math.round(v * 18000), 46 + i * 4); }
fs.writeFileSync('song.wav', b);
const browser = await chromium.launch();
for (const model of ['light', 'full']) {
  const page = await (await browser.newContext()).newPage();
  const log = m => console.log(`[${model}] ` + new Date().toISOString().slice(11, 19), m);
  page.on('console', m => { if (!/cpuid_info/.test(m.text())) log('console.' + m.type() + ': ' + m.text()); });
  page.on('pageerror', e => log('PAGEERROR ' + e.message));
  await page.goto('http://127.0.0.1:8080/'); await page.waitForTimeout(1000);
  await page.evaluate(m => { localStorage.setItem('isola-settings', JSON.stringify({ model: m, kbps: 192 })); }, model);
  await page.reload(); await page.waitForTimeout(1000);
  log('isolated=' + await page.evaluate('crossOriginIsolated') + ' version=' + await page.evaluate('VERSION'));
  await page.setInputFiles('#file', 'song.wav');
  await page.waitForSelector('text=Split now', { timeout: 60000 });
  await page.click('text=Split now');
  let last = ''; const t0 = Date.now(); let ok = false;
  while (Date.now() - t0 < 900000) {
    await page.waitForTimeout(3000);
    const st = await page.evaluate(() => [document.querySelector('#proc-step')?.textContent, document.querySelector('#proc-detail')?.textContent, document.querySelector('#proc-pct')?.textContent, !document.querySelector('#v-play').hidden, document.querySelector('.sheet')?.innerText || ''].join(' | '));
    if (st !== last) { log(st.replace(/\n/g, ' ')); last = st; }
    if (st.includes('didn’t finish')) break;
    if (st.split(' | ')[3] === 'true') { ok = true; break; }
  }
  log('elapsed ' + Math.round((Date.now() - t0) / 1000) + ' s, ok=' + ok);
  if (ok) {
    await page.waitForTimeout(1500);
    const r = await page.evaluate(() => {
      const out = {};
      for (const s of STEMS) { const d = E.buffers[s.key].getChannelData(0); let q = 0; for (let i = 44100 * 5; i < 44100 * 35; i++) q += d[i] * d[i]; out[s.key] = Math.sqrt(q / (44100 * 30)).toFixed(4); }
      return out;
    });
    log('stem RMS ' + JSON.stringify(r));
  }
  await page.context().close();
}
await browser.close();
