import { chromium } from 'playwright';
import fs from 'fs';
// 30 s stereo test song
const sr = 44100, n = sr * 30, b = Buffer.alloc(44 + n * 4);
b.write('RIFF', 0); b.writeUInt32LE(36 + n * 4, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(2, 22);
b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 4, 40);
for (let i = 0; i < n; i++) { const t = i / sr; const v = 0.3 * Math.sin(2 * Math.PI * 110 * t) + 0.2 * Math.sin(2 * Math.PI * 440 * t) * (Math.floor(t * 4) % 2) + (i % 22050 < 800 ? (Math.random() - .5) * 0.6 : 0); b.writeInt16LE(Math.round(v * 20000), 44 + i * 4); b.writeInt16LE(Math.round(v * 18000), 46 + i * 4); }
fs.writeFileSync('song.wav', b);
const URL = 'https://massimilianotesta64-ux.github.io/Isola/';
const browser = await chromium.launch({ args: ['--enable-features=SharedArrayBuffer'] });
const page = await (await browser.newContext()).newPage();
const log = m => console.log(new Date().toISOString().slice(11, 19), m);
page.on('console', m => log('console.' + m.type() + ': ' + m.text()));
page.on('pageerror', e => log('PAGEERROR ' + e.message));
page.on('worker', w => { log('worker started ' + w.url()); w.on('close', () => log('worker closed ' + w.url())); });
await page.goto(URL); await page.waitForTimeout(3000);
await page.reload(); await page.waitForTimeout(2000);
log('crossOriginIsolated=' + await page.evaluate('crossOriginIsolated') + ' cores=' + await page.evaluate('navigator.hardwareConcurrency') + ' version=' + await page.evaluate('VERSION'));
await page.setInputFiles('#file', 'song.wav');
await page.waitForSelector('text=Split now', { timeout: 60000 });
await page.click('text=Split now');
let last = '';
const t0 = Date.now();
while (Date.now() - t0 < 1400000) {
  await page.waitForTimeout(3000);
  const st = await page.evaluate(() => [document.querySelector('#proc-step')?.textContent, document.querySelector('#proc-detail')?.textContent, document.querySelector('#proc-pct')?.textContent, !document.querySelector('#v-play').hidden, document.querySelector('.sheet')?.innerText || ''].join(' | '));
  if (st !== last) { log(st); last = st; }
  if (st.includes('didn’t finish') || st.split(' | ')[3] === 'true') break;
}
log('elapsed ' + Math.round((Date.now() - t0) / 1000) + ' s');
await browser.close();
