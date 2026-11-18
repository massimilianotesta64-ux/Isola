importScripts('lib/lame.min.js');
var encs = {}, parts = {}, kbps = 192;
function load() { return Promise.resolve(); }
function toI16(f) {
  var o = new Int16Array(f.length);
  for (var i = 0; i < f.length; i++) { var s = f[i]; s = s > 1 ? 1 : s < -1 ? -1 : s; o[i] = s < 0 ? s * 32768 : s * 32767; }
  return o;
}
function encode(key, l, r) {
  var enc = encs[key], li = toI16(l), ri = toI16(r), B = 1152 * 32;
  for (var i = 0; i < li.length; i += B) {
    var mp3 = enc.encodeBuffer(li.subarray(i, i + B), ri.subarray(i, i + B));
    if (mp3.length) parts[key].push(mp3);
  }
}
self.onmessage = function (e) {
  var m = e.data;
  load().then(function () {
    if (m.type === 'start') {
      kbps = m.kbps || 192;
      m.keys.forEach(function (k) { encs[k] = new lamejs.Mp3Encoder(2, 44100, kbps); parts[k] = []; });
    } else if (m.type === 'pcm') {
      encode(m.key, m.l, m.r);
    } else if (m.type === 'finish') {
      var blobs = {};
      Object.keys(encs).forEach(function (k) {
        var f = encs[k].flush(); if (f.length) parts[k].push(f);
        blobs[k] = new Blob(parts[k], { type: 'audio/mpeg' });
      });
      encs = {}; parts = {};
      self.postMessage({ type: 'done', blobs: blobs });
    }
  }).catch(function (err) { self.postMessage({ type: 'error', message: String(err && err.message || err) }); });
};
