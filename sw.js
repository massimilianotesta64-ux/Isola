// Isola service worker: works offline and enables multi-threaded splitting
// (adds the cross-origin isolation headers GitHub Pages can't set itself).
const APP_CACHE = 'isola-app-v3';
const LIB_CACHE = 'isola-libs-v1';
const CORE = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './sep-worker.js', './enc-worker.js', './lib/lame.min.js', './lib/SignalsmithStretch.mjs'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(APP_CACHE).then(c => c.addAll(CORE)).catch(() => {}).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k.startsWith('isola-') && k !== APP_CACHE && k !== LIB_CACHE && k !== 'isola-model-v1') await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

function isolate(resp) {
  if (!resp || resp.status === 0 || resp.type === 'opaque') return resp;
  const h = new Headers(resp.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // the app itself: network first (so updates arrive), cache when offline
  if (url.origin === self.location.origin) {
    e.respondWith((async () => {
      try {
        const res = await fetch(req);
        if (res.ok && res.type === 'basic') {
          const copy = res.clone();
          caches.open(APP_CACHE).then(c => c.put(req, copy)).catch(() => {});
        }
        return isolate(res);
      } catch (err) {
        const hit = await caches.match(req, { ignoreSearch: true }) ||
          (req.mode === 'navigate' ? await caches.match('./index.html') : null);
        return hit ? isolate(hit) : Response.error();
      }
    })());
    return;
  }

  // libraries from jsDelivr: cache first (they're version-pinned)
  if (url.hostname === 'cdn.jsdelivr.net' && req.mode === 'cors') {
    e.respondWith((async () => {
      const cache = await caches.open(LIB_CACHE);
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone()).catch(() => {});
      return res;
    })());
  }
  // everything else (e.g. the AI model download) goes straight to the network
});
