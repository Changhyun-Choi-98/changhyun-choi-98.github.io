"use strict";

const CACHE_PREFIX = "image-reveal-game-";
const CACHE_NAME = `${CACHE_PREFIX}v1`;
const ASSETS = [
  "./",
  "./index.html",
  "./display.html",
  "./leaderboard.html",
  "./help.html",
  "./offline.html",
  "./manifest.webmanifest",
  "./css/app.css",
  "./js/domain.js",
  "./js/state-machine.js",
  "./js/ranking.js",
  "./js/image-loader.js",
  "./js/renderer.js",
  "./js/persistence.js",
  "./js/sync.js",
  "./js/app.js",
  "./js/display.js",
  "./js/leaderboard.js",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-maskable-512.png"
];

function absoluteAssetUrls() {
  return ASSETS.map((asset) => new URL(asset, self.registration.scope).href);
}

async function cacheShell() {
  const cache = await caches.open(CACHE_NAME);
  const requests = absoluteAssetUrls().map((url) => new Request(url, { cache: "reload", credentials: "same-origin" }));
  const responses = await Promise.all(requests.map((request) => fetch(request)));
  if (responses.some((response) => !response.ok)) throw new Error("앱 파일 일부를 내려받지 못했습니다.");
  await Promise.all(responses.map((response, index) => cache.put(requests[index], response)));
}

async function isCacheReady() {
  const cache = await caches.open(CACHE_NAME);
  const matches = await Promise.all(absoluteAssetUrls().map((url) => cache.match(url)));
  return matches.every(Boolean);
}

async function notifyClient(client, type) {
  if (client && typeof client.postMessage === "function") client.postMessage({ type });
}

self.addEventListener("install", (event) => {
  event.waitUntil(cacheShell().catch(async (error) => {
    const clients = await self.clients.matchAll({ includeUncontrolled: true });
    await Promise.all(clients.map((client) => notifyClient(client, "CACHE_ERROR")));
    throw error;
  }));
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME).map((name) => caches.delete(name)));
    await self.clients.claim();
    const ready = await isCacheReady();
    const clients = await self.clients.matchAll({ includeUncontrolled: true });
    await Promise.all(clients.map((client) => notifyClient(client, ready ? "CACHE_READY" : "CACHE_ERROR")));
  })());
});

self.addEventListener("message", (event) => {
  if (!event.data) return;
  if (event.data.type === "SKIP_WAITING") self.skipWaiting();
  if (event.data.type === "CHECK_CACHE") {
    event.waitUntil(isCacheReady().then((ready) => notifyClient(event.source, ready ? "CACHE_READY" : "CACHE_ERROR")));
  }
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  const scope = new URL(self.registration.scope);
  if (url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  const allowed = new Set(absoluteAssetUrls().map((asset) => new URL(asset).pathname));
  const normalizedPath = url.pathname.endsWith("/reveal-game/") ? `${url.pathname}index.html` : url.pathname;
  if (!allowed.has(url.pathname) && !allowed.has(normalizedPath)) return;

  if (request.mode === "navigate") {
    event.respondWith((async () => {
      try {
        const response = await fetch(new Request(request, { cache: "no-store" }));
        if (response.ok) {
          const cache = await caches.open(CACHE_NAME);
          await cache.put(request, response.clone());
        }
        return response;
      } catch (error) {
        const cache = await caches.open(CACHE_NAME);
        return (await cache.match(request)) || (await cache.match(new URL("./index.html", self.registration.scope).href));
      }
    })());
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) {
      await cache.put(request, response.clone());
    }
    return response;
  })());
});
