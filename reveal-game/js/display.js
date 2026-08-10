(function initDisplay(global) {
  "use strict";

  const RG = global.RevealGame;
  const { Domain, Renderer, Sync } = RG;
  const canvas = document.getElementById("displayCanvas");
  const connection = document.getElementById("displayConnection");
  const round = document.getElementById("displayRound");
  const timer = document.getElementById("displayTimer");
  const notice = document.getElementById("displayNotice");
  const fullscreen = document.getElementById("displayFullscreen");
  let latest = null;
  let receivedAt = 0;
  let lastContactAt = 0;
  let currentImageId = null;
  let pendingImageId = null;
  let currentUrl = null;
  let decoded = null;
  let animationFrame = 0;
  let loadToken = 0;
  let connectionTimer = 0;

  const participantToken = new URLSearchParams(global.location.hash.slice(1)).get("participant") || "";
  const renderer = new Renderer.CanvasRenderer(canvas, { onResize: () => render(performance.now()) });
  const channel = new Sync.CrossWindowChannel(`image-reveal-game-session-v1:${participantToken || "unpaired"}`, handleMessage);

  function publish(payload) {
    channel.publish(payload, { target: global.opener && !global.opener.closed ? global.opener : undefined, scalar: true });
  }

  function clearImage() {
    loadToken += 1;
    pendingImageId = null;
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    if (decoded && decoded.image) decoded.image.removeAttribute("src");
    currentUrl = null;
    currentImageId = null;
    decoded = null;
    renderer.setImage(null);
  }

  function handleMessage(payload) {
    if (!payload) return;
    if (payload.type === "snapshot") {
      const sameController = latest && payload.controllerId && payload.controllerId === latest.controllerId;
      if (sameController && Number(payload.sequence) < Number(latest.sequence)) return;
      lastContactAt = performance.now();
      latest = payload;
      receivedAt = performance.now();
      if (!payload.imageId && (decoded || pendingImageId)) clearImage();
      round.textContent = payload.total ? `${payload.round} / ${payload.total} 라운드` : "0 / 0 라운드";
      connection.textContent = "진행자 화면 연결됨";
      notice.hidden = Boolean(decoded && (!payload.imageId || payload.imageId === currentImageId));
      if (!notice.hidden) notice.textContent = payload.imageId ? "새 이미지를 준비하고 있습니다." : "진행자 화면에서 이미지를 준비하고 있습니다.";
      ensureAnimation();
    } else if (payload.type === "image" && payload.blob instanceof Blob && latest && payload.imageId === latest.imageId
      && payload.imageId !== currentImageId && payload.imageId !== pendingImageId) {
      loadImage(payload.imageId, payload.blob);
    }
  }

  async function loadImage(imageId, blob) {
    const token = ++loadToken;
    pendingImageId = imageId;
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.decoding = "async";
    try {
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error("참가자 화면에서 이미지를 해석하지 못했습니다."));
        image.src = url;
      });
      if (token !== loadToken || !latest || latest.imageId !== imageId) { URL.revokeObjectURL(url); return; }
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl = url;
      currentImageId = imageId;
      decoded = { image, width: image.naturalWidth, height: image.naturalHeight };
      renderer.setImage(decoded);
      notice.hidden = true;
      render(performance.now());
    } catch (error) {
      URL.revokeObjectURL(url);
      if (token !== loadToken) return;
      notice.hidden = false;
      notice.textContent = "이미지를 표시하지 못했습니다. 진행자에게 참가자 창을 다시 열어 달라고 요청해 주세요.";
    } finally {
      if (token === loadToken) pendingImageId = null;
    }
  }

  function interpolatedElapsed(now) {
    if (!latest) return 0;
    const connected = now - lastContactAt < 2000;
    const drift = latest.running && connected ? Math.max(0, now - receivedAt) : 0;
    return Math.min(latest.durationMs || Infinity, Math.round((latest.elapsedMs || 0) + drift));
  }

  function render(now) {
    if (!latest) { renderer.render(); return; }
    const elapsed = interpolatedElapsed(now);
    timer.textContent = Domain.formatDuration(elapsed);
    renderer.render({
      progress: latest.durationMs ? elapsed / latest.durationMs : 0,
      mode: latest.mode,
      profile: latest.profile,
      forceOriginal: latest.forceOriginal || elapsed >= latest.durationMs,
    });
  }

  function ensureAnimation() {
    if (animationFrame) return;
    animationFrame = global.requestAnimationFrame(loop);
  }

  function loop(now) {
    animationFrame = 0;
    render(now);
    const connected = Boolean(latest) && now - lastContactAt < 2000;
    if (!connected) {
      connection.textContent = "진행자 화면과 연결이 끊어졌습니다";
      notice.hidden = false;
      notice.textContent = "진행자 화면과 연결이 끊어졌습니다. 창을 닫지 말고 잠시 기다려 주세요.";
    }
    if (latest && latest.running && connected) animationFrame = global.requestAnimationFrame(loop);
  }

  function checkConnection() {
    const now = performance.now();
    const connected = Boolean(latest) && now - lastContactAt < 2000;
    if (!connected) {
      connection.textContent = "진행자 화면과 연결이 끊어졌습니다";
      notice.hidden = false;
      notice.textContent = "진행자 화면과 연결이 끊어졌습니다. 창을 닫지 말고 잠시 기다려 주세요.";
      ensureAnimation();
    }
  }

  fullscreen.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else throw new Error("Fullscreen API unavailable");
    } catch (error) {
      notice.hidden = false;
      notice.textContent = "앱 전체화면을 사용할 수 없습니다. 브라우저의 F11 전체화면을 사용해 주세요.";
    }
  });

  global.addEventListener("beforeunload", () => {
    clearImage();
    if (animationFrame) global.cancelAnimationFrame(animationFrame);
    renderer.destroy();
    channel.close();
    global.clearInterval(connectionTimer);
  });
  connectionTimer = global.setInterval(() => { publish({ type: "participant-ping" }); checkConnection(); }, 1000);
  publish({ type: "participant-ready" });
  publish({ type: "request-snapshot" });
  ensureAnimation();
})(window);
