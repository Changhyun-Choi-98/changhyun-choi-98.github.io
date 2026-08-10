(function initRenderer(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const Domain = RevealGame.Domain;
  const CANVAS_PIXEL_BUDGET = 16000000;
  const TEXTURE_PIXEL_BUDGET = 4000000;

  function makeCanvas(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width));
    canvas.height = Math.max(1, Math.round(height));
    return canvas;
  }

  function fitPixelBudget(width, height, budget) {
    const pixels = width * height;
    if (pixels <= budget) return { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) };
    const scale = Math.sqrt(budget / pixels);
    return {
      width: Math.max(1, Math.round(width * scale)),
      height: Math.max(1, Math.round(height * scale)),
    };
  }

  function drawSmoothed(context, source, width, height) {
    context.imageSmoothingEnabled = true;
    if ("imageSmoothingQuality" in context) context.imageSmoothingQuality = "high";
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height);
  }

  function highQualityDownsample(image, sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const target = fitPixelBudget(targetWidth, targetHeight, TEXTURE_PIXEL_BUDGET);
    let currentSource = image;
    let currentWidth = sourceWidth;
    let currentHeight = sourceHeight;

    while (currentWidth > target.width * 2 || currentHeight > target.height * 2) {
      const ratio = Math.min(0.5, Math.max(target.width / currentWidth, target.height / currentHeight, 1 / Math.max(currentWidth, currentHeight) * 2048));
      const nextWidth = Math.max(target.width, Math.round(currentWidth * ratio));
      const nextHeight = Math.max(target.height, Math.round(currentHeight * ratio));
      const next = makeCanvas(nextWidth, nextHeight);
      drawSmoothed(next.getContext("2d"), currentSource, nextWidth, nextHeight);
      currentSource = next;
      currentWidth = nextWidth;
      currentHeight = nextHeight;
    }

    const output = makeCanvas(target.width, target.height);
    drawSmoothed(output.getContext("2d"), currentSource, target.width, target.height);
    return output;
  }

  class CanvasRenderer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d", { alpha: false, desynchronized: true });
      this.pixelBudget = options.pixelBudget || CANVAS_PIXEL_BUDGET;
      this.onResize = options.onResize || (() => {});
      this.decoded = null;
      this.levels = null;
      this.texture = null;
      this.textureKey = "";
      this.cssWidth = 1;
      this.cssHeight = 1;
      this.resizeFrame = 0;
      this.boundWindowResize = () => this.scheduleResize();
      this.resizeObserver = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => this.scheduleResize())
        : null;
      if (this.resizeObserver) this.resizeObserver.observe(canvas);
      global.addEventListener("resize", this.boundWindowResize, { passive: true });
      this.resize();
    }

    scheduleResize() {
      if (this.resizeFrame) return;
      this.resizeFrame = global.requestAnimationFrame(() => {
        this.resizeFrame = 0;
        if (this.resize()) this.onResize();
      });
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      const desiredScale = Math.max(1, global.devicePixelRatio || 1);
      const budgetScale = Math.sqrt(this.pixelBudget / (width * height));
      const scale = Math.max(1, Math.min(desiredScale, budgetScale));
      const backingWidth = Math.max(1, Math.round(width * scale));
      const backingHeight = Math.max(1, Math.round(height * scale));
      const changed = this.canvas.width !== backingWidth || this.canvas.height !== backingHeight;
      this.cssWidth = width;
      this.cssHeight = height;
      if (changed) {
        this.canvas.width = backingWidth;
        this.canvas.height = backingHeight;
      }
      this.context.setTransform(backingWidth / width, 0, 0, backingHeight / height, 0, 0);
      return changed;
    }

    setImage(decoded) {
      this.decoded = decoded || null;
      this.levels = decoded ? Domain.buildMosaicLevels(decoded.width, decoded.height, 96) : null;
      this.texture = null;
      this.textureKey = "";
    }

    imageRect() {
      if (!this.decoded) return { x: 0, y: 0, width: 0, height: 0 };
      return Domain.containRect(this.decoded.width, this.decoded.height, this.cssWidth, this.cssHeight);
    }

    clear() {
      this.context.fillStyle = "#090d12";
      this.context.fillRect(0, 0, this.cssWidth, this.cssHeight);
    }

    drawOriginal(rectangle) {
      const context = this.context;
      context.imageSmoothingEnabled = true;
      if ("imageSmoothingQuality" in context) context.imageSmoothingQuality = "high";
      context.drawImage(this.decoded.image, rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    }

    drawMosaic(rectangle, progress, profile) {
      const grid = Domain.mosaicGrid(this.decoded.width, this.decoded.height, progress, profile, this.levels);
      this.canvas.dataset.grid = grid.original ? "original" : `${grid.width}x${grid.height}`;
      if (grid.original) {
        this.drawOriginal(rectangle);
        return;
      }
      const textureSize = fitPixelBudget(grid.width, grid.height, TEXTURE_PIXEL_BUDGET);
      const key = `${textureSize.width}x${textureSize.height}`;
      if (this.textureKey !== key) {
        this.texture = highQualityDownsample(
          this.decoded.image,
          this.decoded.width,
          this.decoded.height,
          textureSize.width,
          textureSize.height,
        );
        this.textureKey = key;
      }
      this.context.imageSmoothingEnabled = false;
      this.context.drawImage(this.texture, rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    }

    drawCircle(rectangle, progress, profile) {
      this.canvas.dataset.grid = "circle";
      this.drawOriginal(rectangle);
      const circle = Domain.circleGeometry(rectangle, progress, profile, 2);
      this.context.save();
      this.context.beginPath();
      this.context.rect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
      this.context.clip();
      if (circle.radius > 0) {
        this.context.beginPath();
        this.context.arc(circle.cx, circle.cy, circle.radius, 0, Math.PI * 2);
        this.context.fillStyle = "#000000";
        this.context.fill();
      }
      this.context.restore();
    }

    render(options = {}) {
      this.resize();
      this.clear();
      if (!this.decoded) {
        this.canvas.dataset.grid = "empty";
        return;
      }
      const rectangle = this.imageRect();
      const forceOriginal = Boolean(options.forceOriginal);
      const progress = Domain.clamp(Number(options.progress) || 0, 0, 1);
      if (forceOriginal || progress >= 1) {
        this.canvas.dataset.grid = "original";
        this.drawOriginal(rectangle);
      } else if (options.mode === "circle") {
        this.drawCircle(rectangle, progress, options.profile || "balanced");
      } else {
        this.drawMosaic(rectangle, progress, options.profile || "balanced");
      }
    }

    destroy() {
      if (this.resizeObserver) this.resizeObserver.disconnect();
      if (this.resizeFrame) global.cancelAnimationFrame(this.resizeFrame);
      global.removeEventListener("resize", this.boundWindowResize);
      this.decoded = null;
      this.texture = null;
    }
  }

  const Renderer = Object.freeze({
    CANVAS_PIXEL_BUDGET,
    TEXTURE_PIXEL_BUDGET,
    fitPixelBudget,
    highQualityDownsample,
    CanvasRenderer,
  });

  RevealGame.Renderer = Renderer;
})(typeof globalThis !== "undefined" ? globalThis : window);
