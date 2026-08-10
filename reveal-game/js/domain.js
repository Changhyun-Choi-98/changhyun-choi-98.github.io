(function initDomain(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const PROFILE_EXPONENTS = Object.freeze({ slow: 1.65, balanced: 1.20, fast: 0.75 });
  const SUPPORTED_IMAGE_PATTERN = /\.(?:jpe?g|png|webp|bmp)$/i;
  const naturalCollator = new Intl.Collator("ko-KR", {
    numeric: true,
    sensitivity: "base",
  });

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function easedProgress(progress, profile) {
    const exponent = PROFILE_EXPONENTS[profile] || PROFILE_EXPONENTS.balanced;
    return Math.pow(clamp(Number(progress) || 0, 0, 1), exponent);
  }

  function formatDuration(milliseconds) {
    const safeMs = Math.max(0, Math.floor(Number(milliseconds) || 0));
    const totalTenths = Math.floor(safeMs / 100);
    const tenths = totalTenths % 10;
    const totalSeconds = Math.floor(totalTenths / 10);
    const seconds = totalSeconds % 60;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const minutes = totalMinutes % 60;
    const hours = Math.floor(totalMinutes / 60);
    const two = (value) => String(value).padStart(2, "0");

    if (hours > 0) {
      return `${two(hours)}:${two(minutes)}:${two(seconds)}.${tenths}`;
    }
    return `${two(totalMinutes)}:${two(seconds)}.${tenths}`;
  }

  function formatDurationPrecise(milliseconds) {
    const safeMs = Math.max(0, Math.floor(Number(milliseconds) || 0));
    const millisecondsPart = safeMs % 1000;
    const totalSeconds = Math.floor(safeMs / 1000);
    const seconds = totalSeconds % 60;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const minutes = totalMinutes % 60;
    const hours = Math.floor(totalMinutes / 60);
    const two = (value) => String(value).padStart(2, "0");
    const fraction = String(millisecondsPart).padStart(3, "0");
    return hours > 0
      ? `${two(hours)}:${two(minutes)}:${two(seconds)}.${fraction}`
      : `${two(totalMinutes)}:${two(seconds)}.${fraction}`;
  }

  function parseDuration(value) {
    const text = String(value ?? "").trim();
    if (!text || !/^\d+(?::\d+){0,2}(?:\.\d+)?$/.test(text)) {
      return { ok: false, error: "시간 형식이 올바르지 않습니다." };
    }

    const parts = text.split(":");
    const finalPart = parts.pop();
    const seconds = Number(finalPart);
    if (!Number.isFinite(seconds)) {
      return { ok: false, error: "시간을 숫자로 입력해 주세요." };
    }

    let totalSeconds = seconds;
    if (parts.length === 1) {
      if (seconds >= 60) {
        return { ok: false, error: "콜론 뒤 초는 60보다 작아야 합니다." };
      }
      totalSeconds += Number(parts[0]) * 60;
    } else if (parts.length === 2) {
      const hours = Number(parts[0]);
      const minutes = Number(parts[1]);
      if (minutes >= 60 || seconds >= 60) {
        return { ok: false, error: "시:분:초에서 분과 초는 60보다 작아야 합니다." };
      }
      totalSeconds += hours * 3600 + minutes * 60;
    }

    const milliseconds = Math.round(totalSeconds * 1000);
    if (!Number.isSafeInteger(milliseconds) || milliseconds < 0) {
      return { ok: false, error: "지원할 수 있는 시간 범위를 벗어났습니다." };
    }
    return { ok: true, milliseconds };
  }

  function isSupportedImageName(name) {
    return SUPPORTED_IMAGE_PATTERN.test(String(name || ""));
  }

  function naturalCompare(left, right) {
    return naturalCollator.compare(String(left), String(right));
  }

  function fisherYates(items, random = Math.random) {
    const result = Array.from(items);
    for (let index = result.length - 1; index > 0; index -= 1) {
      const target = Math.floor(random() * (index + 1));
      [result[index], result[target]] = [result[target], result[index]];
    }
    return result;
  }

  function buildMosaicLevels(width, height, requestedCount = 96) {
    const safeWidth = Math.floor(Number(width));
    const safeHeight = Math.floor(Number(height));
    if (safeWidth <= 0 || safeHeight <= 0) {
      throw new RangeError("이미지 크기는 0보다 커야 합니다.");
    }

    const maximum = Math.max(safeWidth, safeHeight);
    const count = clamp(Math.floor(requestedCount) || 96, 2, 128);
    const levels = [];
    for (let index = 0; index < count; index += 1) {
      const ratio = index / (count - 1);
      const level = Math.max(1, Math.round(Math.exp(ratio * Math.log(maximum))));
      if (levels[levels.length - 1] !== level) {
        levels.push(level);
      }
    }
    if (levels[0] !== 1) levels.unshift(1);
    if (levels[levels.length - 1] !== maximum) levels.push(maximum);
    return levels;
  }

  function mosaicGrid(width, height, progress, profile = "balanced", levels) {
    const safeWidth = Math.floor(Number(width));
    const safeHeight = Math.floor(Number(height));
    if (safeWidth <= 0 || safeHeight <= 0) {
      throw new RangeError("이미지 크기는 0보다 커야 합니다.");
    }

    const normalized = clamp(Number(progress) || 0, 0, 1);
    if (normalized === 0) {
      return { width: 1, height: 1, longSide: 1, original: false };
    }
    if (normalized === 1) {
      return { width: safeWidth, height: safeHeight, longSide: Math.max(safeWidth, safeHeight), original: true };
    }

    const maximum = Math.max(safeWidth, safeHeight);
    const targetLongSide = Math.max(1, Math.round(Math.exp(easedProgress(normalized, profile) * Math.log(maximum))));
    const availableLevels = levels || buildMosaicLevels(safeWidth, safeHeight);
    let longSide = 1;
    for (const level of availableLevels) {
      if (level > targetLongSide) break;
      longSide = level;
    }

    if (safeWidth >= safeHeight) {
      return {
        width: longSide,
        height: Math.max(1, Math.round(longSide * safeHeight / safeWidth)),
        longSide,
        original: false,
      };
    }
    return {
      width: Math.max(1, Math.round(longSide * safeWidth / safeHeight)),
      height: longSide,
      longSide,
      original: false,
    };
  }

  function containRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const values = [sourceWidth, sourceHeight, targetWidth, targetHeight].map(Number);
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
      return { x: 0, y: 0, width: 0, height: 0 };
    }
    const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      x: (targetWidth - width) / 2,
      y: (targetHeight - height) / 2,
      width,
      height,
    };
  }

  function circleGeometry(rectangle, progress, profile = "balanced", safetyMargin = 2) {
    const rect = rectangle || {};
    const width = Math.max(0, Number(rect.width) || 0);
    const height = Math.max(0, Number(rect.height) || 0);
    const x = Number(rect.x) || 0;
    const y = Number(rect.y) || 0;
    const margin = Math.max(0, Number(safetyMargin) || 0);
    const initialRadius = 0.5 * Math.sqrt(width * width + height * height) + margin;
    const reveal = easedProgress(progress, profile);
    return {
      cx: x + width / 2,
      cy: y + height / 2,
      initialRadius,
      radius: initialRadius * Math.sqrt(Math.max(0, 1 - reveal)),
      clip: { x, y, width, height },
    };
  }

  const Domain = Object.freeze({
    PROFILE_EXPONENTS,
    clamp,
    easedProgress,
    formatDuration,
    formatDurationPrecise,
    parseDuration,
    isSupportedImageName,
    naturalCompare,
    fisherYates,
    buildMosaicLevels,
    mosaicGrid,
    containRect,
    circleGeometry,
  });

  RevealGame.Domain = Domain;
  if (typeof module !== "undefined" && module.exports) module.exports = Domain;
})(typeof globalThis !== "undefined" ? globalThis : window);
