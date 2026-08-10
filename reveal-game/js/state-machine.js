(function initStateMachine(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const Domain = RevealGame.Domain || (typeof require === "function" ? require("./domain.js") : null);
  const STATES = Object.freeze({
    EMPTY: "EMPTY",
    READY: "READY",
    RUNNING: "RUNNING",
    PAUSED: "PAUSED",
    REVEALED_PENDING_RESULT: "REVEALED_PENDING_RESULT",
    FINALIZED: "FINALIZED",
    SESSION_COMPLETE: "SESSION_COMPLETE",
  });

  const DEFAULT_SETTINGS = Object.freeze({
    mode: "mosaic",
    durationMs: 45000,
    profile: "balanced",
    order: "sequential",
  });

  function normalizeSettings(settings = {}) {
    const durationMs = Math.round(Number(settings.durationMs));
    return {
      mode: settings.mode === "circle" ? "circle" : "mosaic",
      durationMs: Number.isFinite(durationMs) && durationMs >= 5000 && durationMs <= 300000 ? durationMs : 45000,
      profile: ["slow", "balanced", "fast"].includes(settings.profile) ? settings.profile : "balanced",
      order: settings.order === "random" ? "random" : "sequential",
    };
  }

  function freshRound() {
    return {
      accumulatedMs: 0,
      runStartedAt: null,
      finalizedMs: null,
      result: null,
      revealed: false,
    };
  }

  function createEmpty(settings) {
    return {
      status: STATES.EMPTY,
      imageCount: 0,
      currentIndex: 0,
      settings: normalizeSettings(settings),
      round: freshRound(),
      records: [],
    };
  }

  function createSession(imageCount, settings) {
    const count = Math.max(0, Math.floor(Number(imageCount) || 0));
    if (count === 0) return createEmpty(settings);
    return {
      status: STATES.READY,
      imageCount: count,
      currentIndex: 0,
      settings: normalizeSettings(settings),
      round: freshRound(),
      records: [],
    };
  }

  function currentElapsed(model, now) {
    const round = model.round || freshRound();
    let elapsed = Math.max(0, Math.round(round.accumulatedMs || 0));
    if (model.status === STATES.RUNNING && round.runStartedAt !== null) {
      elapsed += Math.max(0, Number(now) - Number(round.runStartedAt));
    }
    return Math.min(model.settings.durationMs, Math.round(elapsed));
  }

  function freezeRound(model, now, exactElapsed) {
    const elapsed = exactElapsed === undefined
      ? currentElapsed(model, now)
      : Domain.clamp(Math.round(exactElapsed), 0, model.settings.durationMs);
    return {
      ...model.round,
      accumulatedMs: elapsed,
      runStartedAt: null,
    };
  }

  function resultCounts(model) {
    return model.records.reduce((counts, record) => {
      if (record.result === "success") counts.success += 1;
      if (record.result === "failure") counts.failure += 1;
      return counts;
    }, { success: 0, failure: 0 });
  }

  function sessionElapsed(model, now) {
    const finalized = model.records.reduce((sum, record) => sum + record.elapsedMs, 0);
    if ([STATES.EMPTY, STATES.FINALIZED, STATES.SESSION_COMPLETE].includes(model.status)) return finalized;
    return finalized + currentElapsed(model, now);
  }

  function accepted(state) {
    return { state, accepted: true, reason: "" };
  }

  function rejected(model, reason) {
    return { state: model, accepted: false, reason };
  }

  function transition(model, event, now = 0) {
    if (!model || !event || !event.type) return rejected(model, "잘못된 상태 전이 요청입니다.");

    switch (event.type) {
      case "LOAD_IMAGES":
      case "NEW_GAME":
        return accepted(createSession(event.imageCount, event.settings || model.settings));

      case "UPDATE_SETTINGS":
        if (![STATES.EMPTY, STATES.READY].includes(model.status)) {
          return rejected(model, "라운드가 진행 중일 때는 설정을 변경할 수 없습니다.");
        }
        return accepted({ ...model, settings: normalizeSettings({ ...model.settings, ...event.settings }) });

      case "START":
        if (![STATES.READY, STATES.PAUSED].includes(model.status)) {
          return rejected(model, "현재 상태에서는 시작하거나 재개할 수 없습니다.");
        }
        return accepted({
          ...model,
          status: STATES.RUNNING,
          round: { ...model.round, runStartedAt: Number(now) },
        });

      case "PAUSE":
        if (model.status !== STATES.RUNNING) return rejected(model, "실행 중일 때만 일시정지할 수 있습니다.");
        return accepted({ ...model, status: STATES.PAUSED, round: freezeRound(model, now) });

      case "REVEAL":
        if (![STATES.READY, STATES.RUNNING, STATES.PAUSED].includes(model.status)) {
          return rejected(model, "현재 상태에서는 정답을 공개할 수 없습니다.");
        }
        return accepted({
          ...model,
          status: STATES.REVEALED_PENDING_RESULT,
          round: { ...freezeRound(model, now), revealed: true },
        });

      case "DURATION_REACHED":
        if (model.status !== STATES.RUNNING) return rejected(model, "실행 중인 라운드가 아닙니다.");
        return accepted({
          ...model,
          status: STATES.REVEALED_PENDING_RESULT,
          round: { ...freezeRound(model, now, model.settings.durationMs), revealed: true },
        });

      case "SUCCESS":
      case "FAILURE": {
        if (![STATES.RUNNING, STATES.PAUSED, STATES.REVEALED_PENDING_RESULT].includes(model.status)) {
          return rejected(model, "현재 라운드는 이미 판정되었거나 판정할 수 없습니다.");
        }
        const round = freezeRound(model, now);
        const result = event.type === "SUCCESS" ? "success" : "failure";
        const record = {
          imageIndex: model.currentIndex,
          result,
          elapsedMs: round.accumulatedMs,
        };
        return accepted({
          ...model,
          status: STATES.FINALIZED,
          round: { ...round, finalizedMs: round.accumulatedMs, result, revealed: true },
          records: [...model.records, record],
        });
      }

      case "RESET": {
        if ([STATES.EMPTY, STATES.SESSION_COMPLETE].includes(model.status)) {
          return rejected(model, "초기화할 현재 라운드가 없습니다.");
        }
        const records = model.status === STATES.FINALIZED
          ? model.records.filter((record) => record.imageIndex !== model.currentIndex)
          : model.records;
        return accepted({ ...model, status: STATES.READY, round: freshRound(), records });
      }

      case "NEXT":
        if (model.status !== STATES.FINALIZED) return rejected(model, "현재 이미지를 먼저 판정해 주세요.");
        if (model.currentIndex + 1 >= model.imageCount) {
          return accepted({ ...model, status: STATES.SESSION_COMPLETE, round: freshRound() });
        }
        return accepted({
          ...model,
          status: STATES.READY,
          currentIndex: model.currentIndex + 1,
          round: freshRound(),
        });

      case "REMOVE_CURRENT": {
        if (model.status !== STATES.READY) return rejected(model, "준비 상태에서만 손상된 이미지를 건너뛸 수 있습니다.");
        const imageCount = Math.max(0, model.imageCount - 1);
        if (imageCount === 0) return accepted(createEmpty(model.settings));
        if (model.currentIndex >= imageCount) {
          return accepted({ ...model, imageCount, status: STATES.SESSION_COMPLETE, round: freshRound() });
        }
        return accepted({ ...model, imageCount, round: freshRound() });
      }

      default:
        return rejected(model, "지원하지 않는 상태 전이입니다.");
    }
  }

  const StateMachine = Object.freeze({
    STATES,
    DEFAULT_SETTINGS,
    normalizeSettings,
    freshRound,
    createEmpty,
    createSession,
    currentElapsed,
    sessionElapsed,
    resultCounts,
    transition,
  });

  RevealGame.StateMachine = StateMachine;
  if (typeof module !== "undefined" && module.exports) module.exports = StateMachine;
})(typeof globalThis !== "undefined" ? globalThis : window);
