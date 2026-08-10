(function initController(global) {
  "use strict";

  const RG = global.RevealGame;
  const { Domain, StateMachine, Ranking, ImageLoader, Renderer, Persistence, Sync } = RG;
  const $ = (id) => document.getElementById(id);
  const elements = {
    networkStatus: $("networkStatus"), offlineStatus: $("offlineStatus"), storageStatus: $("storageStatus"), updateButton: $("updateButton"),
    gameState: $("gameState"), roundLabel: $("roundLabel"), participantStatus: $("participantStatus"), canvas: $("gameCanvas"), emptyStage: $("emptyStage"),
    roundTimer: $("roundTimer"), sessionTimer: $("sessionTimer"), currentMode: $("currentMode"), imageCountLabel: $("imageCountLabel"),
    folderButton: $("folderButton"), folderInput: $("folderInput"), dropZone: $("dropZone"), folderMessage: $("folderMessage"),
    order: $("orderSelect"), mode: $("modeSelect"), duration: $("durationInput"), durationError: $("durationError"), profile: $("profileSelect"),
    start: $("startButton"), success: $("successButton"), failure: $("failureButton"), reveal: $("revealButton"), reset: $("resetButton"), next: $("nextButton"), newGame: $("newGameButton"),
    successCount: $("successCount"), failureCount: $("failureCount"), audienceMode: $("audienceModeButton"), exitAudienceMode: $("exitAudienceModeButton"),
    openParticipant: $("openParticipantButton"), openLeaderboard: $("openLeaderboardButton"),
    modalBackdrop: $("modalBackdrop"), modal: document.querySelector(".modal"), modalTitle: $("modalTitle"), modalMessage: $("modalMessage"), modalDetails: $("modalDetails"), modalActions: $("modalActions"),
    toast: $("toastRegion"), rankingForm: $("rankingForm"), rankingEditId: $("rankingEditId"), teamName: $("teamNameInput"), correctCount: $("correctCountInput"), elapsed: $("elapsedInput"),
    teamNameError: $("teamNameError"), correctCountError: $("correctCountError"), elapsedError: $("elapsedError"), rankingSubmit: $("rankingSubmitButton"), rankingCancel: $("rankingCancelEditButton"),
    rankingBody: $("rankingBody"), rankingEmpty: $("rankingEmpty"), rankingMessage: $("rankingMessage"), clearRankings: $("clearRankingsButton"),
    exportJson: $("exportJsonButton"), importJson: $("importJsonButton"), importJsonInput: $("importJsonInput"), exportCsv: $("exportCsvButton"), importCsv: $("importCsvButton"), importCsvInput: $("importCsvInput"),
    rankingFullscreen: $("rankingFullscreenButton"),
  };

  const stateLabels = {
    EMPTY: "이미지 없음",
    READY: "시작 준비",
    RUNNING: "공개 중",
    PAUSED: "일시정지",
    REVEALED_PENDING_RESULT: "정답 판정 대기",
    FINALIZED: "판정 완료",
    SESSION_COMPLETE: "세션 완료",
  };
  const profileLabels = { slow: "천천히 공개", balanced: "균형", fast: "빠르게 공개" };

  let model = StateMachine.createEmpty();
  let sourceEntries = [];
  let sessionEntries = [];
  let rankings = [];
  let currentDecoded = null;
  let imageLoading = false;
  let currentImageToken = 0;
  let animationFrame = 0;
  let lastHeartbeatAt = -Infinity;
  let toastTimer = 0;
  let modalContext = null;
  let sessionStarted = false;
  let serviceWorkerRegistration = null;
  let reloadForUpdate = false;
  let participantLastSeen = 0;
  let localParticipant = null;
  let sequence = 0;
  let persistenceQueue = Promise.resolve({ ok: true, mode: "memory" });
  let participantStatusTimer = 0;
  const controllerId = global.crypto && typeof global.crypto.randomUUID === "function"
    ? global.crypto.randomUUID()
    : `controller-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const participantToken = (() => {
    const key = "image-reveal-game:participant-token:v1";
    try {
      const saved = global.sessionStorage.getItem(key);
      if (saved) return saved;
      const created = global.crypto && typeof global.crypto.randomUUID === "function"
        ? global.crypto.randomUUID()
        : `participant-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      global.sessionStorage.setItem(key, created);
      return created;
    } catch (error) {
      return `participant-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
  })();

  const renderer = new Renderer.CanvasRenderer(elements.canvas, { onResize: () => renderStage(performance.now()) });
  const imagePool = new ImageLoader.ImagePool((index, error) => {
    if (index === model.currentIndex + 1) showToast(`다음 이미지 준비 중 문제가 발견되었습니다. 이동할 때 자동으로 건너뜁니다. ${error.message}`, true);
  });
  const persistence = new Persistence.Persistence(({ mode, message, warning }) => {
    elements.storageStatus.textContent = mode === "indexeddb" ? "저장소 정상" : mode === "localstorage" ? "대체 저장소 사용" : "메모리에만 저장";
    elements.storageStatus.dataset.mode = mode;
    if (message && (warning || mode !== "indexeddb")) showToast(message, true);
  });
  const sessionChannel = new Sync.CrossWindowChannel(`image-reveal-game-session-v1:${participantToken}`, handleSessionMessage);
  const rankingChannel = new Sync.CrossWindowChannel("image-reveal-game-ranking-v1", handleRankingMessage);

  function showToast(message, isError = false) {
    global.clearTimeout(toastTimer);
    elements.toast.textContent = String(message || "");
    elements.toast.classList.toggle("is-error", isError);
    elements.toast.classList.add("is-visible");
    toastTimer = global.setTimeout(() => elements.toast.classList.remove("is-visible"), 5000);
  }

  function setInlineMessage(element, message, isError = false) {
    element.textContent = message || "";
    element.classList.toggle("is-error", isError);
  }

  function settingsFromInputs() {
    const seconds = Number(elements.duration.value);
    if (!Number.isInteger(seconds) || seconds < 5 || seconds > 300) {
      elements.durationError.textContent = "5~300 사이의 정수(초)를 입력해 주세요.";
      elements.duration.setAttribute("aria-invalid", "true");
      return null;
    }
    elements.durationError.textContent = "";
    elements.duration.removeAttribute("aria-invalid");
    return {
      order: elements.order.value,
      mode: elements.mode.value,
      durationMs: seconds * 1000,
      profile: elements.profile.value,
    };
  }

  function setSettingsInputs(settings) {
    const normalized = StateMachine.normalizeSettings(settings);
    elements.order.value = normalized.order;
    elements.mode.value = normalized.mode;
    elements.duration.value = String(normalized.durationMs / 1000);
    elements.profile.value = normalized.profile;
  }

  function applySettings() {
    const settings = settingsFromInputs();
    if (!settings) return false;
    const result = StateMachine.transition(model, { type: "UPDATE_SETTINGS", settings }, performance.now());
    if (!result.accepted) {
      showToast(result.reason, true);
      setSettingsInputs(model.settings);
      return false;
    }
    model = result.state;
    persistData();
    updateUi(performance.now());
    renderStage(performance.now());
    sendSnapshot(true);
    return true;
  }

  function persistData() {
    const rankingSnapshot = rankings.map((record) => ({ ...record }));
    const settingsSnapshot = { ...model.settings };
    persistenceQueue = persistenceQueue
      .catch(() => ({ ok: false, mode: "memory" }))
      .then(() => persistence.save(rankingSnapshot, settingsSnapshot))
      .then((result) => {
        if (!result.ok) showToast("저장하지 못했습니다. JSON 백업을 내려받아 주세요.", true);
        return result;
      });
    return persistenceQueue;
  }

  function perform(event, now = performance.now(), options = {}) {
    const result = StateMachine.transition(model, event, now);
    if (!result.accepted) {
      if (!options.silent) showToast(result.reason, true);
      return false;
    }
    model = result.state;
    updateUi(now);
    renderStage(now);
    if (options.sync !== false) sendSnapshot(options.includeImage === true);
    if (model.status === StateMachine.STATES.RUNNING) ensureAnimation();
    else stopAnimation();
    return true;
  }

  function updateUi(now = performance.now()) {
    const state = model.status;
    const counts = StateMachine.resultCounts(model);
    const elapsed = StateMachine.currentElapsed(model, now);
    const sessionElapsed = StateMachine.sessionElapsed(model, now);
    elements.gameState.textContent = imageLoading ? "이미지 준비 중" : stateLabels[state];
    elements.roundLabel.textContent = model.imageCount ? `${Math.min(model.currentIndex + 1, model.imageCount)} / ${model.imageCount} 라운드` : "0 / 0 라운드";
    elements.roundTimer.textContent = Domain.formatDuration(elapsed);
    elements.sessionTimer.textContent = Domain.formatDuration(sessionElapsed);
    elements.currentMode.textContent = `${model.settings.mode === "circle" ? "검은색 원" : "모자이크"} · ${profileLabels[model.settings.profile]}`;
    elements.imageCountLabel.textContent = `${model.imageCount}개`;
    elements.successCount.textContent = String(counts.success);
    elements.failureCount.textContent = String(counts.failure);
    elements.emptyStage.hidden = Boolean(currentDecoded);

    const startable = state === StateMachine.STATES.READY || state === StateMachine.STATES.PAUSED;
    elements.start.disabled = imageLoading || !currentDecoded || !(startable || state === StateMachine.STATES.RUNNING);
    elements.start.textContent = state === StateMachine.STATES.RUNNING ? "일시정지" : state === StateMachine.STATES.PAUSED ? "재개" : "시작";
    const resultable = [StateMachine.STATES.RUNNING, StateMachine.STATES.PAUSED, StateMachine.STATES.REVEALED_PENDING_RESULT].includes(state);
    elements.success.disabled = !resultable;
    elements.failure.disabled = !resultable;
    elements.reveal.disabled = imageLoading || !currentDecoded || ![StateMachine.STATES.READY, StateMachine.STATES.RUNNING, StateMachine.STATES.PAUSED].includes(state);
    elements.reset.disabled = [StateMachine.STATES.EMPTY, StateMachine.STATES.SESSION_COMPLETE].includes(state);
    elements.next.disabled = state !== StateMachine.STATES.FINALIZED;
    elements.newGame.disabled = sourceEntries.length === 0;

    const roundSettingsEditable = [StateMachine.STATES.EMPTY, StateMachine.STATES.READY].includes(state);
    elements.mode.disabled = !roundSettingsEditable;
    elements.duration.disabled = !roundSettingsEditable;
    elements.profile.disabled = !roundSettingsEditable;
    elements.order.disabled = sourceEntries.length > 0 && sessionStarted;
  }

  function renderStage(now = performance.now()) {
    if (!currentDecoded) {
      renderer.render();
      syncLocalParticipant(now);
      return;
    }
    const elapsed = StateMachine.currentElapsed(model, now);
    const progress = model.settings.durationMs ? elapsed / model.settings.durationMs : 0;
    const forceOriginal = [StateMachine.STATES.REVEALED_PENDING_RESULT, StateMachine.STATES.FINALIZED, StateMachine.STATES.SESSION_COMPLETE].includes(model.status);
    renderer.render({ progress, mode: model.settings.mode, profile: model.settings.profile, forceOriginal });
    syncLocalParticipant(now);
  }

  function ensureAnimation() {
    if (animationFrame) return;
    animationFrame = global.requestAnimationFrame(tick);
  }

  function stopAnimation() {
    if (animationFrame) global.cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  }

  function tick(now) {
    animationFrame = 0;
    if (model.status !== StateMachine.STATES.RUNNING) return;
    if (StateMachine.currentElapsed(model, now) >= model.settings.durationMs) {
      perform({ type: "DURATION_REACHED" }, now);
      showToast("전체 공개 시간이 끝났습니다. 성공 또는 실패/패스를 선택해 주세요.");
      return;
    }
    renderStage(now);
    updateUi(now);
    if (now - lastHeartbeatAt >= 250) {
      sendSnapshot(false, now);
      lastHeartbeatAt = now;
    }
    animationFrame = global.requestAnimationFrame(tick);
  }

  async function activateCurrentImage() {
    const token = ++currentImageToken;
    if (![StateMachine.STATES.READY, StateMachine.STATES.FINALIZED, StateMachine.STATES.SESSION_COMPLETE].includes(model.status)) return;
    if (model.status === StateMachine.STATES.SESSION_COMPLETE) return;
    imageLoading = true;
    currentDecoded = null;
    renderer.setImage(null);
    updateUi(performance.now());
    renderStage(performance.now());
    sendSnapshot(false);
    try {
      const decoded = await imagePool.activate(model.currentIndex);
      if (token !== currentImageToken) return;
      imageLoading = false;
      currentDecoded = decoded;
      renderer.setImage(decoded);
      updateUi(performance.now());
      renderStage(performance.now());
      sendSnapshot(true);
    } catch (error) {
      if (token !== currentImageToken) return;
      imageLoading = false;
      setInlineMessage(elements.folderMessage, `해석할 수 없는 이미지 1개를 건너뛰었습니다. ${error.message}`, true);
      const failedDescriptor = sessionEntries[model.currentIndex];
      sessionEntries.splice(model.currentIndex, 1);
      sourceEntries = sourceEntries.filter((descriptor) => descriptor !== failedDescriptor);
      imagePool.setEntries(sessionEntries);
      currentDecoded = null;
      renderer.setImage(null);
      perform({ type: "REMOVE_CURRENT" }, performance.now(), { sync: false });
      if (model.status === StateMachine.STATES.READY) await activateCurrentImage();
      else if (model.status === StateMachine.STATES.SESSION_COMPLETE) showSessionSummary();
    }
  }

  function createSessionEntries(settings) {
    const ordered = Array.from(sourceEntries).sort((left, right) => Domain.naturalCompare(left.relativePath, right.relativePath));
    return settings.order === "random" ? Domain.fisherYates(ordered) : ordered;
  }

  async function startNewSession(askConfirmation = true) {
    if (!sourceEntries.length) return;
    if (askConfirmation && model.records.length) {
      openModal({
        title: "새 게임을 시작할까요?",
        message: "현재 세션 결과와 누적 시간이 초기화됩니다. 선택한 이미지 폴더는 유지됩니다.",
        resumePolicy: "cancel",
        actions: [
          { id: "cancel", label: "취소", kind: "secondary" },
          { id: "confirm", label: "새 게임 시작", kind: "primary", callback: () => startNewSession(false) },
        ],
      });
      return;
    }
    const settings = settingsFromInputs() || model.settings;
    sessionEntries = createSessionEntries(settings);
    imagePool.setEntries(sessionEntries);
    currentDecoded = null;
    renderer.setImage(null);
    sessionStarted = false;
    perform({ type: "NEW_GAME", imageCount: sessionEntries.length, settings }, performance.now(), { sync: false });
    await activateCurrentImage();
    setInlineMessage(elements.folderMessage, `이미지 ${sessionEntries.length}개로 새 게임을 준비했습니다.`);
  }

  async function loadDescriptors(descriptors) {
    const prepared = ImageLoader.prepareFileEntries(descriptors);
    if (!prepared.accepted.length) {
      setInlineMessage(elements.folderMessage, prepared.ignoredCount
        ? `지원하는 이미지가 없습니다. 지원하지 않는 파일 ${prepared.ignoredCount}개를 무시했습니다.`
        : "선택한 폴더에 이미지가 없습니다.", true);
      return;
    }
    sourceEntries = prepared.accepted;
    const settings = settingsFromInputs() || model.settings;
    sessionEntries = createSessionEntries(settings);
    imagePool.setEntries(sessionEntries);
    currentDecoded = null;
    renderer.setImage(null);
    sessionStarted = false;
    perform({ type: "LOAD_IMAGES", imageCount: sessionEntries.length, settings }, performance.now(), { sync: false });
    const ignored = prepared.ignoredCount ? ` 지원하지 않는 파일 ${prepared.ignoredCount}개는 무시했습니다.` : "";
    setInlineMessage(elements.folderMessage, `이미지 ${sessionEntries.length}개를 준비했습니다.${ignored}`);
    await activateCurrentImage();
  }

  async function chooseFolderNow() {
    if (typeof global.showDirectoryPicker === "function" && global.location.protocol !== "file:") {
      try {
        const handle = await global.showDirectoryPicker({ mode: "read" });
        const descriptors = await ImageLoader.collectDirectoryHandle(handle);
        await loadDescriptors(descriptors);
      } catch (error) {
        if (error && error.name !== "AbortError") {
          setInlineMessage(elements.folderMessage, "폴더 선택 창을 사용할 수 없어 파일 선택 방식으로 전환합니다.", true);
          elements.folderInput.value = "";
          elements.folderInput.click();
        }
      }
    } else {
      elements.folderInput.value = "";
      elements.folderInput.click();
    }
  }

  function requestFolderSelection() {
    if (model.imageCount && model.status !== StateMachine.STATES.EMPTY) {
      openModal({
        title: "다른 폴더를 선택할까요?",
        message: "현재 게임 세션은 초기화됩니다. 수동 랭킹 데이터는 유지됩니다.",
        resumePolicy: "cancel",
        actions: [
          { id: "cancel", label: "취소", kind: "secondary" },
          { id: "confirm", label: "폴더 선택 계속", kind: "primary", callback: chooseFolderNow },
        ],
      });
    } else chooseFolderNow();
  }

  function startOrPause() {
    if (imageLoading || !currentDecoded) return;
    if (model.status === StateMachine.STATES.RUNNING) perform({ type: "PAUSE" });
    else if ([StateMachine.STATES.READY, StateMachine.STATES.PAUSED].includes(model.status)) {
      sessionStarted = true;
      perform({ type: "START" });
    }
  }

  function confirmReveal() {
    if (elements.reveal.disabled) return;
    openModal({
      title: "정답을 공개할까요?",
      message: "정답을 공개하면 원본 이미지가 표시되고 현재 타이머가 멈춥니다. 계속할까요?",
      resumePolicy: "cancel",
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "confirm", label: "정답 공개", kind: "warning", callback: () => perform({ type: "REVEAL" }) },
      ],
    });
  }

  function confirmReset() {
    if (elements.reset.disabled) return;
    const rollback = model.status === StateMachine.STATES.FINALIZED
      ? "이미 기록한 현재 라운드의 판정과 시간이 함께 취소됩니다."
      : "현재 라운드의 공개 상태와 타이머가 0으로 돌아갑니다.";
    openModal({
      title: "현재 라운드를 초기화할까요?",
      message: `${rollback}\n이전 이미지의 기록은 유지됩니다.`,
      resumePolicy: "cancel",
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "confirm", label: "현재 라운드 초기화", kind: "warning", callback: () => perform({ type: "RESET" }) },
      ],
    });
  }

  function confirmNext() {
    if (elements.next.disabled) return;
    openModal({
      title: model.currentIndex + 1 >= model.imageCount ? "세션 결과를 볼까요?" : "다음 이미지로 이동할까요?",
      message: "현재 원본 화면을 닫고 다음 단계로 이동합니다.",
      resumePolicy: "cancel",
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "confirm", label: "계속", kind: "primary", callback: async () => {
          perform({ type: "NEXT" }, performance.now(), { sync: false });
          if (model.status === StateMachine.STATES.SESSION_COMPLETE) {
            sendSnapshot(false);
            showSessionSummary();
          } else await activateCurrentImage();
        } },
      ],
    });
  }

  function showSessionSummary() {
    const counts = StateMachine.resultCounts(model);
    const totalTime = StateMachine.sessionElapsed(model, performance.now());
    const summaryText = `이미지 공개 퀴즈 결과\n전체 이미지: ${model.imageCount}개\n성공: ${counts.success}개\n실패/패스: ${counts.failure}개\n전체 소요 시간: ${Domain.formatDuration(totalTime)}`;
    openModal({
      title: "세션이 완료되었습니다",
      message: "모든 이미지를 판정했습니다. 이 결과는 수동 랭킹에 자동 등록되지 않습니다.",
      details: [
        ["전체 이미지", `${model.imageCount}개`], ["성공", `${counts.success}개`], ["실패/패스", `${counts.failure}개`], ["전체 소요 시간", Domain.formatDuration(totalTime)],
      ],
      actions: [
        { id: "return", label: "원본 화면으로 돌아가기", kind: "secondary" },
        { id: "copy", label: "결과를 클립보드에 복사", kind: "secondary", close: false, callback: () => copyText(summaryText) },
        { id: "same", label: "동일 이미지로 새 게임", kind: "primary", callback: () => startNewSession(false) },
        { id: "other", label: "다른 폴더 선택", kind: "primary", callback: chooseFolderNow },
      ],
    });
  }

  async function copyText(text) {
    try {
      if (!navigator.clipboard || !global.isSecureContext) throw new Error("클립보드 기능을 사용할 수 없습니다.");
      await navigator.clipboard.writeText(text);
      showToast("결과를 클립보드에 복사했습니다.");
    } catch (error) {
      try {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.append(area);
        area.select();
        if (!document.execCommand("copy")) throw new Error("copy failed");
        area.remove();
        showToast("결과를 클립보드에 복사했습니다.");
      } catch (fallbackError) {
        showToast("클립보드 복사에 실패했습니다. 결과를 화면에서 직접 기록해 주세요.", true);
      }
    }
  }

  function openModal(options) {
    if (modalContext) closeModal("replace", false);
    const previouslyFocused = document.activeElement;
    const wasRunning = model.status === StateMachine.STATES.RUNNING;
    if (wasRunning) perform({ type: "PAUSE" }, performance.now(), { silent: true });
    modalContext = { options, previouslyFocused, wasRunning };
    elements.modalTitle.textContent = options.title;
    elements.modalMessage.textContent = options.message || "";
    elements.modalDetails.replaceChildren();
    if (options.details && options.details.length) {
      const list = document.createElement("dl");
      for (const [term, description] of options.details) {
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = term;
        dd.textContent = description;
        list.append(dt, dd);
      }
      elements.modalDetails.append(list);
      elements.modalDetails.hidden = false;
    } else elements.modalDetails.hidden = true;
    elements.modalActions.replaceChildren();
    for (const action of options.actions || [{ id: "close", label: "닫기", kind: "secondary" }]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.dataset.modalAction = action.id;
      button.className = action.kind === "primary" ? "primary-button" : action.kind === "warning" ? "warning-button" : action.kind === "danger" ? "danger-outline-button" : "secondary-button";
      button.addEventListener("click", () => {
        const close = action.close !== false;
        if (close) closeModal(action.id, true);
        if (action.callback) action.callback();
      });
      elements.modalActions.append(button);
    }
    elements.modalBackdrop.hidden = false;
    const firstAction = elements.modalActions.querySelector("button");
    (firstAction || elements.modal).focus();
  }

  function closeModal(result = "cancel", restoreFocus = true) {
    if (!modalContext) return;
    const context = modalContext;
    modalContext = null;
    elements.modalBackdrop.hidden = true;
    elements.modalActions.replaceChildren();
    const shouldResume = context.wasRunning && (context.options.resumePolicy === "always" || (context.options.resumePolicy === "cancel" && ["cancel", "escape"].includes(result)));
    if (shouldResume && model.status === StateMachine.STATES.PAUSED) perform({ type: "START" }, performance.now());
    if (restoreFocus && context.previouslyFocused && typeof context.previouslyFocused.focus === "function") context.previouslyFocused.focus();
    if (context.options.onClose) context.options.onClose(result);
  }

  function handleModalKeys(event) {
    if (!modalContext) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeModal("escape");
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = Array.from(elements.modal.querySelectorAll("button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),[tabindex]:not([tabindex='-1'])"));
    if (!focusable.length) {
      event.preventDefault();
      elements.modal.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (document.activeElement === elements.modal || !elements.modal.contains(document.activeElement)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    } else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }

  function snapshot(now = performance.now()) {
    return {
      type: "snapshot",
      controllerId,
      sequence: ++sequence,
      status: model.status,
      round: model.imageCount ? Math.min(model.currentIndex + 1, model.imageCount) : 0,
      total: model.imageCount,
      elapsedMs: StateMachine.currentElapsed(model, now),
      durationMs: model.settings.durationMs,
      mode: model.settings.mode,
      profile: model.settings.profile,
      running: model.status === StateMachine.STATES.RUNNING,
      forceOriginal: [StateMachine.STATES.REVEALED_PENDING_RESULT, StateMachine.STATES.FINALIZED, StateMachine.STATES.SESSION_COMPLETE].includes(model.status),
      imageId: currentDecoded ? `${model.currentIndex}-${currentDecoded.file.size}-${currentDecoded.file.lastModified}` : null,
    };
  }

  function sendSnapshot(includeImage = false, now = performance.now(), target) {
    const state = snapshot(now);
    sessionChannel.publish(state, { target, scalar: !target });
    if (includeImage && currentDecoded) {
      const blob = currentDecoded.file.slice(0, currentDecoded.file.size, currentDecoded.file.type || "application/octet-stream");
      sessionChannel.publish({ type: "image", imageId: state.imageId, blob }, { target, broadcast: false, scalar: false });
    }
    syncLocalParticipant(now);
  }

  function handleSessionMessage(payload, meta) {
    if (!payload || !meta.sourceWindow) return;
    let openedByController = false;
    try { openedByController = meta.sourceWindow.opener === global; } catch (error) { openedByController = false; }
    if (!sessionChannel.hasWindow(meta.sourceWindow) && !openedByController) return;
    if (payload.type === "request-snapshot" || payload.type === "participant-ready") {
      participantLastSeen = Date.now();
      if (meta.sourceWindow) sessionChannel.addWindow(meta.sourceWindow);
      sendSnapshot(true, performance.now(), meta.sourceWindow || undefined);
      updateParticipantStatus();
    } else if (payload.type === "participant-ping") {
      participantLastSeen = Date.now();
      if (meta.sourceWindow) sessionChannel.addWindow(meta.sourceWindow);
      sendSnapshot(false, performance.now(), meta.sourceWindow || undefined);
      updateParticipantStatus();
    }
  }

  function publishRankings(target) {
    rankingChannel.publish({ type: "rankings-changed", rankings: rankings.map((record) => ({ ...record })) }, { target, scalar: !target });
  }

  function rankingSaveMessage(savedMessage, result) {
    return result && result.mode === "memory"
      ? `${savedMessage} 현재 창에만 반영되었습니다. JSON 백업을 내려받아 주세요.`
      : savedMessage;
  }

  function handleRankingMessage(payload, meta) {
    if (!payload || payload.type !== "request-rankings") return;
    if (meta.sourceWindow) {
      let openedByController = false;
      try { openedByController = meta.sourceWindow.opener === global; } catch (error) { openedByController = false; }
      if (!rankingChannel.hasWindow(meta.sourceWindow) && !openedByController) return;
      rankingChannel.addWindow(meta.sourceWindow);
      publishRankings(meta.sourceWindow);
    } else publishRankings();
  }

  function updateParticipantStatus() {
    sessionChannel.removeClosedWindows();
    const localOpen = localParticipant && !localParticipant.window.closed;
    const connected = localOpen || Date.now() - participantLastSeen < 2500;
    elements.participantStatus.textContent = connected ? "참가자 창 연결됨" : "참가자 창 미연결";
  }

  function openParticipantWindow() {
    let child;
    if (global.location.protocol === "file:") child = global.open("about:blank", "reveal-game-participant", "popup,width=1200,height=800");
    else child = global.open(`display.html#participant=${encodeURIComponent(participantToken)}`, "reveal-game-participant", "popup,width=1200,height=800");
    if (!child) {
      showToast("참가자 창이 차단되었습니다. 주소창 옆 팝업 차단 아이콘에서 이 사이트의 팝업을 허용해 주세요.", true);
      return;
    }
    if (global.location.protocol === "file:") buildLocalParticipant(child);
    else {
      sessionChannel.addWindow(child);
      global.setTimeout(() => sendSnapshot(true, performance.now(), child), 300);
    }
    showToast("새 창을 두 번째 모니터로 옮긴 뒤 전체화면 버튼 또는 F11을 사용하세요.");
  }

  function buildLocalParticipant(child) {
    const doc = child.document;
    doc.replaceChildren();
    const html = doc.createElement("html");
    html.lang = "ko";
    const head = doc.createElement("head");
    const meta = doc.createElement("meta");
    meta.name = "viewport";
    meta.content = "width=device-width,initial-scale=1";
    const title = doc.createElement("title");
    title.textContent = "참가자 화면 · 이미지 공개 퀴즈";
    const style = doc.createElement("style");
    style.textContent = "html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#05080d;color:#fff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}main{height:100%;display:grid;grid-template-rows:auto 1fr auto}.top{display:flex;gap:16px;align-items:center;padding:12px 18px;background:#080d15}.top span:nth-child(2){margin-left:auto}button{min-height:42px;padding:8px 12px;border:1px solid #65758c;border-radius:8px;background:#17243a;color:#fff}canvas{width:100%;height:100%;display:block}.timer{text-align:center;padding:12px;background:#080d15;font-size:clamp(30px,5vw,72px);font-variant-numeric:tabular-nums}";
    head.append(meta, title, style);
    const body = doc.createElement("body");
    const main = doc.createElement("main");
    const top = doc.createElement("div"); top.className = "top";
    const status = doc.createElement("span"); status.textContent = "진행자와 직접 연결됨";
    const round = doc.createElement("span"); round.textContent = "0 / 0 라운드";
    const fullscreen = doc.createElement("button"); fullscreen.type = "button"; fullscreen.textContent = "전체화면";
    const canvas = doc.createElement("canvas"); canvas.setAttribute("aria-label", "참가자용 공개 이미지");
    const timer = doc.createElement("div"); timer.className = "timer"; timer.textContent = "00:00.0";
    fullscreen.addEventListener("click", () => doc.documentElement.requestFullscreen && doc.documentElement.requestFullscreen().catch(() => { status.textContent = "F11로 전체화면을 사용해 주세요"; }));
    top.append(status, round, fullscreen); main.append(top, canvas, timer); body.append(main); html.append(head, body); doc.append(html);
    const participantRenderer = new Renderer.CanvasRenderer(canvas, { onResize: () => syncLocalParticipant(performance.now()) });
    localParticipant = { window: child, renderer: participantRenderer, round, timer, image: null };
    child.addEventListener("beforeunload", () => { participantRenderer.destroy(); localParticipant = null; });
    updateParticipantStatus();
  }

  function syncLocalParticipant(now) {
    if (!localParticipant) return;
    if (localParticipant.window.closed) {
      localParticipant.renderer.destroy();
      localParticipant = null;
      updateParticipantStatus();
      return;
    }
    if (localParticipant.image !== currentDecoded) {
      localParticipant.image = currentDecoded;
      localParticipant.renderer.setImage(currentDecoded);
    }
    const elapsed = StateMachine.currentElapsed(model, now);
    localParticipant.timer.textContent = Domain.formatDuration(elapsed);
    localParticipant.round.textContent = model.imageCount ? `${Math.min(model.currentIndex + 1, model.imageCount)} / ${model.imageCount} 라운드` : "0 / 0 라운드";
    localParticipant.renderer.render({
      progress: model.settings.durationMs ? elapsed / model.settings.durationMs : 0,
      mode: model.settings.mode,
      profile: model.settings.profile,
      forceOriginal: [StateMachine.STATES.REVEALED_PENDING_RESULT, StateMachine.STATES.FINALIZED, StateMachine.STATES.SESSION_COMPLETE].includes(model.status),
    });
  }

  async function enterAudienceMode() {
    document.body.classList.add("audience-mode");
    elements.exitAudienceMode.hidden = false;
    try {
      if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else showToast("이 브라우저에서는 앱 전체화면을 지원하지 않아 화면 채우기 모드로 표시합니다.");
    } catch (error) {
      showToast("전체화면 권한이 거부되어 화면 채우기 모드로 표시합니다. 브라우저 F11도 사용할 수 있습니다.", true);
    }
    renderer.scheduleResize();
  }

  async function exitAudienceMode() {
    document.body.classList.remove("audience-mode");
    elements.exitAudienceMode.hidden = true;
    if (document.fullscreenElement && document.exitFullscreen) {
      try { await document.exitFullscreen(); } catch (error) { /* CSS fallback is already restored. */ }
    }
    renderer.scheduleResize();
  }

  async function toggleFullscreen(target = document.documentElement) {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (target.requestFullscreen) await target.requestFullscreen();
      else throw new Error("Fullscreen API unavailable");
    } catch (error) {
      showToast("앱 전체화면을 시작할 수 없습니다. 브라우저의 F11 전체화면을 사용해 주세요.", true);
    }
  }

  function clearRankingErrors() {
    for (const [input, error] of [[elements.teamName, elements.teamNameError], [elements.correctCount, elements.correctCountError], [elements.elapsed, elements.elapsedError]]) {
      input.removeAttribute("aria-invalid");
      error.textContent = "";
    }
  }

  function showRankingErrors(errors) {
    clearRankingErrors();
    const pairs = {
      teamName: [elements.teamName, elements.teamNameError],
      correctCount: [elements.correctCount, elements.correctCountError],
      elapsed: [elements.elapsed, elements.elapsedError],
    };
    let firstInvalid = null;
    for (const [key, message] of Object.entries(errors)) {
      const pair = pairs[key];
      if (!pair) continue;
      pair[0].setAttribute("aria-invalid", "true");
      pair[1].textContent = message;
      if (!firstInvalid) firstInvalid = pair[0];
    }
    if (firstInvalid) firstInvalid.focus();
  }

  function resetRankingForm() {
    elements.rankingForm.reset();
    elements.rankingEditId.value = "";
    elements.rankingSubmit.textContent = "팀 추가";
    elements.rankingCancel.hidden = true;
    clearRankingErrors();
  }

  function setRankingFormBusy(busy) {
    for (const input of [elements.teamName, elements.correctCount, elements.elapsed, elements.rankingSubmit, elements.rankingCancel]) input.disabled = busy;
  }

  function renderRankings() {
    elements.rankingBody.replaceChildren();
    const sorted = Ranking.sortAndRank(rankings);
    elements.rankingEmpty.hidden = sorted.length > 0;
    for (const record of sorted) {
      const row = document.createElement("tr");
      row.dataset.rank = String(record.rank);
      const rankCell = document.createElement("td"); rankCell.textContent = `${record.rank}위`;
      const teamCell = document.createElement("td"); teamCell.textContent = record.teamName;
      const scoreCell = document.createElement("td"); scoreCell.textContent = `${record.correctCount}개`;
      const timeCell = document.createElement("td"); timeCell.textContent = Domain.formatDuration(record.elapsedMs);
      const actionCell = document.createElement("td");
      const actions = document.createElement("div"); actions.className = "row-actions";
      const edit = document.createElement("button"); edit.type = "button"; edit.className = "secondary-button"; edit.textContent = "수정";
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "danger-outline-button"; remove.textContent = "삭제";
      edit.addEventListener("click", () => editRanking(record.id));
      remove.addEventListener("click", () => confirmDeleteRanking(record.id));
      actions.append(edit, remove); actionCell.append(actions); row.append(rankCell, teamCell, scoreCell, timeCell, actionCell); elements.rankingBody.append(row);
    }
  }

  function editRanking(id) {
    const record = rankings.find((item) => item.id === id);
    if (!record) return;
    elements.rankingEditId.value = id;
    elements.teamName.value = record.teamName;
    elements.correctCount.value = String(record.correctCount);
    elements.elapsed.value = Domain.formatDurationPrecise(record.elapsedMs);
    elements.rankingSubmit.textContent = "수정 저장";
    elements.rankingCancel.hidden = false;
    clearRankingErrors();
    elements.teamName.focus();
  }

  async function submitRanking(event) {
    event.preventDefault();
    const editingId = elements.rankingEditId.value || null;
    const validation = Ranking.validateRankingInput({ teamName: elements.teamName.value, correctCount: elements.correctCount.value, elapsed: elements.elapsed.value }, rankings, editingId);
    if (!validation.ok) {
      showRankingErrors(validation.errors);
      return;
    }
    const now = new Date().toISOString();
    if (editingId) {
      rankings = rankings.map((record) => record.id === editingId ? { ...record, ...validation.value, updatedAt: now } : record);
      setInlineMessage(elements.rankingMessage, "팀 정보를 수정했습니다.");
    } else {
      const created = Ranking.createRanking(validation.value, rankings, now);
      if (!created.ok) { showRankingErrors(created.errors); return; }
      rankings.push(created.value);
      setInlineMessage(elements.rankingMessage, "팀을 추가하고 순위를 다시 계산했습니다.");
    }
    setRankingFormBusy(true);
    setInlineMessage(elements.rankingMessage, "브라우저 저장소에 저장 중입니다.");
    const saveResult = await persistData();
    resetRankingForm();
    setRankingFormBusy(false);
    renderRankings();
    publishRankings();
    setInlineMessage(elements.rankingMessage, rankingSaveMessage(editingId ? "팀 정보를 수정하고 저장했습니다." : "팀을 추가하고 순위를 저장했습니다.", saveResult));
  }

  function confirmDeleteRanking(id) {
    const record = rankings.find((item) => item.id === id);
    if (!record) return;
    openModal({
      title: "팀을 삭제할까요?", message: `“${record.teamName}” 팀의 랭킹을 삭제합니다.`, resumePolicy: "always",
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "delete", label: "삭제", kind: "danger", callback: async () => {
          rankings = rankings.filter((item) => item.id !== id);
          renderRankings();
          const saveResult = await persistData();
          publishRankings();
          setInlineMessage(elements.rankingMessage, rankingSaveMessage("팀을 삭제했습니다.", saveResult));
        } },
      ],
    });
  }

  function confirmClearRankings() {
    if (!rankings.length) { showToast("삭제할 랭킹이 없습니다."); return; }
    openModal({
      title: "랭킹을 모두 삭제할까요?", message: "이 작업은 되돌릴 수 없습니다. 필요하면 먼저 JSON 백업을 내보내세요.", resumePolicy: "always",
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "delete", label: "전체 삭제", kind: "danger", callback: async () => {
          rankings = [];
          resetRankingForm();
          renderRankings();
          const saveResult = await persistData();
          publishRankings();
          setInlineMessage(elements.rankingMessage, rankingSaveMessage("랭킹을 모두 삭제했습니다.", saveResult));
        } },
      ],
    });
  }

  function downloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    global.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function exportJson() {
    const backup = Ranking.makeBackup(rankings, model.settings);
    downloadText("이미지-공개-퀴즈-랭킹.json", `${JSON.stringify(backup, null, 2)}\n`, "application/json;charset=utf-8");
    setInlineMessage(elements.rankingMessage, "JSON 백업을 내보냈습니다.");
  }

  function exportCsv() {
    downloadText("이미지-공개-퀴즈-랭킹.csv", Ranking.exportCsv(rankings), "text/csv;charset=utf-8");
    setInlineMessage(elements.rankingMessage, "Excel용 UTF-8 CSV를 내보냈습니다.");
  }

  function previewImport(imported, importedSettings, report) {
    openModal({
      title: "가져오기 미리보기",
      message: `유효한 팀 ${imported.length}개를 찾았습니다.${report ? `\n${report}` : ""}`,
      resumePolicy: "always",
      details: [["현재 데이터", `${rankings.length}팀`], ["가져올 데이터", `${imported.length}팀`]],
      actions: [
        { id: "cancel", label: "취소", kind: "secondary" },
        { id: "merge", label: "기존 데이터와 병합", kind: "primary", callback: () => applyImport(imported, importedSettings, false) },
        { id: "replace", label: "기존 데이터 교체", kind: "warning", callback: () => applyImport(imported, importedSettings, true) },
      ],
    });
  }

  async function applyImport(imported, importedSettings, replace) {
    if (replace) rankings = Array.from(imported);
    else {
      const merged = Ranking.mergeRankings(rankings, imported);
      rankings = merged.rankings;
      if (merged.skipped.length) showToast(`중복 팀 ${merged.skipped.length}개는 병합에서 건너뛰었습니다.`);
    }
    if (importedSettings && [StateMachine.STATES.EMPTY, StateMachine.STATES.READY].includes(model.status)) {
      const settings = StateMachine.normalizeSettings(importedSettings);
      setSettingsInputs(settings);
      perform({ type: "UPDATE_SETTINGS", settings }, performance.now(), { sync: false });
    }
    resetRankingForm();
    renderRankings();
    const saveResult = await persistData();
    publishRankings();
    setInlineMessage(elements.rankingMessage, rankingSaveMessage(replace ? "가져온 데이터로 교체했습니다." : "가져온 데이터를 병합했습니다.", saveResult));
  }

  async function handleJsonImport(file) {
    try {
      const candidate = JSON.parse(await file.text());
      const validated = Ranking.validateBackup(candidate);
      if (!validated.ok) throw new Error(validated.errors.join("\n"));
      previewImport(validated.value.rankings, validated.value.settings, "JSON 스키마 검증을 통과했습니다.");
    } catch (error) {
      setInlineMessage(elements.rankingMessage, `JSON을 가져오지 못했습니다. ${error.message}`, true);
    }
  }

  async function handleCsvImport(file) {
    try {
      const parsed = Ranking.parseCsv(await file.text());
      if (!parsed.rankings.length) throw new Error(parsed.errors.join("\n") || "유효한 데이터 행이 없습니다.");
      const report = parsed.errors.length ? `잘못된 행 ${parsed.errors.length}개는 건너뜁니다. ${parsed.errors.slice(0, 3).join(" ")}` : "모든 데이터 행이 유효합니다.";
      previewImport(parsed.rankings, null, report);
    } catch (error) {
      setInlineMessage(elements.rankingMessage, `CSV를 가져오지 못했습니다. ${error.message}`, true);
    }
  }

  function selectTab(name) {
    for (const button of document.querySelectorAll("[data-tab]")) {
      const selected = button.dataset.tab === name;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    $("gamePanel").hidden = name !== "game";
    $("rankingPanel").hidden = name !== "ranking";
    $("helpPanel").hidden = name !== "help";
    if (name === "game") renderer.scheduleResize();
  }

  function isShortcutExcludedTarget(target) {
    return Boolean(target) && ((typeof target.matches === "function" && target.matches("input,textarea,select,button,a[href],[role='button']")) || target.isContentEditable);
  }

  function isTextEntryTarget(target) {
    return Boolean(target) && ((typeof target.matches === "function" && target.matches("input,textarea")) || target.isContentEditable);
  }

  function handleShortcuts(event) {
    if (event.key === "Backspace" && !isTextEntryTarget(event.target)) event.preventDefault();
    if (event.isComposing || event.repeat || event.ctrlKey || event.metaKey || event.altKey || isShortcutExcludedTarget(event.target) || modalContext) return;
    const key = event.key;
    if (key === " ") { event.preventDefault(); startOrPause(); }
    else if (key === "Enter") { if (!elements.success.disabled) elements.success.click(); }
    else if (key.toLowerCase() === "p") { if (!elements.failure.disabled) elements.failure.click(); }
    else if (key.toLowerCase() === "r") { if (!elements.reveal.disabled) confirmReveal(); }
    else if (key.toLowerCase() === "n") { if (!elements.next.disabled) confirmNext(); }
    else if (key === "Backspace") { if (!elements.reset.disabled) confirmReset(); }
    else if (key.toLowerCase() === "l" && !event.ctrlKey && !event.metaKey) selectTab("ranking");
    else if (key === "?" || (key === "/" && event.shiftKey)) selectTab("help");
    else if (key.toLowerCase() === "f" && !event.ctrlKey && !event.metaKey) toggleFullscreen();
    else if (key === "Escape" && document.body.classList.contains("audience-mode")) exitAudienceMode();
  }

  function updateNetworkStatus() {
    elements.networkStatus.textContent = navigator.onLine ? "온라인" : "오프라인";
  }

  async function registerServiceWorker() {
    if (global.location.protocol === "file:") {
      elements.offlineStatus.textContent = "비상용 단일 파일 모드";
      return;
    }
    if (!("serviceWorker" in navigator)) {
      elements.offlineStatus.textContent = "오프라인 기능 미지원";
      showToast("이 브라우저에서는 서비스 워커를 지원하지 않습니다. 비상용 단일 HTML을 사용해 주세요.", true);
      return;
    }
    try {
      serviceWorkerRegistration = await navigator.serviceWorker.register("./sw.js", { scope: "./", updateViaCache: "none" });
      const showUpdate = () => { elements.offlineStatus.textContent = "업데이트 가능"; elements.updateButton.hidden = false; };
      if (serviceWorkerRegistration.waiting) showUpdate();
      serviceWorkerRegistration.addEventListener("updatefound", () => {
        const worker = serviceWorkerRegistration.installing;
        if (!worker) return;
        worker.addEventListener("statechange", () => {
          if (worker.state === "installed" && navigator.serviceWorker.controller) showUpdate();
        });
      });
      navigator.serviceWorker.addEventListener("message", (event) => {
        if (!event.data) return;
        if (event.data.type === "CACHE_READY") elements.offlineStatus.textContent = "오프라인 사용 준비 완료";
        if (event.data.type === "CACHE_ERROR") { elements.offlineStatus.textContent = "오프라인 준비 실패"; showToast("오프라인 캐시 준비에 실패했습니다. 온라인에서 새로고침하거나 단일 HTML을 사용해 주세요.", true); }
      });
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (reloadForUpdate) global.location.reload();
        else if (navigator.serviceWorker.controller) navigator.serviceWorker.controller.postMessage({ type: "CHECK_CACHE" });
      });
      const active = serviceWorkerRegistration.active || navigator.serviceWorker.controller;
      if (active) active.postMessage({ type: "CHECK_CACHE" });
      serviceWorkerRegistration.update().catch(() => {});
    } catch (error) {
      elements.offlineStatus.textContent = "오프라인 준비 실패";
      showToast("서비스 워커 등록에 실패했습니다. 비상용 단일 HTML을 준비해 주세요.", true);
    }
  }

  function bindEvents() {
    const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
    for (const button of tabButtons) {
      button.addEventListener("click", () => selectTab(button.dataset.tab));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const current = tabButtons.indexOf(button);
        const target = event.key === "Home" ? 0 : event.key === "End" ? tabButtons.length - 1
          : (current + (event.key === "ArrowRight" ? 1 : -1) + tabButtons.length) % tabButtons.length;
        tabButtons[target].focus();
        tabButtons[target].click();
      });
    }
    elements.folderButton.addEventListener("click", requestFolderSelection);
    elements.dropZone.addEventListener("click", requestFolderSelection);
    elements.dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); requestFolderSelection(); } });
    elements.folderInput.addEventListener("change", () => { if (elements.folderInput.files.length) loadDescriptors(elements.folderInput.files); });
    for (const name of ["dragenter", "dragover"]) elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.add("is-dragging"); });
    for (const name of ["dragleave", "drop"]) elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.remove("is-dragging"); });
    elements.dropZone.addEventListener("drop", async (event) => {
      try {
        const descriptors = await ImageLoader.collectDataTransfer(event.dataTransfer);
        if (model.imageCount) {
          openModal({ title: "놓은 폴더로 교체할까요?", message: "현재 게임 세션은 초기화됩니다. 수동 랭킹은 유지됩니다.", resumePolicy: "cancel", actions: [
            { id: "cancel", label: "취소", kind: "secondary" }, { id: "confirm", label: "교체", kind: "primary", callback: () => loadDescriptors(descriptors) },
          ] });
        } else await loadDescriptors(descriptors);
      } catch (error) { setInlineMessage(elements.folderMessage, "끌어다 놓은 폴더를 읽지 못했습니다. 폴더 선택 버튼을 사용해 주세요.", true); }
    });
    for (const input of [elements.mode, elements.duration, elements.profile]) input.addEventListener("change", applySettings);
    elements.order.addEventListener("change", async () => {
      if (!applySettings() || !sourceEntries.length || sessionStarted) return;
      sessionEntries = createSessionEntries(model.settings); imagePool.setEntries(sessionEntries); currentDecoded = null; renderer.setImage(null);
      perform({ type: "NEW_GAME", imageCount: sessionEntries.length, settings: model.settings }, performance.now(), { sync: false }); await activateCurrentImage();
    });
    elements.start.addEventListener("click", startOrPause);
    elements.success.addEventListener("click", () => perform({ type: "SUCCESS" }));
    elements.failure.addEventListener("click", () => perform({ type: "FAILURE" }));
    elements.reveal.addEventListener("click", confirmReveal);
    elements.reset.addEventListener("click", confirmReset);
    elements.next.addEventListener("click", confirmNext);
    elements.newGame.addEventListener("click", () => startNewSession(true));
    elements.audienceMode.addEventListener("click", enterAudienceMode);
    elements.exitAudienceMode.addEventListener("click", exitAudienceMode);
    elements.openParticipant.addEventListener("click", openParticipantWindow);
    elements.openLeaderboard.addEventListener("click", () => {
      if (global.location.protocol === "file:") {
        selectTab("ranking");
        showToast("단일 파일 모드에서는 이 창의 랭킹 탭을 사용합니다.");
        return;
      }
      const child = global.open("leaderboard.html", "reveal-game-leaderboard", "popup,width=1200,height=850");
      if (!child) showToast("랭킹 창이 차단되었습니다. 이 사이트의 팝업을 허용해 주세요.", true);
      else rankingChannel.addWindow(child);
    });
    elements.rankingFullscreen.addEventListener("click", () => toggleFullscreen($("rankingPanel")));
    for (const link of document.querySelectorAll("[data-offline-help-link]")) link.addEventListener("click", (event) => { event.preventDefault(); selectTab("help"); });
    elements.rankingForm.addEventListener("submit", submitRanking);
    elements.rankingCancel.addEventListener("click", resetRankingForm);
    elements.clearRankings.addEventListener("click", confirmClearRankings);
    elements.exportJson.addEventListener("click", exportJson);
    elements.exportCsv.addEventListener("click", exportCsv);
    elements.importJson.addEventListener("click", () => { elements.importJsonInput.value = ""; elements.importJsonInput.click(); });
    elements.importCsv.addEventListener("click", () => { elements.importCsvInput.value = ""; elements.importCsvInput.click(); });
    elements.importJsonInput.addEventListener("change", () => { if (elements.importJsonInput.files[0]) handleJsonImport(elements.importJsonInput.files[0]); });
    elements.importCsvInput.addEventListener("change", () => { if (elements.importCsvInput.files[0]) handleCsvImport(elements.importCsvInput.files[0]); });
    elements.updateButton.addEventListener("click", () => {
      if (!serviceWorkerRegistration || !serviceWorkerRegistration.waiting) return;
      reloadForUpdate = true;
      serviceWorkerRegistration.waiting.postMessage({ type: "SKIP_WAITING" });
    });
    elements.modalBackdrop.addEventListener("mousedown", (event) => { if (event.target === elements.modalBackdrop) closeModal("cancel"); });
    document.addEventListener("keydown", handleModalKeys, true);
    document.addEventListener("keydown", handleShortcuts);
    global.addEventListener("online", () => { elements.networkStatus.textContent = "온라인"; });
    global.addEventListener("offline", () => { elements.networkStatus.textContent = "오프라인"; });
    document.addEventListener("fullscreenchange", () => {
      if (!document.fullscreenElement && document.body.classList.contains("audience-mode")) exitAudienceMode();
    });
    global.addEventListener("beforeunload", () => { global.clearInterval(participantStatusTimer); imagePool.dispose(); renderer.destroy(); persistence.close(); sessionChannel.close(); rankingChannel.close(); });
    participantStatusTimer = global.setInterval(updateParticipantStatus, 1000);
  }

  async function initialize() {
    bindEvents();
    updateNetworkStatus();
    const stored = await persistence.init();
    rankings = Array.isArray(stored.snapshot.rankings) ? stored.snapshot.rankings : [];
    const settings = StateMachine.normalizeSettings(stored.snapshot.settings || StateMachine.DEFAULT_SETTINGS);
    model = StateMachine.createEmpty(settings);
    setSettingsInputs(settings);
    renderRankings();
    updateUi(performance.now());
    renderStage(performance.now());
    registerServiceWorker();
  }

  global.addEventListener("error", (event) => {
    const message = event.error && event.error.message ? event.error.message : event.message;
    showToast(`예기치 않은 오류가 발생했습니다. 현재 데이터를 JSON으로 백업하고 새로고침해 주세요. ${message || ""}`, true);
  });
  global.addEventListener("unhandledrejection", (event) => {
    const message = event.reason && event.reason.message ? event.reason.message : String(event.reason || "");
    showToast(`처리하지 못한 작업 오류가 발생했습니다. 다시 시도해 주세요. ${message}`, true);
  });

  initialize();
})(window);
