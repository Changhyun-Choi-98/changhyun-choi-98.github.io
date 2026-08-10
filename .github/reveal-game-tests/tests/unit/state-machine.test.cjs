const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const root = path.resolve(__dirname, "../../../..");
require(path.join(root, "reveal-game/js/domain.js"));
const Machine = require(path.join(root, "reveal-game/js/state-machine.js"));
const { STATES } = Machine;

function dispatch(model, type, now, extra = {}) {
  const result = Machine.transition(model, { type, ...extra }, now);
  assert.equal(result.accepted, true, `${type}: ${result.reason}`);
  return result.state;
}

test("EMPTY에서 load 후 READY, RUNNING, PAUSED, resume 전이가 정확하다", () => {
  let model = Machine.createEmpty();
  assert.equal(model.status, STATES.EMPTY);
  model = dispatch(model, "LOAD_IMAGES", 0, { imageCount: 2 });
  assert.equal(model.status, STATES.READY);
  model = dispatch(model, "START", 100);
  assert.equal(model.status, STATES.RUNNING);
  model = dispatch(model, "PAUSE", 1100);
  assert.equal(model.status, STATES.PAUSED);
  assert.equal(Machine.currentElapsed(model, 9000), 1000);
  model = dispatch(model, "START", 9000);
  assert.equal(Machine.currentElapsed(model, 9500), 1500);
});

test("여러 pause/resume와 modal freeze 시간은 누적하지 않는다", () => {
  let model = Machine.createSession(1, { durationMs: 45000 });
  model = dispatch(model, "START", 0);
  model = dispatch(model, "PAUSE", 1000);
  model = dispatch(model, "START", 10000);
  model = dispatch(model, "PAUSE", 10500);
  model = dispatch(model, "START", 50000);
  assert.equal(Machine.currentElapsed(model, 50250), 1750);
  assert.equal(Machine.sessionElapsed(model, 50250), 1750);
});

test("정답 공개는 자동 집계하지 않고 성공/실패는 한 번만 집계한다", () => {
  let model = Machine.createSession(2);
  model = dispatch(model, "START", 0);
  model = dispatch(model, "REVEAL", 1200);
  assert.equal(model.status, STATES.REVEALED_PENDING_RESULT);
  assert.deepEqual(Machine.resultCounts(model), { success: 0, failure: 0 });
  model = dispatch(model, "SUCCESS", 9999);
  assert.equal(model.status, STATES.FINALIZED);
  assert.deepEqual(Machine.resultCounts(model), { success: 1, failure: 0 });
  const duplicate = Machine.transition(model, { type: "SUCCESS" }, 10000);
  assert.equal(duplicate.accepted, false);
  assert.strictEqual(duplicate.state, model);
  assert.deepEqual(Machine.resultCounts(model), { success: 1, failure: 0 });
});

test("Reset은 현재 FINALIZED 기록만 rollback한다", () => {
  let model = Machine.createSession(2);
  model = dispatch(model, "START", 0);
  model = dispatch(model, "SUCCESS", 1000);
  model = dispatch(model, "NEXT", 1100);
  model = dispatch(model, "START", 2000);
  model = dispatch(model, "FAILURE", 3500);
  assert.deepEqual(Machine.resultCounts(model), { success: 1, failure: 1 });
  model = dispatch(model, "RESET", 4000);
  assert.equal(model.status, STATES.READY);
  assert.deepEqual(Machine.resultCounts(model), { success: 1, failure: 0 });
  assert.equal(Machine.sessionElapsed(model, 9999), 1000);
});

test("Next는 판정 뒤에만 가능하고 마지막 라운드에서 완료된다", () => {
  let model = Machine.createSession(1);
  assert.equal(Machine.transition(model, { type: "NEXT" }, 0).accepted, false);
  model = dispatch(model, "START", 0);
  model = dispatch(model, "FAILURE", 2300);
  model = dispatch(model, "NEXT", 2400);
  assert.equal(model.status, STATES.SESSION_COMPLETE);
  assert.deepEqual(Machine.resultCounts(model), { success: 0, failure: 1 });
  assert.equal(Machine.sessionElapsed(model, 999999), 2300);
});

test("duration 완료는 정확히 clamp하고 finalized 전까지 판정을 기다린다", () => {
  let model = Machine.createSession(1, { durationMs: 5000 });
  model = dispatch(model, "START", 100);
  assert.equal(Machine.currentElapsed(model, 9000), 5000);
  model = dispatch(model, "DURATION_REACHED", 9000);
  assert.equal(model.status, STATES.REVEALED_PENDING_RESULT);
  assert.equal(model.round.accumulatedMs, 5000);
  assert.deepEqual(Machine.resultCounts(model), { success: 0, failure: 0 });
  model = dispatch(model, "FAILURE", 20000);
  assert.equal(model.round.finalizedMs, 5000);
});

test("session 누적에는 실패 active time이 들어가고 pause 시간은 빠진다", () => {
  let model = Machine.createSession(2);
  model = dispatch(model, "START", 1000);
  model = dispatch(model, "PAUSE", 3000);
  model = dispatch(model, "START", 10000);
  model = dispatch(model, "SUCCESS", 11000);
  model = dispatch(model, "NEXT", 12000);
  model = dispatch(model, "START", 20000);
  model = dispatch(model, "FAILURE", 24000);
  assert.equal(Machine.sessionElapsed(model, 1000000), 7000);
});

test("monotonic now 주입만 사용하므로 wall clock 변경과 무관하다", () => {
  const originalNow = Date.now;
  let model = Machine.createSession(1);
  try {
    Date.now = () => -999999999;
    model = dispatch(model, "START", 500);
    Date.now = () => 9999999999999;
    assert.equal(Machine.currentElapsed(model, 1500), 1000);
  } finally {
    Date.now = originalNow;
  }
});

test("invalid transition은 원본 state를 변경하지 않는다", () => {
  const model = Machine.createEmpty();
  const result = Machine.transition(model, { type: "START" }, 0);
  assert.equal(result.accepted, false);
  assert.strictEqual(result.state, model);
});
