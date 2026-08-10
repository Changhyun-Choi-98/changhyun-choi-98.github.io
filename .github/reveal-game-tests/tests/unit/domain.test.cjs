const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const root = path.resolve(__dirname, "../../../..");
const Domain = require(path.join(root, "reveal-game/js/domain.js"));

test("시간 파싱과 표시가 지원 형식을 왕복한다", () => {
  const cases = [
    ["5", 5000], ["5.125", 5125], ["01:05", 65000], ["02:03.4", 123400],
    ["01:02:03", 3723000], ["01:02:03.045", 3723045],
  ];
  for (const [input, expected] of cases) {
    const parsed = Domain.parseDuration(input);
    assert.equal(parsed.ok, true, input);
    assert.equal(parsed.milliseconds, expected, input);
  }
  assert.equal(Domain.formatDuration(0), "00:00.0");
  assert.equal(Domain.formatDuration(3723456), "01:02:03.4");
  assert.equal(Domain.formatDurationPrecise(3723456), "01:02:03.456");
  assert.equal(Domain.parseDuration("1:60").ok, false);
  assert.equal(Domain.parseDuration("-1").ok, false);
});

test("mosaic은 landscape, portrait, square의 1x1과 원본 endpoint를 보장한다", () => {
  for (const [width, height] of [[1600, 900], [900, 1600], [1024, 1024]]) {
    assert.deepEqual(Domain.mosaicGrid(width, height, 0), { width: 1, height: 1, longSide: 1, original: false });
    const end = Domain.mosaicGrid(width, height, 1);
    assert.equal(end.width, width);
    assert.equal(end.height, height);
    assert.equal(end.original, true);
    const middle = Domain.mosaicGrid(width, height, 0.5);
    assert.ok(Math.abs(middle.width / middle.height - width / height) < 0.08);
  }
});

test("mosaic 단계는 중복 없이 단조 증가하고 profile 차이를 반영한다", () => {
  const levels = Domain.buildMosaicLevels(6000, 4000, 96);
  assert.equal(levels[0], 1);
  assert.equal(levels.at(-1), 6000);
  assert.ok(levels.length >= 64 && levels.length <= 128);
  assert.equal(new Set(levels).size, levels.length);
  for (let index = 1; index < levels.length; index += 1) assert.ok(levels[index] > levels[index - 1]);

  let previous = 0;
  for (let step = 0; step <= 100; step += 1) {
    const grid = Domain.mosaicGrid(6000, 4000, step / 100, "balanced", levels);
    assert.ok(grid.longSide >= previous);
    previous = grid.longSide;
  }
  const slow = Domain.mosaicGrid(6000, 4000, 0.5, "slow", levels).longSide;
  const balanced = Domain.mosaicGrid(6000, 4000, 0.5, "balanced", levels).longSide;
  const fast = Domain.mosaicGrid(6000, 4000, 0.5, "fast", levels).longSide;
  assert.ok(slow < balanced && balanced < fast);
  assert.throws(() => Domain.buildMosaicLevels(0, 100), /0보다/);
});

test("circle은 시작 시 모든 corner를 덮고 끝에서 0이며 단조 감소한다", () => {
  const rectangle = { x: 20, y: 40, width: 1600, height: 900 };
  const start = Domain.circleGeometry(rectangle, 0, "balanced", 2);
  const corners = [[20, 40], [1620, 40], [20, 940], [1620, 940]];
  for (const [x, y] of corners) assert.ok(Math.hypot(x - start.cx, y - start.cy) < start.radius);
  assert.equal(start.cx, 820);
  assert.equal(start.cy, 490);
  assert.deepEqual(start.clip, rectangle);
  assert.equal(Domain.circleGeometry(rectangle, 1).radius, 0);

  let previous = Infinity;
  for (let step = 0; step <= 100; step += 1) {
    const radius = Domain.circleGeometry(rectangle, step / 100).radius;
    assert.ok(radius <= previous);
    previous = radius;
  }
  const slow = Domain.circleGeometry(rectangle, 0.5, "slow").radius;
  const fast = Domain.circleGeometry(rectangle, 0.5, "fast").radius;
  assert.ok(slow > fast);
});

test("이미지 확장자, 자연 정렬, 무작위 순서가 정확하다", () => {
  for (const name of ["a.JPG", "b.jpeg", "c.PNG", "d.WebP", "e.BMP"]) assert.equal(Domain.isSupportedImageName(name), true);
  assert.equal(Domain.isSupportedImageName("notes.txt"), false);
  const sorted = ["folder/image10.jpg", "folder/image2.jpg", "folder/image1.jpg"].sort(Domain.naturalCompare);
  assert.deepEqual(sorted, ["folder/image1.jpg", "folder/image2.jpg", "folder/image10.jpg"]);
  const shuffled = Domain.fisherYates([1, 2, 3, 4], () => 0);
  assert.equal(shuffled.length, 4);
  assert.deepEqual(new Set(shuffled), new Set([1, 2, 3, 4]));
});
