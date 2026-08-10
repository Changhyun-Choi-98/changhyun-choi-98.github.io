const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const root = path.resolve(__dirname, "../../../..");
require(path.join(root, "reveal-game/js/domain.js"));
const Ranking = require(path.join(root, "reveal-game/js/ranking.js"));

function record(id, teamName, correctCount, elapsedMs, createdAt = "2026-01-01T00:00:00.000Z") {
  return { id, teamName, normalizedTeamName: Ranking.normalizeTeamName(teamName), correctCount, elapsedMs, createdAt, updatedAt: createdAt };
}

test("정답 내림차순, 시간 오름차순, competition ranking 1,1,3을 적용한다", () => {
  const sorted = Ranking.sortAndRank([
    record("c", "다팀", 8, 9000), record("a", "가팀", 10, 12000), record("b", "나팀", 10, 12000), record("d", "라팀", 10, 15000),
  ]);
  assert.deepEqual(sorted.map((item) => item.teamName), ["가팀", "나팀", "라팀", "다팀"]);
  assert.deepEqual(sorted.map((item) => item.rank), [1, 1, 3, 4]);
});

test("동점 팀 화면 순서는 Korean name, createdAt, id 순서로 안정적이다", () => {
  const sorted = Ranking.sortAndRank([
    record("z", "하늘", 3, 1000, "2026-01-02"), record("b", "가람", 3, 1000, "2026-01-03"), record("a", "가람", 3, 1000, "2026-01-02"),
  ]);
  assert.deepEqual(sorted.map((item) => item.id), ["a", "b", "z"]);
  assert.deepEqual(sorted.map((item) => item.rank), [1, 1, 1]);
});

test("입력 validation과 대소문자/공백 정규화 duplicate를 처리한다", () => {
  const existing = [record("1", " Robot 팀 ", 2, 1000)];
  assert.equal(Ranking.validateRankingInput({ teamName: "", correctCount: "", elapsed: "" }, existing).ok, false);
  const duplicate = Ranking.validateRankingInput({ teamName: "robot   팀", correctCount: "2", elapsed: "10" }, existing);
  assert.equal(duplicate.ok, false);
  assert.match(duplicate.errors.teamName, /이미/);
  assert.equal(Ranking.validateRankingInput({ teamName: "새 팀", correctCount: "2.5", elapsed: "10" }).ok, false);
  assert.equal(Ranking.validateRankingInput({ teamName: "새 팀", correctCount: "2", elapsed: "01:02.123" }).value.elapsedMs, 62123);
});

test("JSON schema를 검증하고 잘못된 record를 거부한다", () => {
  const settings = { mode: "mosaic", order: "sequential", profile: "balanced", durationMs: 45000 };
  const valid = Ranking.makeBackup([record("1", "팀", 1, 1000)], settings, "2026-01-01T00:00:00.000Z");
  assert.equal(Ranking.validateBackup(valid).ok, true);
  assert.equal(Ranking.validateBackup({ ...valid, schemaVersion: 99 }).ok, false);
  assert.equal(Ranking.validateBackup({ ...valid, rankings: [{ teamName: "", correctCount: -1, elapsedMs: -1 }] }).ok, false);
  assert.equal(Ranking.validateBackup({ ...valid, rankings: [record("1", "가팀", 1, 1000), record("1", "나팀", 2, 900)] }).ok, false);
  assert.equal(Ranking.validateBackup({ ...valid, exportedAt: "잘못된 날짜" }).ok, false);
  assert.equal(Ranking.validateBackup({ ...valid, settings: { ...settings, durationMs: 1 } }).ok, false);
});

test("CSV quote, comma, formula injection 보호와 import 재계산을 처리한다", () => {
  const source = [record("1", '=합계,"팀"', 5, 62345)];
  const csv = Ranking.exportCsv(source);
  assert.ok(csv.startsWith("\uFEFF"));
  assert.match(csv, /'=합계/);
  assert.match(csv, /""팀""/);
  const parsed = Ranking.parseCsv(csv, "2026-01-01T00:00:00.000Z");
  assert.equal(parsed.ok, true);
  assert.equal(parsed.rankings[0].teamName, '=합계,"팀"');
  assert.equal(parsed.rankings[0].correctCount, 5);
  assert.equal(parsed.rankings[0].elapsedMs, 62345);
});

test("CSV malformed row는 전체를 crash하지 않고 오류를 보고한다", () => {
  const csv = "팀명,맞힌 이미지 개수,소요 시간\n정상팀,3,10.5\n오류팀,abc,잘못됨\n";
  const parsed = Ranking.parseCsv(csv);
  assert.equal(parsed.rankings.length, 1);
  assert.equal(parsed.errors.length, 1);
  assert.equal(Ranking.parseCsvRows('"닫히지 않음').ok, false);
});

test("병합 중 팀 id 충돌은 두 팀을 보존하도록 새 id를 부여한다", () => {
  const merged = Ranking.mergeRankings(
    [record("same-id", "기존 팀", 3, 1000)],
    [record("same-id", "새 팀", 4, 900)],
  );
  assert.equal(merged.rankings.length, 2);
  assert.equal(new Set(merged.rankings.map((item) => item.id)).size, 2);
  assert.deepEqual(merged.rankings.map((item) => item.teamName), ["기존 팀", "새 팀"]);
});
