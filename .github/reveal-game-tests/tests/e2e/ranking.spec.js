const { test, expect } = require("@playwright/test");
const { monitorPage } = require("./helpers");

async function openRanking(page) {
  await page.getByRole("tab", { name: "랭킹", exact: true }).click();
  await expect(page.locator("#rankingPanel")).toBeVisible();
}

async function addTeam(page, team, score, time) {
  const previousCount = await page.locator("#rankingBody tr").count();
  await page.locator("#teamNameInput").fill(team);
  await page.locator("#correctCountInput").fill(String(score));
  await page.locator("#elapsedInput").fill(time);
  await page.locator("#rankingSubmitButton").click();
  await expect(page.locator("#rankingBody tr")).toHaveCount(previousCount + 1);
  await expect(page.locator("#rankingSubmitButton")).toBeEnabled();
}

test("랭킹 CRUD, 공동 순위, reload persistence가 동작한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await openRanking(page);
  await addTeam(page, "가람 팀", 5, "01:10.250");
  await addTeam(page, "나래 팀", 5, "01:10.250");
  const rows = page.locator("#rankingBody tr");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(0).locator("td").nth(0)).toHaveText("1위");
  await expect(rows.nth(1).locator("td").nth(0)).toHaveText("1위");

  await page.reload();
  await openRanking(page);
  await expect(page.locator("#rankingBody")).toContainText("가람 팀");
  await expect(page.locator("#rankingBody")).toContainText("나래 팀");
  const firstRow = page.locator("#rankingBody tr").filter({ hasText: "가람 팀" });
  await firstRow.getByRole("button", { name: "수정" }).click();
  await page.locator("#correctCountInput").fill("6");
  await page.locator("#rankingSubmitButton").click();
  await expect(page.locator("#rankingBody tr").first().locator("td").nth(1)).toHaveText("가람 팀");

  await page.locator("#teamNameInput").fill("  가람   팀  ");
  await page.locator("#correctCountInput").fill("1");
  await page.locator("#elapsedInput").fill("10");
  await page.locator("#rankingSubmitButton").click();
  await expect(page.locator("#teamNameError")).toContainText("이미");
  const secondRow = page.locator("#rankingBody tr").filter({ hasText: "나래 팀" });
  await secondRow.getByRole("button", { name: "삭제" }).click();
  await page.locator("#modalActions").getByRole("button", { name: "삭제", exact: true }).click();
  await expect(page.locator("#rankingMessage")).toContainText("팀을 삭제했습니다");
  await page.reload();
  await openRanking(page);
  await expect(page.locator("#rankingBody")).not.toContainText("나래 팀");
  assertClean();
});

test("JSON/CSV export와 preview import 교체·병합이 동작한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await openRanking(page);
  await addTeam(page, "=수식 방지 팀", 4, "62.345");

  const jsonDownloadPromise = page.waitForEvent("download");
  await page.locator("#exportJsonButton").click();
  const jsonDownload = await jsonDownloadPromise;
  const jsonPath = await jsonDownload.path();
  const jsonText = require("node:fs").readFileSync(jsonPath, "utf8");
  expect(JSON.parse(jsonText).schemaVersion).toBe(1);

  const csvDownloadPromise = page.waitForEvent("download");
  await page.locator("#exportCsvButton").click();
  const csvDownload = await csvDownloadPromise;
  const csvPath = await csvDownload.path();
  const csvText = require("node:fs").readFileSync(csvPath, "utf8");
  expect(csvText.charCodeAt(0)).toBe(0xfeff);
  expect(csvText).toContain("'=수식 방지 팀");

  await page.locator("#clearRankingsButton").click();
  await page.locator("#modalActions").getByRole("button", { name: "전체 삭제" }).click();
  await expect(page.locator("#rankingBody tr")).toHaveCount(0);
  await page.locator("#importJsonInput").setInputFiles({ name: "backup.json", mimeType: "application/json", buffer: Buffer.from(jsonText) });
  await expect(page.locator("#modalTitle")).toHaveText("가져오기 미리보기");
  await page.getByRole("button", { name: "기존 데이터 교체" }).click();
  await expect(page.locator("#rankingBody")).toContainText("=수식 방지 팀");

  const extraCsv = "\uFEFF\"순위\",\"팀명\",\"맞힌 이미지 개수\",\"소요 시간\"\r\n\"1\",\"새 팀\",\"7\",\"00:40.001\"\r\n\"2\",\"오류 팀\",\"x\",\"bad\"\r\n";
  await page.locator("#importCsvInput").setInputFiles({ name: "ranking.csv", mimeType: "text/csv", buffer: Buffer.from(extraCsv) });
  await expect(page.locator("#modalMessage")).toContainText("잘못된 행 1개");
  await page.getByRole("button", { name: "기존 데이터와 병합" }).click();
  await expect(page.locator("#rankingBody")).toContainText("새 팀");
  assertClean();
});

test("별도 랭킹 창이 진행자 창의 수정을 즉시 반영한다", async ({ page }) => {
  const assertControllerClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await openRanking(page);
  await addTeam(page, "첫 번째 팀", 3, "30.100");
  await page.getByRole("tab", { name: "게임", exact: true }).click();
  const popupPromise = page.waitForEvent("popup");
  await page.locator("#openLeaderboardButton").click();
  const leaderboard = await popupPromise;
  const assertLeaderboardClean = monitorPage(leaderboard);
  await expect(leaderboard.locator("#podium")).toContainText("첫 번째 팀");

  await openRanking(page);
  await addTeam(page, "두 번째 팀", 5, "20.250");
  await expect(leaderboard.locator("#podium")).toContainText("두 번째 팀");
  assertLeaderboardClean();
  assertControllerClean();
});

test("IndexedDB보다 최신인 로컬 복구 사본을 우선 복원한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await page.evaluate(async () => {
    const settings = { mode: "mosaic", order: "sequential", profile: "balanced", durationMs: 45000 };
    const makeRecord = (id, teamName) => ({
      id, teamName, normalizedTeamName: teamName, correctCount: 1, elapsedMs: 1000,
      createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z",
    });
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open("image-reveal-game", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(["rankings", "settings", "snapshots"], "readwrite");
      transaction.objectStore("rankings").clear();
      transaction.objectStore("rankings").put(makeRecord("old", "오래된 팀"));
      transaction.objectStore("settings").put({ key: "game", value: settings });
      transaction.objectStore("snapshots").put({ key: "last", updatedAt: "2026-01-01T00:00:00.000Z" });
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
    });
    database.close();
    localStorage.setItem("image-reveal-game:last-good-snapshot:v1", JSON.stringify({
      rankings: [makeRecord("new", "최신 복구 팀")], settings, updatedAt: "2026-01-02T00:00:00.000Z",
    }));
  });
  await page.reload();
  await openRanking(page);
  await expect(page.locator("#rankingBody")).toContainText("최신 복구 팀");
  await expect(page.locator("#rankingBody")).not.toContainText("오래된 팀");
  assertClean();
});

test("최종 메모리 대체 상태에서도 별도 랭킹 창을 직접 동기화한다", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(globalThis, "indexedDB", { configurable: true, value: null });
    Object.defineProperty(globalThis, "BroadcastChannel", { configurable: true, value: undefined });
    Storage.prototype.setItem = () => { throw new Error("합성 저장소 거부"); };
  });
  const assertControllerClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await expect(page.locator("#storageStatus")).toHaveText("메모리에만 저장");
  await openRanking(page);
  await addTeam(page, "메모리 팀", 2, "10.5");
  await page.getByRole("tab", { name: "게임", exact: true }).click();
  const popupPromise = page.waitForEvent("popup");
  await page.locator("#openLeaderboardButton").click();
  const leaderboard = await popupPromise;
  const assertLeaderboardClean = monitorPage(leaderboard);
  await expect(leaderboard.locator("#podium")).toContainText("메모리 팀");
  assertLeaderboardClean();
  assertControllerClean();
});
