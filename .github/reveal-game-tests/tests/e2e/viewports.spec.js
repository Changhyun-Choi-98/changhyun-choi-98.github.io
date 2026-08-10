const { test, expect } = require("@playwright/test");
const { loadFixtureFolder, monitorPage } = require("./helpers");

async function seedRankings(page, count = 20) {
  const records = Array.from({ length: count }, (_, index) => ({
    id: `viewport-team-${index}`,
    teamName: `화면 점검 ${String(index + 1).padStart(2, "0")}팀`,
    normalizedTeamName: `화면 점검 ${String(index + 1).padStart(2, "0")}팀`,
    correctCount: count - index,
    elapsedMs: 30000 + index * 1250,
    createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
    updatedAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
  }));
  await page.evaluate(async (rankings) => {
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open("image-reveal-game", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(["rankings", "snapshots"], "readwrite");
      const store = transaction.objectStore("rankings");
      store.clear();
      for (const ranking of rankings) store.put(ranking);
      transaction.objectStore("snapshots").put({ key: "last", updatedAt: new Date().toISOString() });
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
    });
    database.close();
  }, records);
}

for (const viewport of [{ width: 1366, height: 768 }, { width: 1920, height: 1080 }, { width: 3840, height: 2160 }]) {
  test(`controller, participant, leaderboard ${viewport.width}x${viewport.height} smoke와 screenshot`, async ({ page }, testInfo) => {
    const assertClean = monitorPage(page);
    await page.setViewportSize(viewport);
    await page.goto("/reveal-game/");
    await loadFixtureFolder(page);
    await expect(page.locator("#gameCanvas")).toBeVisible();
    await expect(page.locator("#controllerPanel")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath(`controller-${viewport.width}x${viewport.height}.png`), fullPage: true });

    const popupPromise = page.waitForEvent("popup");
    await page.locator("#openParticipantButton").click();
    const participant = await popupPromise;
    const assertParticipantClean = monitorPage(participant);
    await participant.setViewportSize(viewport);
    await expect(participant.locator("#displayCanvas")).toHaveAttribute("data-grid", "1x1");
    await participant.screenshot({ path: testInfo.outputPath(`participant-${viewport.width}x${viewport.height}.png`), fullPage: true });
    assertParticipantClean();
    await participant.close();

    await seedRankings(page);
    await page.goto("/reveal-game/leaderboard.html");
    await expect(page.locator("h1")).toHaveText("랭킹 보드");
    await expect(page.locator("#podium .podium-card")).toHaveCount(3);
    await expect(page.locator("#leaderboardBody tr")).toHaveCount(17);
    await page.screenshot({ path: testInfo.outputPath(`leaderboard-${viewport.width}x${viewport.height}.png`), fullPage: true });
    assertClean();
  });
}

for (const viewport of [{ width: 1093, height: 614 }, { width: 911, height: 512 }]) {
  test(`Windows 확대 대응 ${viewport.width}x${viewport.height} 조작 smoke`, async ({ page }) => {
    const assertClean = monitorPage(page);
    await page.setViewportSize(viewport);
    await page.goto("/reveal-game/");
    await loadFixtureFolder(page);
    const columns = await page.locator(".game-layout").evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(" ").length);
    expect(columns).toBe(2);
    await page.locator("#startButton").scrollIntoViewIfNeeded();
    await page.locator("#startButton").click();
    await expect(page.locator("#gameState")).toHaveText("공개 중");
    assertClean();
  });
}
