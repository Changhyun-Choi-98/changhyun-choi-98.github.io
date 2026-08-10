const { test, expect } = require("@playwright/test");
const { pathToFileURL } = require("node:url");
const { loadFixtureFolder, centerPixel, offlinePath, fixtureDirectory, monitorPage } = require("./helpers");

test("참가자 popup이 image와 timer를 동기화하고 닫은 뒤 재연결한다", async ({ page }) => {
  const assertControllerClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await loadFixtureFolder(page);
  const firstPopupPromise = page.waitForEvent("popup");
  await page.locator("#openParticipantButton").click();
  const participant = await firstPopupPromise;
  const assertParticipantClean = monitorPage(participant);
  await expect(participant.locator("#displayConnection")).toHaveText("진행자 화면 연결됨");
  await expect(participant.locator("#displayCanvas")).toHaveAttribute("data-grid", "1x1");
  await expect(participant.locator("body")).not.toContainText("image1-red.png");
  const participantPixel = await centerPixel(participant, "#displayCanvas");
  expect(participantPixel[0]).toBeGreaterThan(180);
  await page.locator("#startButton").click();
  await expect(participant.locator("#displayTimer")).not.toHaveText("00:00.0", { timeout: 3000 });
  await page.locator("#startButton").click();
  const pausedTimer = await participant.locator("#displayTimer").textContent();
  const pausedGrid = await participant.locator("#displayCanvas").getAttribute("data-grid");
  await page.waitForTimeout(2300);
  await expect(participant.locator("#displayConnection")).toHaveText("진행자 화면 연결됨");
  await expect(participant.locator("#displayTimer")).toHaveText(pausedTimer);
  await expect(participant.locator("#displayCanvas")).toHaveAttribute("data-grid", pausedGrid);
  assertParticipantClean();
  await participant.close();

  const reopenedPromise = page.waitForEvent("popup");
  await page.locator("#openParticipantButton").click();
  const reopened = await reopenedPromise;
  const assertReopenedClean = monitorPage(reopened);
  await expect(reopened.locator("#displayConnection")).toHaveText("진행자 화면 연결됨");
  await expect(reopened.locator("#displayTimer")).not.toHaveText("00:00.0");
  assertReopenedClean();
  assertControllerClean();
});

test("진행자 연결이 끊기면 참가자 공개 시간과 화면을 고정한다", async ({ page }) => {
  const assertControllerClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await loadFixtureFolder(page);
  const popupPromise = page.waitForEvent("popup");
  await page.locator("#openParticipantButton").click();
  const participant = await popupPromise;
  const assertParticipantClean = monitorPage(participant);
  await expect(participant.locator("#displayConnection")).toHaveText("진행자 화면 연결됨");
  await page.locator("#startButton").click();
  await expect(participant.locator("#displayTimer")).not.toHaveText("00:00.0", { timeout: 3000 });
  assertControllerClean();
  await page.close();
  await expect(participant.locator("#displayConnection")).toHaveText("진행자 화면과 연결이 끊어졌습니다", { timeout: 4000 });
  const frozenTimer = await participant.locator("#displayTimer").textContent();
  const frozenGrid = await participant.locator("#displayCanvas").getAttribute("data-grid");
  await participant.waitForTimeout(600);
  await expect(participant.locator("#displayTimer")).toHaveText(frozenTimer);
  await expect(participant.locator("#displayCanvas")).toHaveAttribute("data-grid", frozenGrid);
  assertParticipantClean();
});

test("service worker cache 완료 후 browser offline reload가 된다", async ({ page, context, browserName }) => {
  const assertClean = monitorPage(page, {
    allowFailedRequest: (request) => {
      const failure = request.failure();
      const pathname = new URL(request.url()).pathname;
      const expectedOfflineRequest = pathname === "/reveal-game/"
        || (browserName === "webkit" && pathname === "/reveal-game/sw.js");
      return expectedOfflineRequest
        && /internet|network|connection|offline|NSURLErrorDomain|-1005/i.test(failure ? failure.errorText : "");
    },
    ignoreConsolePatterns: browserName === "webkit" ? [/^Failed to load resource: The network connection was lost\.$/] : [],
  });
  await page.goto("/reveal-game/");
  await expect(page.locator("#offlineStatus")).toHaveText("오프라인 사용 준비 완료", { timeout: 15000 });
  const registration = await page.evaluate(async () => {
    const current = await navigator.serviceWorker.ready;
    return { scope: current.scope, controlled: Boolean(navigator.serviceWorker.controller) };
  });
  expect(registration.scope).toContain("/reveal-game/");
  expect(registration.scope.endsWith("/reveal-game/")).toBe(true);
  if (browserName === "webkit") {
    await page.evaluate(() => fetch("/__reveal_game_test__/network?state=offline", { cache: "no-store" }));
    try {
      await page.reload();
    } finally {
      await context.request.get("http://127.0.0.1:4173/__reveal_game_test__/network?state=online");
    }
  } else {
    await context.setOffline(true);
    await page.reload();
  }
  await expect(page).toHaveTitle("이미지 공개 퀴즈");
  await page.evaluate(() => window.dispatchEvent(new Event("offline")));
  await expect(page.locator("#networkStatus")).toHaveText("오프라인");
  if (browserName !== "webkit") await context.setOffline(false);
  assertClean();
});

test("standalone offline.html이 file URL에서 게임, 랭킹, 단일창 참가자 모드를 실행한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto(pathToFileURL(offlinePath).href);
  await expect(page.locator("#offlineStatus")).toHaveText("비상용 단일 파일 모드");
  await page.locator("#folderInput").setInputFiles(fixtureDirectory);
  await expect(page.locator("#gameState")).toHaveText("시작 준비");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "1x1");
  await page.locator("#startButton").click();
  await page.waitForTimeout(150);
  await page.locator("#failureButton").click();
  await expect(page.locator("#failureCount")).toHaveText("1");
  const participantPromise = page.waitForEvent("popup");
  await page.locator("#openParticipantButton").click();
  const participant = await participantPromise;
  await expect(participant).toHaveTitle("참가자 화면 · 이미지 공개 퀴즈");
  await expect(participant.locator("canvas")).toBeVisible();
  await participant.close();
  await page.getByRole("tab", { name: "랭킹", exact: true }).click();
  await page.locator("#teamNameInput").fill("오프라인 팀");
  await page.locator("#correctCountInput").fill("2");
  await page.locator("#elapsedInput").fill("10.125");
  await page.locator("#rankingSubmitButton").click();
  await expect(page.locator("#rankingBody")).toContainText("오프라인 팀");
  await page.getByRole("tab", { name: "게임", exact: true }).click();
  await page.locator("#audienceModeButton").click();
  await expect(page.locator("#controllerPanel")).toBeHidden();
  await page.locator("#exitAudienceModeButton").click();
  await expect(page.locator("#controllerPanel")).toBeVisible();
  assertClean();
});
