const { test, expect } = require("@playwright/test");
const { loadFixtureFolder, centerPixel, monitorPage, fixtureDirectory, corruptFixtureDirectory, emptyFixtureDirectory, representativeFixtureDirectory } = require("./helpers");

test("모자이크 게임의 pause, reveal 확인, reset, next, summary가 정확하다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await loadFixtureFolder(page);
  const firstPixel = await centerPixel(page);
  expect(firstPixel[0]).toBeGreaterThan(180);
  expect(firstPixel[1]).toBeLessThan(80);

  await page.locator("#startButton").click();
  await expect(page.locator("#gameState")).toHaveText("공개 중");
  await page.waitForTimeout(350);
  await page.locator("#startButton").click();
  await expect(page.locator("#gameState")).toHaveText("일시정지");
  const pausedTime = await page.locator("#roundTimer").textContent();
  const pausedGrid = await page.locator("#gameCanvas").getAttribute("data-grid");
  await page.waitForTimeout(350);
  await expect(page.locator("#roundTimer")).toHaveText(pausedTime);
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", pausedGrid);

  await page.locator("#startButton").click();
  await page.locator("#revealButton").click();
  await expect(page.locator("#modalTitle")).toHaveText("정답을 공개할까요?");
  const modalTimer = await page.locator("#roundTimer").textContent();
  const modalGrid = await page.locator("#gameCanvas").getAttribute("data-grid");
  await page.waitForTimeout(350);
  await expect(page.locator("#roundTimer")).toHaveText(modalTimer);
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", modalGrid);
  await page.getByRole("button", { name: "취소" }).click();
  await expect(page.locator("#gameState")).toHaveText("공개 중");
  await expect(page.locator("#roundTimer")).not.toHaveText(modalTimer, { timeout: 1500 });
  await page.locator("#revealButton").click();
  await page.locator("#modalActions").getByRole("button", { name: "정답 공개", exact: true }).click();
  await expect(page.locator("#gameState")).toHaveText("정답 판정 대기");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "original");
  await page.locator("#successButton").click();
  await expect(page.locator("#successCount")).toHaveText("1");
  await expect(page.locator("#nextButton")).toBeEnabled();

  await page.locator("#resetButton").click();
  await page.locator("#modalActions").getByRole("button", { name: "현재 라운드 초기화", exact: true }).click();
  await expect(page.locator("#gameState")).toHaveText("시작 준비");
  await expect(page.locator("#successCount")).toHaveText("0");
  await expect(page.locator("#roundTimer")).toHaveText("00:00.0");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "1x1");

  await page.locator("#startButton").click();
  await page.waitForTimeout(120);
  await page.locator("#failureButton").click();
  await page.locator("#nextButton").click();
  await page.getByRole("button", { name: "계속" }).click();
  await expect(page.locator("#roundLabel")).toContainText("2 / 3");
  const secondPixel = await centerPixel(page);
  expect(secondPixel[1]).toBeGreaterThan(140);

  await page.locator("#startButton").click();
  await page.waitForTimeout(120);
  await page.locator("#successButton").click();
  await page.locator("#nextButton").click();
  await page.getByRole("button", { name: "계속" }).click();
  await expect(page.locator("#roundLabel")).toContainText("3 / 3");
  await page.locator("#startButton").click();
  await page.waitForTimeout(120);
  await page.locator("#failureButton").click();
  await page.locator("#nextButton").click();
  await page.getByRole("button", { name: "계속" }).click();
  await expect(page.locator("#modalTitle")).toHaveText("세션이 완료되었습니다");
  await expect(page.locator("#modalDetails")).toContainText("전체 이미지");
  await expect(page.locator("#modalDetails")).toContainText("3개");
  await expect(page.locator("#successCount")).toHaveText("1");
  await expect(page.locator("#failureCount")).toHaveText("2");
  assertClean();
});

test("무작위 순서, 검은색 원, duration 자동 완료가 동작한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await page.evaluate(() => { window.__testOriginalRandom = Math.random; Math.random = () => 0; });
  await page.locator("#orderSelect").selectOption("random");
  await loadFixtureFolder(page);
  await page.evaluate(() => { Math.random = window.__testOriginalRandom; delete window.__testOriginalRandom; });
  const randomizedFirst = await centerPixel(page);
  expect(randomizedFirst[1]).toBeGreaterThan(140);

  await page.locator("#modeSelect").selectOption("circle");
  await page.locator("#durationInput").fill("5");
  await page.locator("#durationInput").dispatchEvent("change");
  await page.locator("#durationInput").blur();
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "circle");
  const covered = await centerPixel(page);
  expect(covered[0]).toBeLessThan(15);
  expect(covered[1]).toBeLessThan(15);
  expect(covered[2]).toBeLessThan(15);

  await page.locator("#startButton").click();
  await expect(page.locator("#gameState")).toHaveText("공개 중");
  await expect(page.locator("#gameState")).toHaveText("정답 판정 대기", { timeout: 7000 });
  await expect(page.locator("#roundTimer")).toHaveText("00:05.0");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "original");
  await expect(page.locator("#nextButton")).toBeDisabled();
  await page.locator("#failureButton").click();
  await expect(page.locator("#failureCount")).toHaveText("1");
  assertClean();
});

test("빈 폴더와 decode 실패 이미지를 한국어 오류로 복구한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await page.locator("#folderInput").setInputFiles(emptyFixtureDirectory);
  await expect(page.locator("#gameState")).toHaveText("이미지 없음");
  await expect(page.locator("#folderMessage")).toContainText("지원하는 이미지가 없습니다");
  await page.locator("#folderInput").setInputFiles(corruptFixtureDirectory);
  await expect(page.locator("#gameState")).toHaveText("이미지 없음");
  await expect(page.locator("#folderMessage")).toContainText("해석할 수 없는 이미지 1개를 건너뛰었습니다");
  await expect(page.locator("#startButton")).toBeDisabled();
  await expect(page.locator("#newGameButton")).toBeDisabled();
  assertClean();
});

test("이미지 해석이 끝나기 전에는 라운드를 시작하지 않는다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await page.evaluate(() => {
    const prototype = window.RevealGame.ImageLoader.ImagePool.prototype;
    const originalDecode = prototype.decode;
    const gates = new Map();
    for (const index of [0, 1]) {
      let release;
      const promise = new Promise((resolve) => { release = resolve; });
      gates.set(index, { promise, release });
    }
    window.__releaseDecodeGate = (index) => {
      const gate = gates.get(index);
      if (!gate) return;
      gates.delete(index);
      gate.release();
    };
    prototype.decode = async function delayedDecode(index) {
      const gate = gates.get(index);
      if (gate) await gate.promise;
      return originalDecode.call(this, index);
    };
    const createObjectURL = URL.createObjectURL.bind(URL);
    window.__objectUrlCount = 0;
    URL.createObjectURL = (blob) => {
      window.__objectUrlCount += 1;
      return createObjectURL(blob);
    };
  });
  await page.locator("#folderInput").setInputFiles(fixtureDirectory);
  await expect(page.locator("#gameState")).toHaveText("이미지 준비 중");
  await expect(page.locator("#startButton")).toBeDisabled();
  await page.keyboard.press("Space");
  await expect(page.locator("#roundTimer")).toHaveText("00:00.0");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "empty");
  await page.evaluate(() => window.__releaseDecodeGate(0));
  await expect(page.locator("#gameState")).toHaveText("시작 준비", { timeout: 3000 });
  await expect(page.locator("#startButton")).toBeEnabled();
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "1x1");
  await page.locator("#startButton").click();
  await page.locator("#failureButton").click();
  await page.locator("#nextButton").click();
  await page.getByRole("button", { name: "계속" }).click();
  await expect(page.locator("#gameState")).toHaveText("이미지 준비 중");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "empty");
  await expect(page.locator("#startButton")).toBeDisabled();
  await page.evaluate(() => window.__releaseDecodeGate(1));
  await expect(page.locator("#gameState")).toHaveText("시작 준비", { timeout: 3000 });
  await expect.poll(() => page.evaluate(() => window.__objectUrlCount)).toBe(3);
  assertClean();
});

test("1x1 모자이크는 좌상단 픽셀이 아닌 전체 이미지 대표색을 사용한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await page.locator("#folderInput").setInputFiles(representativeFixtureDirectory);
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "1x1");
  const pixel = await centerPixel(page);
  expect(pixel[0]).toBeGreaterThan(90);
  expect(pixel[0]).toBeLessThan(180);
  expect(pixel[1]).toBeGreaterThan(90);
  expect(pixel[1]).toBeLessThan(180);
  expect(pixel[2]).toBeGreaterThan(90);
  expect(pixel[2]).toBeLessThan(180);
  assertClean();
});

test("게임 단축키와 입력 focus 예외, 확인 모달 Esc가 동작한다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await loadFixtureFolder(page);
  await page.locator("#canvasWrap").click();
  await page.keyboard.press("Space");
  await expect(page.locator("#gameState")).toHaveText("공개 중");
  await page.keyboard.press("Space");
  await expect(page.locator("#gameState")).toHaveText("일시정지");
  await page.getByRole("tab", { name: "랭킹", exact: true }).click();
  await page.locator("#teamNameInput").focus();
  await page.keyboard.press("p");
  await expect(page.locator("#failureCount")).toHaveText("0");
  await page.getByRole("tab", { name: "게임", exact: true }).click();
  await page.locator("#canvasWrap").click();
  await page.keyboard.press("Enter");
  await expect(page.locator("#gameState")).toHaveText("판정 완료");
  await page.keyboard.press("Backspace");
  await expect(page.locator("#modalTitle")).toHaveText("현재 라운드를 초기화할까요?");
  const firstModalAction = page.locator("#modalActions button").first();
  const lastModalAction = page.locator("#modalActions button").last();
  await expect(firstModalAction).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(lastModalAction).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(firstModalAction).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.locator("#gameState")).toHaveText("판정 완료");
  await page.keyboard.press("l");
  await expect(page.locator("#rankingPanel")).toBeVisible();
  await page.getByRole("tab", { name: "게임", exact: true }).click();
  await page.locator("#canvasWrap").click();
  await page.keyboard.press("Shift+/");
  await expect(page.locator("#helpPanel")).toBeVisible();
  assertClean();
});

test("버튼 기본 키보드 동작과 브라우저 modifier가 게임 단축키와 충돌하지 않는다", async ({ page }) => {
  const assertClean = monitorPage(page);
  await page.goto("/reveal-game/");
  await loadFixtureFolder(page);
  await page.locator("#startButton").click();
  await page.locator("#startButton").click();
  await page.locator("#failureButton").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#failureCount")).toHaveText("1");
  await expect(page.locator("#successCount")).toHaveText("0");

  await page.locator("#resetButton").click();
  await page.locator("#modalActions").getByRole("button", { name: "현재 라운드 초기화", exact: true }).click();
  await page.locator("#startButton").focus();
  await page.keyboard.press("Space");
  await expect(page.locator("#gameState")).toHaveText("공개 중");

  await page.locator("#gameCanvas").evaluate((canvas) => {
    for (const key of ["r", "p", "n"]) canvas.dispatchEvent(new KeyboardEvent("keydown", { key, ctrlKey: true, bubbles: true }));
  });
  await expect(page.locator("#modalBackdrop")).toBeHidden();
  await expect(page.locator("#failureCount")).toHaveText("0");
  await expect(page.locator("#gameState")).toHaveText("공개 중");
  assertClean();
});
