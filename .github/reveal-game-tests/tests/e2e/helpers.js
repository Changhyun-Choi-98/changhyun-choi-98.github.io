const path = require("node:path");
const { expect } = require("@playwright/test");

const repositoryRoot = path.resolve(__dirname, "../../../..");
const fixtureDirectory = path.join(repositoryRoot, ".github/reveal-game-tests/fixtures/generated/images");
const corruptFixtureDirectory = path.join(repositoryRoot, ".github/reveal-game-tests/fixtures/generated/corrupt");
const emptyFixtureDirectory = path.join(repositoryRoot, ".github/reveal-game-tests/fixtures/generated/empty");
const representativeFixtureDirectory = path.join(repositoryRoot, ".github/reveal-game-tests/fixtures/generated/representative");
const offlinePath = path.join(repositoryRoot, "reveal-game/offline.html");

async function loadFixtureFolder(page) {
  await page.locator("#folderInput").setInputFiles(fixtureDirectory);
  await expect(page.locator("#gameState")).toHaveText("시작 준비");
  await expect(page.locator("#imageCountLabel")).toHaveText("3개");
  await expect(page.locator("#folderMessage")).toContainText("지원하지 않는 파일 1개");
  await expect(page.locator("#gameCanvas")).toHaveAttribute("data-grid", "1x1");
}

async function centerPixel(page, selector = "#gameCanvas") {
  return page.locator(selector).evaluate((canvas) => {
    const context = canvas.getContext("2d");
    return Array.from(context.getImageData(Math.floor(canvas.width / 2), Math.floor(canvas.height / 2), 1, 1).data);
  });
}

function monitorPage(page, options = {}) {
  const failures = [];
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    const ignored = (options.ignoreConsolePatterns || []).some((pattern) => pattern.test(message.text()));
    if (message.type() === "error" && !ignored) failures.push(`console: ${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    const allowed = options.allowFailedRequest && options.allowFailedRequest(request);
    if (!allowed) failures.push(`requestfailed: ${request.url()} ${request.failure() && request.failure().errorText}`);
  });
  return () => expect(failures, failures.join("\n")).toEqual([]);
}

module.exports = { repositoryRoot, fixtureDirectory, corruptFixtureDirectory, emptyFixtureDirectory, representativeFixtureDirectory, offlinePath, loadFixtureFolder, centerPixel, monitorPage };
