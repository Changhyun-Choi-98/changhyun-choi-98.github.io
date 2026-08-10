import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { solidPng, encodePng } from "./png-utils.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const fixtureDir = path.resolve(scriptDir, "../fixtures/generated/images");
export const corruptFixtureDir = path.resolve(scriptDir, "../fixtures/generated/corrupt");
export const emptyFixtureDir = path.resolve(scriptDir, "../fixtures/generated/empty");
export const representativeFixtureDir = path.resolve(scriptDir, "../fixtures/generated/representative");

export async function generateFixtures() {
  await rm(path.dirname(fixtureDir), { recursive: true, force: true });
  await Promise.all([
    mkdir(path.join(fixtureDir, "nested"), { recursive: true }),
    mkdir(corruptFixtureDir, { recursive: true }),
    mkdir(emptyFixtureDir, { recursive: true }),
    mkdir(representativeFixtureDir, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(fixtureDir, "image1-red.png"), solidPng(160, 90, [220, 38, 60, 255])),
    writeFile(path.join(fixtureDir, "image2-green.PNG"), solidPng(90, 160, [29, 190, 105, 255])),
    writeFile(path.join(fixtureDir, "nested/image10-blue.png"), encodePng(128, 128, (x, y) => [30, 105, 230, x < 32 && y < 32 ? 90 : 255])),
    writeFile(path.join(fixtureDir, "무시할-문서.txt"), "합성 fixture의 지원하지 않는 파일입니다.\n", "utf8"),
    writeFile(path.join(corruptFixtureDir, "broken.jpg"), "이미지 형식이 아닌 합성 오류 fixture\n", "utf8"),
    writeFile(path.join(emptyFixtureDir, "no-image.txt"), "지원 이미지가 없는 폴더 fixture\n", "utf8"),
    writeFile(path.join(representativeFixtureDir, "quadrants.png"), encodePng(100, 100, (x, y) => {
      if (x < 50 && y < 50) return [255, 0, 0, 255];
      if (x >= 50 && y < 50) return [0, 255, 0, 255];
      if (x < 50) return [0, 0, 255, 255];
      return [255, 255, 255, 255];
    })),
  ]);
}

if (path.resolve(process.argv[1] || "") === fileURLToPath(import.meta.url)) await generateFixtures();
