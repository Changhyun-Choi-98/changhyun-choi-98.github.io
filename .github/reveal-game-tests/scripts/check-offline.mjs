import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildOfflineSource } from "./build-offline.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const outputPath = path.join(root, "reveal-game/offline.html");
const [expected, actual] = await Promise.all([buildOfflineSource(), readFile(outputPath, "utf8")]);
if (expected !== actual) {
  console.error("offline.html이 production source와 일치하지 않습니다. npm run build:offline을 실행하세요.");
  process.exitCode = 1;
} else {
  console.log("offline.html 재생성 결과가 tracked artifact와 일치합니다.");
}
