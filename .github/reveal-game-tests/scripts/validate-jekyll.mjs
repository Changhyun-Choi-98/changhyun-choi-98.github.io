import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const site = path.join(root, "_site");
const required = [
  "index.html", "profile/index.html", "paper/index.html", "study/index.html",
  "reveal-game/index.html", "reveal-game/display.html", "reveal-game/leaderboard.html", "reveal-game/help.html", "reveal-game/offline.html", "reveal-game/sw.js",
  "reveal-game/manifest.webmanifest", "reveal-game/icons/icon-192.png", "reveal-game/icons/icon-512.png", "reveal-game/icons/icon-maskable-512.png",
];
const errors = [];
for (const relative of required) {
  try { await access(path.join(site, relative)); } catch (error) { errors.push(`_site/${relative} 누락`); }
}
try { await access(path.join(site, ".github")); errors.push("개발 test directory가 _site에 노출되었습니다."); } catch (error) { /* Expected. */ }
try { await access(path.join(root, ".nojekyll")); errors.push("repository root .nojekyll 금지 위반"); } catch (error) { /* Expected. */ }
const builtSw = await readFile(path.join(site, "reveal-game/sw.js"), "utf8");
if (!builtSw.includes("url.pathname.startsWith(scope.pathname)")) errors.push("built service worker scope guard 누락");
if (builtSw.includes("Service-Worker-Allowed")) errors.push("built service worker가 broad scope를 요청합니다.");
if (errors.length) {
  errors.forEach((error) => console.error(`- ${error}`));
  process.exitCode = 1;
} else console.log(`Jekyll integration 검증 통과: 앱 ${required.length - 4}개 asset과 기존 주요 페이지 4개 확인`);
