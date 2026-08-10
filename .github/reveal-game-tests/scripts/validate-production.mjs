import { access, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const appDir = path.join(root, "reveal-game");
const required = [
  "index.html", "display.html", "leaderboard.html", "help.html", "offline.html", "manifest.webmanifest", "sw.js", "css/app.css",
  "js/domain.js", "js/state-machine.js", "js/ranking.js", "js/image-loader.js", "js/renderer.js", "js/persistence.js", "js/sync.js", "js/app.js", "js/display.js", "js/leaderboard.js",
  "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png",
];
const errors = [];

async function exists(target) {
  try { await access(target); return true; } catch (error) { return false; }
}

for (const relative of required) if (!await exists(path.join(appDir, relative))) errors.push(`필수 파일 누락: reveal-game/${relative}`);

const htmlFiles = ["index.html", "display.html", "leaderboard.html", "help.html"];
for (const name of htmlFiles) {
  const source = await readFile(path.join(appDir, name), "utf8");
  if (!source.includes('<meta name="robots" content="noindex,nofollow,noarchive">')) errors.push(`${name}: noindex meta 누락`);
  if (/(?:src|href)=["'](?:https?:)?\/\//i.test(source)) errors.push(`${name}: 외부 asset URL 발견`);
  for (const match of source.matchAll(/(?:src|href)="([^"]+)"/g)) {
    const reference = match[1];
    if (!reference || reference.startsWith("#") || reference.startsWith("data:") || reference.startsWith("blob:")) continue;
    const clean = reference.split(/[?#]/)[0];
    if (!clean) continue;
    if (!await exists(path.resolve(appDir, clean))) errors.push(`${name}: 깨진 상대 링크 ${reference}`);
  }
}

for (const name of (await readdir(path.join(appDir, "js"))).filter((entry) => entry.endsWith(".js"))) {
  const source = await readFile(path.join(appDir, "js", name), "utf8");
  if (/\.innerHTML\s*=/.test(source)) errors.push(`${name}: innerHTML 대입 금지 위반`);
  if (/\b(?:TODO|FIXME)\b/.test(source)) errors.push(`${name}: 미해결 TODO/FIXME 발견`);
}

const manifest = JSON.parse(await readFile(path.join(appDir, "manifest.webmanifest"), "utf8"));
if (manifest.start_url !== "./" || manifest.scope !== "./") errors.push("manifest start_url/scope는 ./ 이어야 합니다.");
if (!manifest.icons.some((icon) => icon.sizes === "192x192")) errors.push("manifest 192x192 icon 누락");
if (!manifest.icons.some((icon) => icon.sizes === "512x512" && String(icon.purpose).includes("any"))) errors.push("manifest 512x512 icon 누락");
if (!manifest.icons.some((icon) => String(icon.purpose).includes("maskable"))) errors.push("manifest maskable icon 누락");
for (const [name, expected] of [["icon-192.png", 192], ["icon-512.png", 512], ["icon-maskable-512.png", 512]]) {
  const target = path.join(appDir, "icons", name);
  if (!await exists(target)) continue;
  const data = await readFile(target);
  const pngSignature = data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  const width = data.length >= 24 ? data.readUInt32BE(16) : 0;
  const height = data.length >= 24 ? data.readUInt32BE(20) : 0;
  if (!pngSignature || width !== expected || height !== expected) errors.push(`${name}: 실제 PNG 크기는 ${expected}x${expected}이어야 합니다.`);
}

const sw = await readFile(path.join(appDir, "sw.js"), "utf8");
if (!sw.includes("url.pathname.startsWith(scope.pathname)")) errors.push("service worker scope path guard 누락");
if (/Service-Worker-Allowed/i.test(sw)) errors.push("service worker scope 확장 header를 사용하면 안 됩니다.");
if (/\.\.\//.test(sw)) errors.push("service worker asset 경로가 상위 디렉터리를 참조합니다.");
for (const forbidden of ["localStorage", "indexedDB", "blob:"]) if (sw.includes(forbidden)) errors.push(`service worker 금지 데이터 참조: ${forbidden}`);

const offline = await readFile(path.join(appDir, "offline.html"), "utf8");
if (!offline.includes("build-offline.mjs가 production source에서 생성")) errors.push("offline.html generated marker 누락");
if (/<script\s+src=|<link\s+rel="(?:stylesheet|manifest|icon|apple-touch-icon)"/i.test(offline)) errors.push("offline.html에 외부 production asset 요청이 남아 있습니다.");
if (/(?:src|href)=["']https?:\/\//i.test(offline)) errors.push("offline.html에 network URL이 남아 있습니다.");

if (await exists(path.join(root, ".nojekyll"))) errors.push("repository root에 .nojekyll이 존재합니다.");

const status = execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" });
for (const line of status.split(/\r?\n/).filter(Boolean)) {
  const changed = line.slice(3).replace(/^"|"$/g, "");
  if (!(changed.startsWith("reveal-game/") || changed.startsWith(".github/reveal-game-tests/") || changed === ".github/workflows/reveal-game-ci.yml")) {
    errors.push(`허용 경로 밖 변경: ${changed}`);
  }
}

if (errors.length) {
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else console.log(`production 정적 검증 통과: 필수 파일 ${required.length}개, 외부 runtime dependency 없음, scope와 변경 경계 정상`);
