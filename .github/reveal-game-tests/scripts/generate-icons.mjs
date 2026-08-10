import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { encodePng } from "./png-utils.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const outputDir = path.join(root, "reveal-game/icons");

function icon(size, maskable) {
  return encodePng(size, size, (x, y) => {
    const center = size / 2;
    const dx = x - center;
    const dy = y - center;
    const distance = Math.hypot(dx, dy);
    const safe = maskable ? size * 0.32 : size * 0.42;
    if (distance > safe) return [11, 18, 32, 255];
    const cell = Math.max(1, Math.floor(size / 12));
    const checker = (Math.floor(x / cell) + Math.floor(y / cell)) % 2;
    if (distance < size * 0.13) return [7, 11, 18, 255];
    return checker ? [102, 217, 255, 255] : [31, 111, 235, 255];
  });
}

await mkdir(outputDir, { recursive: true });
await Promise.all([
  writeFile(path.join(outputDir, "icon-192.png"), icon(192, false)),
  writeFile(path.join(outputDir, "icon-512.png"), icon(512, false)),
  writeFile(path.join(outputDir, "icon-maskable-512.png"), icon(512, true)),
]);
