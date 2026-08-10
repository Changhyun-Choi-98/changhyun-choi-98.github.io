import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const appDir = path.join(root, "reveal-game");
const outputPath = path.join(appDir, "offline.html");

export async function buildOfflineSource() {
  let html = await readFile(path.join(appDir, "index.html"), "utf8");
  const css = await readFile(path.join(appDir, "css/app.css"), "utf8");
  html = html
    .replace(/\s*<link rel="manifest"[^>]*>/g, "")
    .replace(/\s*<link rel="(?:icon|apple-touch-icon)"[^>]*>/g, "")
    .replace(/<link rel="stylesheet" href="css\/app\.css">/, `<style>\n${css}\n  </style>`)
    .replace("<title>이미지 공개 퀴즈</title>", "<title>이미지 공개 퀴즈 · 비상용 단일 파일</title>")
    .replace("<body class=\"controller-page\">", "<body class=\"controller-page\" data-standalone=\"true\">")
    .replace('href="help.html" target="_blank" rel="noopener"', 'href="#helpPanel" data-offline-help-link="true"')
    .replace('href="offline.html" download="이미지-공개-퀴즈-offline.html"', 'href="#" download="이미지-공개-퀴즈-offline.html"');

  const scriptPattern = /<script src="(js\/[^"]+)"><\/script>/g;
  const scripts = [...html.matchAll(scriptPattern)].map((match) => match[1]);
  for (const relative of scripts) {
    const source = await readFile(path.join(appDir, relative), "utf8");
    html = html.replace(`<script src="${relative}"></script>`, `<script>\n${source.replace(/<\/script/gi, "<\\/script")}\n  </script>`);
  }
  html = html.replace("<!doctype html>", "<!doctype html>\n<!-- 이 파일은 build-offline.mjs가 production source에서 생성합니다. 직접 편집하지 마세요. -->");
  return html.endsWith("\n") ? html : `${html}\n`;
}

export async function writeOffline() {
  const source = await buildOfflineSource();
  await writeFile(outputPath, source, "utf8");
  return outputPath;
}

if (path.resolve(process.argv[1] || "") === fileURLToPath(import.meta.url)) await writeOffline();
