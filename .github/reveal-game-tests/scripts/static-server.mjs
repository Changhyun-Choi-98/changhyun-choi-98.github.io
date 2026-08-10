import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../../..");
const port = Number(process.env.REVEAL_GAME_PORT || 4173);
let simulatedOffline = false;
const mime = new Map([
  [".html", "text/html; charset=utf-8"], [".css", "text/css; charset=utf-8"], [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"], [".webmanifest", "application/manifest+json; charset=utf-8"], [".png", "image/png"], [".webp", "image/webp"],
]);

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host}`);
    if (url.pathname === "/__reveal_game_test__/network") {
      simulatedOffline = url.searchParams.get("state") === "offline";
      response.writeHead(204, { "Cache-Control": "no-store" });
      response.end();
      return;
    }
    if (simulatedOffline) {
      request.socket.destroy();
      return;
    }
    let relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    if (!relative || relative.endsWith("/")) relative += "index.html";
    const resolved = path.resolve(root, relative);
    if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) throw new Error("잘못된 경로");
    const info = await stat(resolved);
    if (!info.isFile()) throw new Error("파일 아님");
    const content = await readFile(resolved);
    response.writeHead(200, { "Content-Type": mime.get(path.extname(resolved).toLowerCase()) || "application/octet-stream", "Cache-Control": "no-cache" });
    response.end(content);
  } catch (error) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("찾을 수 없습니다.");
  }
});

server.listen(port, "127.0.0.1", () => console.log(`Reveal game test server: http://127.0.0.1:${port}`));

for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.close(() => process.exit(0)));
