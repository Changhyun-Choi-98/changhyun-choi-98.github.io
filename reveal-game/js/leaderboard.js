(function initLeaderboard(global) {
  "use strict";

  const RG = global.RevealGame;
  const { Domain, Ranking, Persistence, Sync } = RG;
  const podium = document.getElementById("podium");
  const body = document.getElementById("leaderboardBody");
  const empty = document.getElementById("leaderboardEmpty");
  const storage = document.getElementById("leaderboardStorage");
  const fullscreen = document.getElementById("leaderboardFullscreen");
  let rankings = [];
  let reloadTimer = 0;

  const persistence = new Persistence.Persistence(({ mode }) => {
    storage.textContent = mode === "indexeddb" ? "저장소 정상" : mode === "localstorage" ? "대체 저장소 사용" : "메모리에만 저장";
  });
  const channel = new Sync.CrossWindowChannel("image-reveal-game-ranking-v1", (payload) => {
    if (!payload || payload.type !== "rankings-changed") return;
    if (Array.isArray(payload.rankings)) {
      rankings = payload.rankings.map((record) => ({ ...record }));
      render();
    } else scheduleReload();
  });

  function makeCell(text) {
    const cell = document.createElement("td");
    cell.textContent = text;
    return cell;
  }

  function render() {
    const sorted = Ranking.sortAndRank(rankings);
    podium.replaceChildren();
    body.replaceChildren();
    empty.hidden = sorted.length > 0;

    for (const record of sorted.slice(0, 3)) {
      const card = document.createElement("article");
      card.className = "podium-card";
      const rank = document.createElement("span"); rank.className = "podium-rank"; rank.textContent = `${record.rank}위`;
      const team = document.createElement("strong"); team.className = "podium-team"; team.textContent = record.teamName;
      const score = document.createElement("span"); score.className = "podium-score"; score.textContent = `정답 ${record.correctCount}개 · ${Domain.formatDuration(record.elapsedMs)}`;
      card.append(rank, team, score); podium.append(card);
    }

    for (const record of sorted.slice(3)) {
      const row = document.createElement("tr");
      row.dataset.rank = String(record.rank);
      row.append(makeCell(`${record.rank}위`), makeCell(record.teamName), makeCell(`${record.correctCount}개`), makeCell(Domain.formatDuration(record.elapsedMs)));
      body.append(row);
    }
  }

  async function reload() {
    const snapshot = await persistence.loadSnapshot();
    rankings = snapshot && Array.isArray(snapshot.rankings) ? snapshot.rankings : [];
    render();
  }

  function scheduleReload() {
    global.clearTimeout(reloadTimer);
    reloadTimer = global.setTimeout(reload, 60);
  }

  fullscreen.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else throw new Error("Fullscreen API unavailable");
    } catch (error) {
      storage.textContent = "전체화면 실패 · F11 사용";
    }
  });

  global.addEventListener("beforeunload", () => { persistence.close(); channel.close(); });
  persistence.init().then((result) => {
    rankings = Array.isArray(result.snapshot.rankings) ? result.snapshot.rankings : [];
    render();
    channel.publish({ type: "request-rankings" }, { target: global.opener && !global.opener.closed ? global.opener : undefined, scalar: true });
  });
})(window);
