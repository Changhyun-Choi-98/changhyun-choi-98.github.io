(function initRanking(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const Domain = RevealGame.Domain || (typeof require === "function" ? require("./domain.js") : null);
  const SCHEMA_VERSION = 1;
  const TEAM_NAME_MAX_LENGTH = 40;
  const teamCollator = new Intl.Collator("ko-KR", { sensitivity: "base", numeric: true });

  function normalizeTeamName(value) {
    return String(value ?? "").trim().replace(/\s+/g, " ").toLocaleLowerCase("ko-KR");
  }

  function validateRankingInput(input, existing = [], editingId = null) {
    const teamName = String(input.teamName ?? "").trim().replace(/\s+/g, " ");
    const normalizedTeamName = normalizeTeamName(teamName);
    const rawCorrectCount = input.correctCount;
    const correctCount = Number(rawCorrectCount);
    const parsedTime = typeof input.elapsedMs === "number"
      ? { ok: Number.isSafeInteger(input.elapsedMs) && input.elapsedMs >= 0, milliseconds: input.elapsedMs }
      : Domain.parseDuration(input.elapsed);
    const errors = {};

    if (!teamName) errors.teamName = "팀명을 입력해 주세요.";
    else if (teamName.length > TEAM_NAME_MAX_LENGTH) errors.teamName = `팀명은 ${TEAM_NAME_MAX_LENGTH}자 이하로 입력해 주세요.`;
    else if (existing.some((record) => record.id !== editingId && normalizeTeamName(record.teamName) === normalizedTeamName)) {
      errors.teamName = "같은 팀명이 이미 있습니다. 기존 팀을 수정해 주세요.";
    }

    if (String(rawCorrectCount ?? "").trim() === "" || !Number.isInteger(correctCount) || correctCount < 0) {
      errors.correctCount = "맞힌 이미지 개수는 0 이상의 정수여야 합니다.";
    }
    if (!parsedTime.ok) errors.elapsed = parsedTime.error || "소요 시간을 확인해 주세요.";

    return {
      ok: Object.keys(errors).length === 0,
      errors,
      value: {
        teamName,
        normalizedTeamName,
        correctCount,
        elapsedMs: parsedTime.ok ? parsedTime.milliseconds : 0,
      },
    };
  }

  function sortAndRank(records) {
    const sorted = Array.from(records).sort((left, right) => {
      if (left.correctCount !== right.correctCount) return right.correctCount - left.correctCount;
      if (left.elapsedMs !== right.elapsedMs) return left.elapsedMs - right.elapsedMs;
      const teamOrder = teamCollator.compare(left.teamName, right.teamName);
      if (teamOrder !== 0) return teamOrder;
      if (left.createdAt !== right.createdAt) return String(left.createdAt).localeCompare(String(right.createdAt));
      return String(left.id).localeCompare(String(right.id));
    });

    let previousScore = null;
    let rank = 0;
    return sorted.map((record, index) => {
      const score = `${record.correctCount}:${record.elapsedMs}`;
      if (score !== previousScore) rank = index + 1;
      previousScore = score;
      return { ...record, rank };
    });
  }

  function createRanking(input, existing = [], now = new Date().toISOString(), id) {
    const validation = validateRankingInput(input, existing);
    if (!validation.ok) return validation;
    const stableId = id || (global.crypto && typeof global.crypto.randomUUID === "function"
      ? global.crypto.randomUUID()
      : `team-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    return {
      ok: true,
      errors: {},
      value: {
        id: stableId,
        ...validation.value,
        createdAt: now,
        updatedAt: now,
      },
    };
  }

  function makeBackup(rankings, settings, exportedAt = new Date().toISOString()) {
    return { schemaVersion: SCHEMA_VERSION, exportedAt, rankings: Array.from(rankings), settings: { ...settings } };
  }

  function validateBackup(candidate) {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
      return { ok: false, errors: ["JSON 최상위 값은 객체여야 합니다."] };
    }
    const errors = [];
    if (candidate.schemaVersion !== SCHEMA_VERSION) errors.push(`지원하는 schemaVersion은 ${SCHEMA_VERSION}입니다.`);
    if (typeof candidate.exportedAt !== "string" || Number.isNaN(Date.parse(candidate.exportedAt))) errors.push("exportedAt은 유효한 날짜 문자열이어야 합니다.");
    if (!Array.isArray(candidate.rankings)) errors.push("rankings 배열이 없습니다.");
    if (!candidate.settings || typeof candidate.settings !== "object" || Array.isArray(candidate.settings)) errors.push("settings 객체가 없습니다.");
    else {
      if (!["mosaic", "circle"].includes(candidate.settings.mode)) errors.push("settings.mode가 올바르지 않습니다.");
      if (!["sequential", "random"].includes(candidate.settings.order)) errors.push("settings.order가 올바르지 않습니다.");
      if (!["slow", "balanced", "fast"].includes(candidate.settings.profile)) errors.push("settings.profile이 올바르지 않습니다.");
      if (!Number.isSafeInteger(candidate.settings.durationMs) || candidate.settings.durationMs < 5000 || candidate.settings.durationMs > 300000) {
        errors.push("settings.durationMs는 5000~300000 사이의 정수여야 합니다.");
      }
    }
    if (errors.length) return { ok: false, errors };

    const validated = [];
    const ids = new Set();
    for (let index = 0; index < candidate.rankings.length; index += 1) {
      const source = candidate.rankings[index];
      if (!source || typeof source !== "object") {
        errors.push(`${index + 1}번째 랭킹 항목이 객체가 아닙니다.`);
        continue;
      }
      const result = validateRankingInput({
        teamName: source.teamName,
        correctCount: source.correctCount,
        elapsedMs: source.elapsedMs,
      }, validated);
      if (!result.ok) {
        errors.push(`${index + 1}번째 랭킹: ${Object.values(result.errors).join(" ")}`);
        continue;
      }
      const id = String(source.id || `import-${index + 1}`);
      if (ids.has(id)) {
        errors.push(`${index + 1}번째 랭킹: 중복된 id가 있습니다.`);
        continue;
      }
      ids.add(id);
      validated.push({
        id,
        ...result.value,
        createdAt: String(source.createdAt || candidate.exportedAt || new Date(0).toISOString()),
        updatedAt: String(source.updatedAt || candidate.exportedAt || new Date(0).toISOString()),
      });
    }
    return errors.length
      ? { ok: false, errors }
      : { ok: true, value: { rankings: validated, settings: { ...candidate.settings } }, errors: [] };
  }

  function protectCsvFormula(value) {
    const text = String(value ?? "");
    return /^[=+\-@]/.test(text) ? `'${text}` : text;
  }

  function csvCell(value) {
    return `"${String(value ?? "").replace(/"/g, '""')}"`;
  }

  function exportCsv(records) {
    const lines = [["순위", "팀명", "맞힌 이미지 개수", "소요 시간"]];
    for (const record of sortAndRank(records)) {
      lines.push([record.rank, protectCsvFormula(record.teamName), record.correctCount, Domain.formatDurationPrecise(record.elapsedMs)]);
    }
    return `\uFEFF${lines.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;
  }

  function parseCsvRows(text) {
    const source = String(text ?? "").replace(/^\uFEFF/, "");
    const rows = [];
    let row = [];
    let cell = "";
    let quoted = false;
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (quoted) {
        if (character === '"' && source[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else if (character === '"') {
          quoted = false;
        } else {
          cell += character;
        }
      } else if (character === '"' && cell === "") {
        quoted = true;
      } else if (character === ",") {
        row.push(cell);
        cell = "";
      } else if (character === "\n") {
        row.push(cell.replace(/\r$/, ""));
        if (row.some((value) => value !== "")) rows.push(row);
        row = [];
        cell = "";
      } else {
        cell += character;
      }
    }
    if (quoted) return { ok: false, error: "닫히지 않은 큰따옴표가 있습니다.", rows: [] };
    row.push(cell.replace(/\r$/, ""));
    if (row.some((value) => value !== "")) rows.push(row);
    return { ok: true, rows };
  }

  function parseCsv(text, now = new Date().toISOString()) {
    const parsed = parseCsvRows(text);
    if (!parsed.ok) return { ok: false, errors: [parsed.error], rankings: [] };
    if (parsed.rows.length < 2) return { ok: false, errors: ["CSV에 헤더와 데이터 행이 필요합니다."], rankings: [] };

    const headers = parsed.rows[0].map((header) => header.trim().toLocaleLowerCase("ko-KR"));
    const findHeader = (aliases) => headers.findIndex((header) => aliases.includes(header));
    const teamIndex = findHeader(["팀명", "team", "teamname", "team name"]);
    const countIndex = findHeader(["맞힌 이미지 개수", "correctcount", "correct count", "score"]);
    const timeIndex = findHeader(["소요 시간", "elapsed", "elapsedtime", "elapsed time", "time"]);
    if ([teamIndex, countIndex, timeIndex].some((index) => index < 0)) {
      return { ok: false, errors: ["CSV 헤더에 팀명, 맞힌 이미지 개수, 소요 시간이 필요합니다."], rankings: [] };
    }

    const rankings = [];
    const errors = [];
    for (let rowIndex = 1; rowIndex < parsed.rows.length; rowIndex += 1) {
      const row = parsed.rows[rowIndex];
      let teamName = row[teamIndex] || "";
      if (/^'[=+\-@]/.test(teamName)) teamName = teamName.slice(1);
      const result = createRanking({
        teamName,
        correctCount: row[countIndex],
        elapsed: row[timeIndex],
      }, rankings, now, `csv-${rowIndex}-${Date.now()}`);
      if (result.ok) rankings.push(result.value);
      else errors.push(`${rowIndex + 1}행: ${Object.values(result.errors).join(" ")}`);
    }
    return { ok: rankings.length > 0, rankings, errors };
  }

  function mergeRankings(existing, incoming) {
    const merged = Array.from(existing);
    const skipped = [];
    for (const record of incoming) {
      if (merged.some((item) => normalizeTeamName(item.teamName) === normalizeTeamName(record.teamName))) {
        skipped.push(record.teamName);
      } else {
        const originalId = String(record.id);
        let id = originalId;
        let suffix = 2;
        while (merged.some((item) => String(item.id) === id)) {
          id = `${originalId}-merged-${suffix}`;
          suffix += 1;
        }
        merged.push(id === originalId ? record : { ...record, id });
      }
    }
    return { rankings: merged, skipped };
  }

  const Ranking = Object.freeze({
    SCHEMA_VERSION,
    TEAM_NAME_MAX_LENGTH,
    normalizeTeamName,
    validateRankingInput,
    sortAndRank,
    createRanking,
    makeBackup,
    validateBackup,
    protectCsvFormula,
    exportCsv,
    parseCsvRows,
    parseCsv,
    mergeRankings,
  });

  RevealGame.Ranking = Ranking;
  if (typeof module !== "undefined" && module.exports) module.exports = Ranking;
})(typeof globalThis !== "undefined" ? globalThis : window);
