(function initPersistence(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const DB_NAME = "image-reveal-game";
  const DB_VERSION = 1;
  const LOCAL_SNAPSHOT_KEY = "image-reveal-game:last-good-snapshot:v1";

  function requestResult(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("IndexedDB 요청이 실패했습니다."));
    });
  }

  function transactionDone(transaction) {
    return new Promise((resolve, reject) => {
      transaction.oncomplete = resolve;
      transaction.onabort = () => reject(transaction.error || new Error("IndexedDB 작업이 중단되었습니다."));
      transaction.onerror = () => reject(transaction.error || new Error("IndexedDB 작업이 실패했습니다."));
    });
  }

  class Persistence {
    constructor(onStatus) {
      this.onStatus = onStatus || (() => {});
      this.db = null;
      this.mode = "memory";
      this.memory = { rankings: [], settings: null, updatedAt: null };
    }

    report(message, warning = false) {
      this.onStatus({ mode: this.mode, message, warning });
    }

    async init() {
      try {
        if (!global.indexedDB) throw new Error("이 브라우저에서 IndexedDB를 사용할 수 없습니다.");
        this.db = await this.openDatabase();
        this.mode = "indexeddb";
        this.report("IndexedDB에 저장합니다.");
      } catch (error) {
        this.db = null;
        this.activateLocalFallback("IndexedDB를 사용할 수 없어 브라우저 임시 저장소로 전환했습니다.");
      }
      const snapshot = await this.loadSnapshot();
      if (snapshot) this.memory = snapshot;
      return { mode: this.mode, snapshot: this.cloneSnapshot(this.memory) };
    }

    openDatabase() {
      return new Promise((resolve, reject) => {
        const request = global.indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = () => {
          const database = request.result;
          if (!database.objectStoreNames.contains("rankings")) database.createObjectStore("rankings", { keyPath: "id" });
          if (!database.objectStoreNames.contains("settings")) database.createObjectStore("settings", { keyPath: "key" });
          if (!database.objectStoreNames.contains("snapshots")) database.createObjectStore("snapshots", { keyPath: "key" });
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("IndexedDB를 열지 못했습니다."));
        request.onblocked = () => reject(new Error("다른 창이 저장소 업데이트를 막고 있습니다."));
      });
    }

    activateLocalFallback(message) {
      try {
        const probe = "image-reveal-game:probe";
        global.localStorage.setItem(probe, "1");
        global.localStorage.removeItem(probe);
        this.mode = "localstorage";
        this.report(message);
      } catch (error) {
        this.mode = "memory";
        this.report("브라우저 저장소를 사용할 수 없어 현재 창의 메모리에만 저장합니다. JSON 백업을 권장합니다.");
      }
    }

    cloneSnapshot(snapshot) {
      return JSON.parse(JSON.stringify(snapshot || { rankings: [], settings: null, updatedAt: null }));
    }

    readLocalSnapshot() {
      try {
        const raw = global.localStorage.getItem(LOCAL_SNAPSHOT_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || !Array.isArray(parsed.rankings)) return null;
        return parsed;
      } catch (error) {
        return null;
      }
    }

    writeLocalSnapshot(snapshot) {
      try {
        global.localStorage.setItem(LOCAL_SNAPSHOT_KEY, JSON.stringify(snapshot));
        return true;
      } catch (error) {
        if (this.mode !== "indexeddb") {
          this.mode = "memory";
          this.report("브라우저 저장 용량 또는 권한 문제로 현재 창의 메모리에만 저장합니다. JSON 백업을 권장합니다.");
        } else this.report("IndexedDB 저장은 유지되지만 로컬 복구 사본을 갱신하지 못했습니다. 저장 용량과 사이트 권한을 확인하고 JSON 백업을 내려받아 주세요.", true);
        return false;
      }
    }

    async loadSnapshot() {
      const localSnapshot = this.readLocalSnapshot();
      let indexedSnapshot = null;
      if (this.mode === "indexeddb" && this.db) {
        try {
          const transaction = this.db.transaction(["rankings", "settings", "snapshots"], "readonly");
          const rankingRequest = transaction.objectStore("rankings").getAll();
          const settingRequest = transaction.objectStore("settings").get("game");
          const snapshotRequest = transaction.objectStore("snapshots").get("last");
          const [rankings, settingRecord, snapshotRecord] = await Promise.all([
            requestResult(rankingRequest),
            requestResult(settingRequest),
            requestResult(snapshotRequest),
          ]);
          await transactionDone(transaction);
          if (rankings.length || settingRecord || snapshotRecord) {
            indexedSnapshot = {
              rankings,
              settings: settingRecord ? settingRecord.value : null,
              updatedAt: snapshotRecord ? snapshotRecord.updatedAt : null,
            };
          }
        } catch (error) {
          this.activateLocalFallback("IndexedDB 읽기에 실패해 마지막 정상 브라우저 저장 사본으로 전환했습니다.");
        }
      }
      if (!indexedSnapshot) return localSnapshot;
      if (!localSnapshot) return indexedSnapshot;
      const indexedTime = Date.parse(indexedSnapshot.updatedAt || "") || 0;
      const localTime = Date.parse(localSnapshot.updatedAt || "") || 0;
      return localTime > indexedTime ? localSnapshot : indexedSnapshot;
    }

    async save(rankings, settings) {
      const snapshot = {
        rankings: Array.from(rankings || []),
        settings: settings ? { ...settings } : null,
        updatedAt: new Date().toISOString(),
      };
      this.memory = this.cloneSnapshot(snapshot);

      if (this.mode === "indexeddb" && this.db) {
        try {
          const transaction = this.db.transaction(["rankings", "settings", "snapshots"], "readwrite");
          const rankingStore = transaction.objectStore("rankings");
          rankingStore.clear();
          for (const ranking of snapshot.rankings) rankingStore.put(ranking);
          transaction.objectStore("settings").put({ key: "game", value: snapshot.settings });
          transaction.objectStore("snapshots").put({ key: "last", updatedAt: snapshot.updatedAt });
          await transactionDone(transaction);
          this.writeLocalSnapshot(snapshot);
          return { ok: true, mode: this.mode };
        } catch (error) {
          this.activateLocalFallback("IndexedDB 저장에 실패해 마지막 정상 브라우저 저장 사본으로 전환했습니다.");
        }
      }

      if (this.mode === "localstorage") {
        const ok = this.writeLocalSnapshot(snapshot);
        return { ok, mode: this.mode };
      }
      return { ok: true, mode: "memory" };
    }

    snapshot() {
      return this.cloneSnapshot(this.memory);
    }

    close() {
      if (this.db) this.db.close();
      this.db = null;
    }
  }

  const PersistenceApi = Object.freeze({ DB_NAME, DB_VERSION, LOCAL_SNAPSHOT_KEY, Persistence });
  RevealGame.Persistence = PersistenceApi;
})(typeof globalThis !== "undefined" ? globalThis : window);
