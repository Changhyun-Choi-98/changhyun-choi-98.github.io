(function initImageLoader(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};
  const Domain = RevealGame.Domain || (typeof require === "function" ? require("./domain.js") : null);
  const MAX_FILE_BYTES = 200 * 1024 * 1024;
  const MAX_SOURCE_PIXELS = 120000000;

  function descriptorFromFile(file, relativePath) {
    return {
      file,
      relativePath: String(relativePath || file.webkitRelativePath || file.name || "image"),
    };
  }

  function prepareFileEntries(files) {
    const accepted = [];
    let ignoredCount = 0;
    for (const item of Array.from(files || [])) {
      const descriptor = item && item.file
        ? descriptorFromFile(item.file, item.relativePath)
        : descriptorFromFile(item, item.webkitRelativePath || item.name);
      if (Domain.isSupportedImageName(descriptor.file && descriptor.file.name)) accepted.push(descriptor);
      else ignoredCount += 1;
    }
    accepted.sort((left, right) => Domain.naturalCompare(left.relativePath, right.relativePath));
    return { accepted, ignoredCount };
  }

  async function collectDirectoryHandle(handle, prefix = "") {
    const descriptors = [];
    for await (const [name, entry] of handle.entries()) {
      const relativePath = prefix ? `${prefix}/${name}` : name;
      if (entry.kind === "directory") {
        descriptors.push(...await collectDirectoryHandle(entry, relativePath));
      } else if (entry.kind === "file") {
        descriptors.push(descriptorFromFile(await entry.getFile(), relativePath));
      }
    }
    return descriptors;
  }

  function readDirectoryEntries(reader) {
    return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
  }

  async function collectWebkitEntry(entry, prefix = "") {
    if (!entry) return [];
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      return [descriptorFromFile(file, relativePath)];
    }
    if (!entry.isDirectory) return [];
    const reader = entry.createReader();
    const children = [];
    while (true) {
      const batch = await readDirectoryEntries(reader);
      if (!batch.length) break;
      children.push(...batch);
    }
    const nested = [];
    for (const child of children) nested.push(...await collectWebkitEntry(child, relativePath));
    return nested;
  }

  async function collectDataTransfer(dataTransfer) {
    const descriptors = [];
    const items = Array.from((dataTransfer && dataTransfer.items) || []);
    if (items.length) {
      for (const item of items) {
        const entry = typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null;
        if (entry) descriptors.push(...await collectWebkitEntry(entry));
        else {
          const file = item.getAsFile && item.getAsFile();
          if (file) descriptors.push(descriptorFromFile(file, file.name));
        }
      }
      return descriptors;
    }
    return Array.from((dataTransfer && dataTransfer.files) || []).map((file) => descriptorFromFile(file, file.name));
  }

  class ImagePool {
    constructor(onPreloadError) {
      this.entries = [];
      this.decoded = new Map();
      this.failures = new Map();
      this.pending = new Map();
      this.inflight = new Map();
      this.generation = 0;
      this.onPreloadError = onPreloadError || (() => {});
    }

    setEntries(entries) {
      this.dispose();
      this.entries = Array.from(entries || []);
    }

    async decode(index) {
      if (this.decoded.has(index)) return this.decoded.get(index);
      if (this.failures.has(index)) throw this.failures.get(index);
      if (this.pending.has(index)) return this.pending.get(index);
      const task = this.decodeFile(index);
      this.pending.set(index, task);
      const cleanup = () => { if (this.pending.get(index) === task) this.pending.delete(index); };
      task.then(cleanup, cleanup);
      return task;
    }

    async decodeFile(index) {
      const descriptor = this.entries[index];
      if (!descriptor) throw new Error("이미지 항목을 찾을 수 없습니다.");
      if (descriptor.file.size > MAX_FILE_BYTES) {
        const error = new Error("파일 크기가 너무 커서 이 이미지를 건너뜁니다.");
        this.failures.set(index, error);
        throw error;
      }

      const url = URL.createObjectURL(descriptor.file);
      const image = new Image();
      const generation = this.generation;
      image.decoding = "async";
      try {
        await new Promise((resolve, reject) => {
          const cancel = () => {
            const error = new Error("이전 이미지 불러오기를 취소했습니다.");
            error.name = "AbortError";
            image.onload = null;
            image.onerror = null;
            image.removeAttribute("src");
            reject(error);
          };
          this.inflight.set(index, { url, image, cancel });
          image.onload = resolve;
          image.onerror = () => reject(new Error("이미지를 해석할 수 없어 건너뜁니다."));
          image.src = url;
        });
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        if (generation !== this.generation) {
          const cancelled = new Error("이전 이미지 불러오기를 취소했습니다.");
          cancelled.name = "AbortError";
          throw cancelled;
        }
        if (!width || !height || width * height > MAX_SOURCE_PIXELS) {
          throw new Error("이미지 해상도가 너무 크거나 올바르지 않아 건너뜁니다.");
        }
        const decoded = { descriptor, file: descriptor.file, image, url, width, height };
        this.decoded.set(index, decoded);
        return decoded;
      } catch (error) {
        URL.revokeObjectURL(url);
        image.removeAttribute("src");
        const safeError = error instanceof Error ? error : new Error("이미지를 불러오지 못했습니다.");
        if (generation === this.generation && safeError.name !== "AbortError") this.failures.set(index, safeError);
        throw safeError;
      } finally {
        const inflight = this.inflight.get(index);
        if (inflight && inflight.url === url) this.inflight.delete(index);
      }
    }

    async activate(index) {
      this.releaseExcept(new Set([index, index + 1]));
      const current = await this.decode(index);
      if (index + 1 < this.entries.length) {
        this.decode(index + 1).catch((error) => { if (error.name !== "AbortError") this.onPreloadError(index + 1, error); });
      }
      return current;
    }

    releaseExcept(keep) {
      for (const [index, decoded] of this.decoded.entries()) {
        if (!keep.has(index)) {
          URL.revokeObjectURL(decoded.url);
          decoded.image.removeAttribute("src");
          this.decoded.delete(index);
        }
      }
    }

    dispose() {
      this.generation += 1;
      for (const inflight of this.inflight.values()) {
        URL.revokeObjectURL(inflight.url);
        inflight.cancel();
      }
      this.inflight.clear();
      this.pending.clear();
      for (const decoded of this.decoded.values()) {
        URL.revokeObjectURL(decoded.url);
        decoded.image.removeAttribute("src");
      }
      this.decoded.clear();
      this.failures.clear();
      this.entries = [];
    }
  }

  const ImageLoader = Object.freeze({
    MAX_FILE_BYTES,
    MAX_SOURCE_PIXELS,
    descriptorFromFile,
    prepareFileEntries,
    collectDirectoryHandle,
    collectWebkitEntry,
    collectDataTransfer,
    ImagePool,
  });

  RevealGame.ImageLoader = ImageLoader;
  if (typeof module !== "undefined" && module.exports) module.exports = ImageLoader;
})(typeof globalThis !== "undefined" ? globalThis : window);
