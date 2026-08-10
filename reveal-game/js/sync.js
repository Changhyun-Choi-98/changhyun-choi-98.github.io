(function initSync(global) {
  "use strict";

  const RevealGame = global.RevealGame = global.RevealGame || {};

  class CrossWindowChannel {
    constructor(name, onMessage) {
      this.name = name;
      this.storageKey = `${name}:event`;
      this.onMessage = onMessage || (() => {});
      this.windows = new Set();
      this.channel = null;
      if (typeof global.BroadcastChannel === "function") {
        try { this.channel = new BroadcastChannel(name); } catch (error) { this.channel = null; }
      }
      this.boundWindowMessage = (event) => this.handleWindowMessage(event);
      this.boundStorage = (event) => this.handleStorage(event);
      if (this.channel) this.channel.addEventListener("message", (event) => this.deliver(event.data, "broadcast"));
      global.addEventListener("message", this.boundWindowMessage);
      global.addEventListener("storage", this.boundStorage);
    }

    isAllowedOrigin(origin) {
      return global.location.protocol === "file:" ? origin === "null" || origin === "" : origin === global.location.origin;
    }

    deliver(message, transport, sourceWindow) {
      if (!message || message.channel !== this.name) return;
      this.onMessage(message.payload, { transport, sourceWindow });
    }

    handleWindowMessage(event) {
      if (!this.isAllowedOrigin(event.origin)) return;
      this.deliver(event.data, "postMessage", event.source);
    }

    handleStorage(event) {
      if (event.key !== this.storageKey || !event.newValue) return;
      try {
        const envelope = JSON.parse(event.newValue);
        this.deliver(envelope, "storage", null);
      } catch (error) {
        return;
      }
    }

    envelope(payload) {
      return { channel: this.name, payload };
    }

    publish(payload, options = {}) {
      const envelope = this.envelope(payload);
      if (this.channel && options.broadcast !== false) {
        try { this.channel.postMessage(envelope); } catch (error) { /* Blob clone fallback continues below. */ }
      }

      const targets = options.target ? [options.target] : Array.from(this.windows);
      for (const target of targets) {
        if (!target || target.closed) {
          this.windows.delete(target);
          continue;
        }
        try { target.postMessage(envelope, global.location.protocol === "file:" ? "*" : global.location.origin); } catch (error) { /* Closed races are harmless. */ }
      }

      if (options.scalar) {
        try {
          global.localStorage.setItem(this.storageKey, JSON.stringify({ ...envelope, nonce: `${Date.now()}-${Math.random()}` }));
        } catch (error) {
          return;
        }
      }
    }

    addWindow(target) {
      if (target) this.windows.add(target);
    }

    hasWindow(target) {
      return Boolean(target) && this.windows.has(target);
    }

    removeClosedWindows() {
      for (const target of this.windows) if (!target || target.closed) this.windows.delete(target);
    }

    close() {
      if (this.channel) this.channel.close();
      global.removeEventListener("message", this.boundWindowMessage);
      global.removeEventListener("storage", this.boundStorage);
      this.windows.clear();
    }
  }

  const Sync = Object.freeze({ CrossWindowChannel });
  RevealGame.Sync = Sync;
})(typeof globalThis !== "undefined" ? globalThis : window);
