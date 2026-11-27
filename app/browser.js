// The browser inside the app (the computer panel's Browser tab): a real Mac browser the assistant may use, with
// the owner's OK, for a site that turns the sandbox's browser away (coderunner/browser/app.py, switch.py).
//
// Each tab is a WebContentsView on its own persistent session (persist:coderunner-browser): apart from the app's
// page, from the owner's Chrome, and from each other's windows. WebContentsView rather than <webview>: a <webview>
// would need webviewTag on the daemon's page (which paired devices load too) and lives in that page, while the
// views live here in the main process, where webContents.debugger is, and the page never reaches into a site.
// The page only draws the tabs, the address field and the buttons, and says where the view goes.
//
// The Mac's real platform, GPU, fonts and timezone are left as-is; only Electron's custom app/runtime tokens
// ("CodeRunner/..." and "Electron/...") are stripped from this session's User-Agent and Sec-CH-UA Client Hints are
// aligned to standard desktop Chrome matching the real underlying Chromium version so Google Sign-In and bot checks
// do not reject the embedded browser.
//
// The assistant drives it as it drives the owner's Chrome through the extension (extension/background.js): the
// daemon's broker speaks the DevTools protocol over a unix socket in the owner's assistant folder (0600), the app
// says hello with a token only the owner can read, and the browser-level Target methods are answered here with
// these tabs alone. A tab's commands go to its webContents.debugger, for the protocol domains the broker needs.
"use strict";

const {BrowserWindow, WebContentsView, clipboard, session, shell} = require("electron");
const fs = require("fs");
const net = require("net");
const path = require("path");

const PARTITION = "persist:coderunner-browser";
const ORIGIN = "coderunner-app";
// Protocol domains a tab session may use: what the broker needs, as for the extension. Nothing that reads the
// session's cookies or storage, or reaches the app itself; of Network only the request events (readiness).
const DOMAINS = new Set(["Accessibility", "DOM", "Emulation", "Fetch", "Input", "Page", "Runtime"]);
const ALSO = new Set(["Network.enable", "Network.disable"]);
const BLOCKED = new Set(["Page.setDownloadBehavior"]);
const AUTO_ATTACH = {autoAttach: true, waitForDebuggerOnStart: true, flatten: true, filter: [{type: "iframe"}]};
const RETRY_MS = 3000;
const MAX_FRAME = 256 * 1024 * 1024;
const WEB = /^(https?:|about:blank$)/i;
const IN_FRAME = /^(https?|about|data|blob|javascript):/i;     // what a frame inside a page may show
const SEARCH = "https://duckduckgo.com/?q=";
// Keep cookies: "always" unless the owner chooses "until quit". A site that blocked the sandbox's browser lets the
// Mac's through on the strength of its cookies (Cloudflare's cf_clearance, DataDome's datadome), and the sign-ins
// the owner chose to bring over are cookies too: cleared at every quit, each would be asked for again.
const KEEP_DEFAULT = "always";
const GREASE_BRANDS = [
  ["Not(A:Brand", "99"],
  ["Not/A)Brand", "8"],
  ["Not;A=Brand", "24"],
];

// Strip Electron's app/runtime tokens ("CodeRunner/x.y.z", "Electron/x.y.z") and normalize Chrome/<major>.0.0.0
// with matching Sec-CH-UA Client Hints derived from the actual running Chromium version.
function desktopChromeIdentity(rawUA, chromeVersion = (process.versions && process.versions.chrome) || "") {
  const fromUA = /Chrome\/(\d+)(?:\.(\d+\.\d+\.\d+))?/.exec(String(rawUA || ""));
  const full = /^\d+\.\d+\.\d+\.\d+$/.test(chromeVersion)
    ? chromeVersion
    : (fromUA ? (fromUA[2] ? `${fromUA[1]}.${fromUA[2]}` : `${fromUA[1]}.0.0.0`) : "136.0.0.0");
  const major = full.split(".")[0];
  let ua = String(rawUA || "")
    .replace(/\s+CodeRunner\/[^\s]+/gi, "")
    .replace(/\s+Electron\/[^\s]+/gi, "")
    .replace(/Chrome\/\d+\.\d+\.\d+\.\d+/g, `Chrome/${major}.0.0.0`)
    .replace(/\s{2,}/g, " ")
    .trim();
  if (!ua || !/Chrome\/\d+/.test(ua)) {
    ua = `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${major}.0.0.0 Safari/537.36`;
  }
  const [greaseName, greaseVer] = GREASE_BRANDS[(parseInt(major, 10) || 0) % GREASE_BRANDS.length];
  const secChUa = `"Chromium";v="${major}", "Google Chrome";v="${major}", "${greaseName}";v="${greaseVer}"`;
  const secChUaFull = `"Chromium";v="${full}", "Google Chrome";v="${full}", "${greaseName}";v="${greaseVer}.0.0.0"`;
  const cdpParams = {
    userAgent: ua,
    acceptLanguage: "en-US,en",
    platform: "MacIntel",
    userAgentMetadata: {
      brands: [
        {brand: "Chromium", version: major},
        {brand: "Google Chrome", version: major},
        {brand: greaseName, version: greaseVer},
      ],
      fullVersionList: [
        {brand: "Chromium", version: full},
        {brand: "Google Chrome", version: full},
        {brand: greaseName, version: `${greaseVer}.0.0.0`},
      ],
      fullVersion: full,
      platform: "macOS",
      platformVersion: "15.4.0",
      architecture: "arm",
      model: "",
      mobile: false,
      bitness: "64",
      wow64: false,
    },
  };
  return {userAgent: ua, full, major, secChUa, secChUaFull, cdpParams};
}

class AppBrowser {
  // assistantDir: the daemon's assistant folder (its token and socket address are there); daemonOrigin: the
  // daemon's own address, which pages in this browser may never reach.
  // keep / setKeep: the owner's "Keep cookies" choice, "always" or "quit", kept by the shell in its settings.
  // openExternal and writeClipboard are the system's (a test passes its own, so nothing reaches the owner's).
  constructor({assistantDir, daemonOrigin = "", keep = () => KEEP_DEFAULT, setKeep = () => {},
               openExternal = (url) => shell.openExternal(url), writeClipboard = (text) => clipboard.writeText(text)}) {
    this.assistantDir = assistantDir;
    this.daemonOrigin = daemonOrigin;
    Object.assign(this, {keep, setKeep, openExternal, writeClipboard});
    this.tabs = new Map();          // tab id -> {id, view, targetId, opener, session, assistant}
    this.order = [];
    this.active = null;
    this.host = null;               // the window the views are in: the app's main window, or a hidden one
    this.holder = null;
    this.bounds = {x: 0, y: 0, width: 1280, height: 800};
    this.shown = false;             // the page shows the Browser tab
    this.onState = () => {};
    this.onFront = () => {};        // the assistant works in a tab: the page shows the Browser tab
    this.socket = null;
    this.connected = false;
    this.discover = false;
    this.fetchPatterns = null;
    this.denyDownloads = false;
    this.children = new Map();      // child session (a frame in its own process) -> tab id
    this.stopped = false;
    this.session = session.fromPartition(PARTITION);
    this.guardSession();
    // "Until quit": what the last run left (a quit that did not finish clearing, a crash) goes before any page loads.
    if (this.keep() === "quit") this.clearCookies();
  }

  clearCookies() {
    return this.session.clearStorageData({storages: ["cookies"]}).catch(() => {});
  }

  // Clear browsing data: cookies, site storage and the cache of this browser alone (never the app's own session,
  // never the owner's Chrome), and each tab's history.
  async clearData() {
    await this.session.clearStorageData();
    await this.session.clearCache();
    await this.session.clearAuthCache();
    for (const tab of this.tabs.values()) {
      const contents = tab.view.webContents;
      if (!contents.isDestroyed()) contents.navigationHistory.clear();
    }
    this.changed();
    return {cleared: true};
  }

  // A still of the active page, shown in its place while the page's own menu or a form covers the browser (a view
  // is drawn over the page, so nothing of the page could be seen on top of it).
  async still() {
    const contents = this.tabs.get(this.active)?.view.webContents;
    if (!contents || contents.isDestroyed()) return "";
    try {
      const shot = await contents.debugger.sendCommand("Page.captureScreenshot", {format: "jpeg", quality: 80});
      return `data:image/jpeg;base64,${shot.data}`;
    } catch {
      return "";
    }
  }

  // ---- the session: what any page in it may do --------------------------------------------------------

  guardSession() {
    const ses = this.session;
    // Strip "CodeRunner/..." and "Electron/..." from this browser session's User-Agent and align Sec-CH-UA Client
    // Hints with the real underlying Chromium version so Google Sign-In and WAFs see standard desktop Chrome.
    // Note: Electron's setUserAgent(ua, acceptLanguages) takes bare language codes ("en-US,en") and adds ;q= itself.
    const identity = desktopChromeIdentity(typeof ses.getUserAgent === "function" ? ses.getUserAgent() : "");
    this.chromeIdentity = identity;
    if (typeof ses.setUserAgent === "function") ses.setUserAgent(identity.userAgent, "en-US,en");
    if (ses.webRequest && typeof ses.webRequest.onBeforeSendHeaders === "function") {
      ses.webRequest.onBeforeSendHeaders({urls: ["http://*/*", "https://*/*"]}, (details, callback) => {
        const headers = {...details.requestHeaders};
        const isHttps = String(details.url || "").startsWith("https://");
        let hasChUa = false, hasChMob = false, hasChPlat = false;
        for (const key of Object.keys(headers)) {
          const low = key.toLowerCase();
          if (low === "sec-ch-ua") { headers[key] = identity.secChUa; hasChUa = true; }
          else if (low === "sec-ch-ua-mobile") { headers[key] = "?0"; hasChMob = true; }
          else if (low === "sec-ch-ua-platform") { headers[key] = '"macOS"'; hasChPlat = true; }
          else if (low === "sec-ch-ua-full-version-list") headers[key] = identity.secChUaFull;
        }
        if (isHttps) {
          if (!hasChUa) headers["Sec-CH-UA"] = identity.secChUa;
          if (!hasChMob) headers["Sec-CH-UA-Mobile"] = "?0";
          if (!hasChPlat) headers["Sec-CH-UA-Platform"] = '"macOS"';
        }
        callback({requestHeaders: headers});
      });
    }
    // Fullscreen video and copy buttons; every other permission (camera, location, notifications...) is refused,
    // as the broker refuses them in the assistant's other browsers.
    const fine = new Set(["fullscreen", "clipboard-sanitized-write"]);
    ses.setPermissionRequestHandler((contents, permission, callback) => callback(fine.has(permission)));
    ses.setPermissionCheckHandler((contents, permission) => fine.has(permission));
    // The daemon is never reached from a page here (its admin token rides only on the app's own session), and
    // nor are local files.
    const banned = ["file://*/*"];
    if (this.daemonOrigin) banned.push(`${this.daemonOrigin}/*`, `${this.daemonOrigin.replace(/^http/, "ws")}/*`);
    ses.webRequest.onBeforeRequest({urls: banned}, (details, callback) => callback({cancel: true}));
    ses.on("will-download", (event, item, contents) => {
      const tab = this.tabOf(contents);
      if (tab && tab.session && this.denyDownloads) item.cancel();      // the assistant's steps download nothing
    });
  }

  // ---- tabs ----------------------------------------------------------------------------------------------

  // Where the views are: in the app's window while the page shows the Browser tab, otherwise in a window nobody
  // sees (also while the app's window is closed and it stays in the menu bar). A view hidden in the app's window,
  // or moved outside it, draws nothing, and the broker's screenshot of its page waited forever; in a window that
  // was never shown, Chromium keeps drawing it.
  container() {
    return this.shown && this.host && !this.host.isDestroyed() ? this.host : this.hidden();
  }

  hidden() {
    if (!this.holder || this.holder.isDestroyed()) {
      this.holder = new BrowserWindow({show: false, width: this.bounds.width, height: this.bounds.height, skipTaskbar: true,
                                       paintWhenInitiallyHidden: true, webPreferences: {sandbox: true, contextIsolation: true}});
    }
    return this.holder;
  }

  // The app's main window, once it is there (null when it closes, and the views go to the hidden one first).
  setHost(win) {
    this.host = win && !win.isDestroyed() ? win : null;
    if (this.host) this.host.once("close", () => { if (this.host === win) this.setHost(null); });
    this.layout();
  }

  // Where the page shows the browser (in the window's coordinates), and whether it shows it.
  place(bounds, shown) {
    if (bounds && [bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite) && bounds.width > 0 && bounds.height > 0)
      this.bounds = {x: Math.round(bounds.x), y: Math.round(bounds.y), width: Math.round(bounds.width), height: Math.round(bounds.height)};
    this.shown = Boolean(shown);
    this.layout();
  }

  // Every tab keeps the size the page gives it, so a page the assistant reads in the background is laid out as it
  // will be seen; the active one is drawn over the others.
  layout() {
    const into = this.container(), {width, height} = this.bounds;
    if (into === this.holder) {
      const [w, h] = into.getContentSize();
      if (w !== width || h !== height) into.setContentSize(width, height);
    }
    const box = into === this.holder ? {x: 0, y: 0, width, height} : this.bounds;
    for (const tab of this.tabs.values()) this.put(tab, into, box);
    const active = this.tabs.get(this.active);
    if (active && !active.view.webContents.isDestroyed() && into.contentView.children.at(-1) !== active.view)
      into.contentView.addChildView(active.view);
  }

  put(tab, into, box) {
    if (tab.view.webContents.isDestroyed()) return;
    if (tab.parent !== into) {
      if (tab.parent && !tab.parent.isDestroyed()) tab.parent.contentView.removeChildView(tab.view);
      into.contentView.addChildView(tab.view);
      tab.parent = into;
    }
    tab.view.setBounds(box);
  }

  async createTab(url = "about:blank", {opener = null, background = false} = {}) {
    const view = new WebContentsView({webPreferences: {partition: PARTITION, sandbox: true, contextIsolation: true,
                                                       nodeIntegration: false, backgroundThrottling: false, spellcheck: true}});
    const contents = view.webContents;
    const tab = {id: contents.id, view, targetId: "", opener, session: null, assistant: false};
    this.tabs.set(tab.id, tab);
    this.order.push(tab.id);
    this.layout();
    this.watch(tab);
    try {
      contents.debugger.attach("1.3");
      tab.targetId = (await contents.debugger.sendCommand("Target.getTargetInfo")).targetInfo.targetId;
    } catch (error) {
      this.closeTab(tab.id);
      throw error;
    }
    if (!background || this.active === null) this.active = tab.id;
    this.layout();
    this.announce(tab);
    // Even a blank tab gets its page loaded: until a view has loaded something it has no page process, and the
    // protocol's page commands (Page.enable, Emulation.setUserAgentOverride) wait for one.
    contents.loadURL(url || "about:blank").catch(() => {});
    if (this.chromeIdentity) {
      contents.debugger.sendCommand("Emulation.setUserAgentOverride", this.chromeIdentity.cdpParams).catch(() => {});
    }
    this.changed();
    return tab;
  }

  watch(tab) {
    const contents = tab.view.webContents;
    for (const event of ["page-title-updated", "did-navigate", "did-navigate-in-page", "did-start-loading", "did-stop-loading"])
      contents.on(event, () => { this.announce(tab, "Target.targetInfoChanged"); this.changed(); });
    // Only web pages: no file:, no browser-internal pages, no custom schemes that would open other apps.
    contents.on("will-navigate", (event, url) => { if (!WEB.test(url)) event.preventDefault(); });
    contents.on("will-frame-navigate", (details) => { if (!IN_FRAME.test(details.url)) details.preventDefault(); });
    // A popup opens as a tab of this browser that keeps its opener (sign-in windows need it).
    contents.setWindowOpenHandler(({url}) => {
      if (!WEB.test(url)) return {action: "deny"};
      return {action: "allow", outlivesOpener: false, createWindow: (options) => {
        const popup = this.createTabNow(options, tab.targetId);
        return popup.view.webContents;
      }};
    });
    contents.debugger.on("message", (event, method, params, sessionId) => this.onEvent(tab, method, params, sessionId));
    contents.debugger.on("detach", (event, reason) => {
      if (tab.session) this.emit("Target.detachedFromTarget", {sessionId: tab.session, targetId: tab.targetId});
      tab.session = null;
      // Never re-attach synchronously (or at all while closing/releasing/target-closed): Chromium can emit
      // "detach" during teardown before isDestroyed() flips, and synchronous re-attach spins the main thread at 100%.
      if (tab.closing || tab.releasing || !this.tabs.has(tab.id) || contents.isDestroyed()) return;
      if (reason === "target closed" || reason === "canceled_by_user" || reason === "replaced_with_devtools") return;
      const now = Date.now();
      tab.reattachTimes = (tab.reattachTimes || []).filter((t) => now - t < 5000);
      if (tab.reattachTimes.length >= 3) return;
      tab.reattachTimes.push(now);
      clearTimeout(tab.reattachTimer);
      tab.reattachTimer = setTimeout(() => {
        if (tab.closing || !this.tabs.has(tab.id) || contents.isDestroyed() || contents.debugger.isAttached()) return;
        try { contents.debugger.attach("1.3"); } catch { /* gone */ }
      }, 150);
    });
    contents.on("destroyed", () => this.forget(tab.id));
  }

  // createWindow must hand back a webContents at once: the popup's protocol id follows a moment later.
  createTabNow(options, opener) {
    const preferences = {...(options.webPreferences || {}), partition: PARTITION, sandbox: true, contextIsolation: true,
                         nodeIntegration: false, backgroundThrottling: false};
    const view = new WebContentsView({...options, webPreferences: preferences});
    const tab = {id: view.webContents.id, view, targetId: "", opener, session: null, assistant: false};
    this.tabs.set(tab.id, tab);
    this.order.push(tab.id);
    this.active = tab.id;
    this.layout();
    this.watch(tab);
    tab.ready = (async () => {
      try {
        view.webContents.debugger.attach("1.3");
        tab.targetId = (await view.webContents.debugger.sendCommand("Target.getTargetInfo")).targetInfo.targetId;
        if (this.chromeIdentity) {
          view.webContents.debugger.sendCommand("Emulation.setUserAgentOverride", this.chromeIdentity.cdpParams).catch(() => {});
        }
        this.announce(tab);
      } catch { /* closed already */ }
    })();
    this.changed();
    return tab;
  }

  closeTab(id) {
    const tab = this.tabs.get(id);
    if (!tab) return;
    tab.closing = true;
    clearTimeout(tab.reattachTimer);
    const contents = tab.view.webContents;
    if (!contents.isDestroyed()) {
      try { contents.debugger.removeAllListeners(); } catch { /* gone */ }
      try { if (contents.debugger.isAttached()) contents.debugger.detach(); } catch { /* gone */ }
    }
    this.forget(id);
    try { if (tab.parent && !tab.parent.isDestroyed()) tab.parent.contentView.removeChildView(tab.view); } catch { /* gone */ }
    if (!contents.isDestroyed()) contents.close();
  }

  forget(id) {
    const tab = this.tabs.get(id);
    if (!tab) return;
    tab.closing = true;
    clearTimeout(tab.reattachTimer);
    this.tabs.delete(id);
    this.order = this.order.filter((other) => other !== id);
    for (const [child, owner] of this.children) if (owner === id) this.children.delete(child);
    if (tab.session) this.emit("Target.detachedFromTarget", {sessionId: tab.session, targetId: tab.targetId});
    if (tab.targetId && this.discover) this.emit("Target.targetDestroyed", {targetId: tab.targetId});
    if (this.active === id) this.active = this.order[this.order.length - 1] ?? null;
    this.layout();
    this.changed();
  }

  select(id) {
    if (!this.tabs.has(id)) return;
    this.active = id;
    this.layout();
    this.changed();
  }

  tabOf(contents) {
    for (const tab of this.tabs.values()) if (tab.view.webContents === contents) return tab;
    return null;
  }

  tabByTarget(targetId) {
    for (const tab of this.tabs.values()) if (tab.targetId === targetId) return tab;
    throw new Error("No target with given id found");
  }

  // What the page draws: the tabs, the active one's address and whether it can go back or forward.
  state() {
    const tabs = this.order.map((id) => this.tabs.get(id)).filter(Boolean).map((tab) => {
      const contents = tab.view.webContents;
      return {id: tab.id, title: contents.isDestroyed() ? "" : contents.getTitle(), url: contents.isDestroyed() ? "" : contents.getURL(),
              loading: !contents.isDestroyed() && contents.isLoading(), active: tab.id === this.active,
              assistant: Boolean(tab.session)};
    });
    const active = this.tabs.get(this.active)?.view.webContents;
    const history = active && !active.isDestroyed() ? active.navigationHistory : null;
    return {tabs, canBack: Boolean(history?.canGoBack()), canForward: Boolean(history?.canGoForward()),
            connected: this.connected, keep: this.keep() === "quit" ? "quit" : "always"};
  }

  changed() {
    clearTimeout(this.stateTimer);
    this.stateTimer = setTimeout(() => { try { this.onState(this.state()); } catch { /* no page */ } }, 30);
  }

  // The owner's buttons and address field.
  async act(request) {
    const action = String(request?.action || "");
    const tab = this.tabs.get(Number(request?.id)) || this.tabs.get(this.active);
    const contents = tab?.view.webContents;
    if (action === "new") return void await this.createTab(addressOf(request?.url) || "about:blank");
    // The "..." menu beside the address.
    if (action === "keep") { this.setKeep(request?.url === "quit" ? "quit" : "always"); this.changed(); return this.keep(); }
    if (action === "clear") return this.clearData();
    if (action === "still") return this.still();
    if (action === "open-outside" || action === "copy") {
      const url = contents && !contents.isDestroyed() ? contents.getURL() : "";
      if (!/^https?:\/\//i.test(url)) return false;          // only a web page's address leaves the browser
      if (action === "copy") this.writeClipboard(url); else await this.openExternal(url);
      return true;
    }
    if (action === "select") return this.select(Number(request.id));
    if (action === "close") return tab && this.closeTab(tab.id);
    if (!contents || contents.isDestroyed()) {
      if (action === "go" && addressOf(request?.url)) await this.createTab(addressOf(request.url));
      return;
    }
    if (action === "back" && contents.navigationHistory.canGoBack()) contents.navigationHistory.goBack();
    else if (action === "forward" && contents.navigationHistory.canGoForward()) contents.navigationHistory.goForward();
    else if (action === "reload") contents.reload();
    else if (action === "stop") contents.stop();
    else if (action === "go" && addressOf(request?.url)) contents.loadURL(addressOf(request.url)).catch(() => {});
  }

  // ---- the daemon's connection ---------------------------------------------------------------------------

  read(name) {
    try { return fs.readFileSync(path.join(this.assistantDir, "app-browser", name), "utf8").trim(); } catch { return ""; }
  }

  connect() {
    if (this.stopped || this.socket) return;
    const where = this.read("socket"), token = this.read("token");
    if (!where || !token) { this.retry(); return; }
    const socket = net.createConnection(where);
    this.socket = socket;
    let buffer = Buffer.alloc(0);
    socket.on("connect", () => this.send({type: "hello", origin: ORIGIN, token, pid: process.pid}));
    socket.on("data", (chunk) => {
      buffer = Buffer.concat([buffer, chunk]);
      while (buffer.length >= 4) {
        const size = buffer.readUInt32BE(0);
        if (size > MAX_FRAME) { socket.destroy(); return; }
        if (buffer.length < 4 + size) break;
        let message = null;
        try { message = JSON.parse(buffer.subarray(4, 4 + size).toString("utf8")); } catch { /* not a message */ }
        buffer = buffer.subarray(4 + size);
        if (message) this.onMessage(message);
      }
    });
    socket.on("error", () => {});
    socket.on("close", () => {
      if (this.socket === socket) this.socket = null;
      this.setConnected(false);
      this.retry();
    });
  }

  retry() {
    if (this.stopped) return;
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => this.connect(), RETRY_MS);
  }

  stop() {
    this.stopped = true;
    clearTimeout(this.retryTimer);
    if (this.socket) this.socket.destroy();
  }

  send(message) {
    if (!this.socket || this.socket.destroyed) return;
    const data = Buffer.from(JSON.stringify(message), "utf8");
    const header = Buffer.alloc(4);
    header.writeUInt32BE(data.length, 0);
    this.socket.write(Buffer.concat([header, data]));
  }

  emit(method, params, sessionId) {
    if (!this.connected) return;
    const message = {method, params};
    if (sessionId) message.sessionId = sessionId;
    this.send(message);
  }

  setConnected(on) {
    if (this.connected === on) return;
    this.connected = on;
    if (!on) {
      this.discover = false;
      this.fetchPatterns = null;
      this.denyDownloads = false;
      this.releaseAll();
    }
    this.changed();
  }

  onMessage(message) {
    if (message.type === "welcome") this.setConnected(true);
    else if (message.type === "busy") this.socket?.destroy();         // another copy of the app has it
    else if (message.type === "cookies") this.setCookies(message);
    else if (message.type === "cookie_domains") this.listCookieDomains(message);
    else if (message.type === "clear_site_cookies") this.clearSiteCookies(message);
    else if (message.id !== undefined && message.method) this.run(message);
  }

  async run({id, method, params = {}, sessionId}) {
    try {
      const result = sessionId ? await this.tabCommand(sessionId, method, params) : await this.browserCommand(method, params);
      this.send({id, result: result || {}});
    } catch (error) {
      this.send({id, error: {message: String(error?.message || error)}});
    }
  }

  // The owner's sign-ins they chose to bring over (coderunner/browser/cookies.py). Counted, never logged.
  async setCookies({ref, cookies}) {
    let set = 0, failed = 0;
    for (const cookie of Array.isArray(cookies) ? cookies : []) {
      try { await this.session.cookies.set(cookie); set += 1; } catch { failed += 1; }
    }
    this.send({type: "cookies-set", ref, set, failed});
  }

  // Return only domain names and cookie counts (never cookie names or values) for the Settings session vault.
  async listCookieDomains({ref}) {
    const counts = new Map();
    try {
      const all = await this.session.cookies.get({});
      for (const c of all || []) {
        const d = String(c.domain || "").replace(/^\./, "").replace(/^www\./, "").toLowerCase();
        if (!d) continue;
        counts.set(d, (counts.get(d) || 0) + 1);
      }
    } catch { /* empty */ }
    const domains = [...counts.entries()].map(([domain, count]) => ({domain, count})).sort((a, b) => b.count - a.count);
    this.send({type: "cookie-domains", ref, domains});
  }

  // Sign out of a single site in CodeRunner's built-in Mac browser (leaves all other sites untouched).
  async clearSiteCookies({ref, domain}) {
    const target = String(domain || "").replace(/^\./, "").replace(/^www\./, "").toLowerCase();
    let removed = 0;
    if (target) {
      try {
        const all = await this.session.cookies.get({});
        for (const c of all || []) {
          const d = String(c.domain || "").replace(/^\./, "").replace(/^www\./, "").toLowerCase();
          if (d === target || d.endsWith("." + target)) {
            const url = `${c.secure ? "https" : "http"}://${String(c.domain || "").replace(/^\./, "")}${c.path || "/"}`;
            try { await this.session.cookies.remove(url, c.name); removed += 1; } catch { /* skip */ }
          }
        }
      } catch { /* empty */ }
    }
    this.send({type: "site-cookies-cleared", ref, domain: target, removed});
  }

  // ---- the protocol ----------------------------------------------------------------------------------------

  async info(tab) {
    if (tab.ready) await tab.ready;
    const contents = tab.view.webContents;
    const result = {targetId: tab.targetId, type: "page", title: contents.getTitle() || "", url: contents.getURL() || "about:blank",
                    attached: Boolean(tab.session), canAccessOpener: false, browserContextId: "coderunner-app"};
    if (tab.opener) result.openerId = tab.opener;
    return result;
  }

  async announce(tab, method = "Target.targetCreated") {
    if (this.discover && this.connected && tab.targetId && !tab.view.webContents.isDestroyed())
      this.emit(method, {targetInfo: await this.info(tab)});
  }

  async attach(tab) {
    if (tab.session) return tab.session;
    const debuggee = tab.view.webContents.debugger;
    if (!debuggee.isAttached()) debuggee.attach("1.3");
    tab.session = `tab:${tab.id}`;
    try {
      if (this.fetchPatterns) await debuggee.sendCommand("Fetch.enable", {patterns: this.fetchPatterns});
      await debuggee.sendCommand("Target.setAutoAttach", AUTO_ATTACH);
    } catch (error) {
      await this.release(tab);
      throw error;
    }
    this.changed();
    return tab.session;
  }

  // Let go of a tab: the debugger is attached afresh, so nothing the assistant set (request interception, an
  // emulated size) stays on the owner's tab.
  async release(tab) {
    if (!tab.session) return;
    const sessionId = tab.session;
    tab.session = null;
    for (const [child, owner] of this.children) if (owner === tab.id) this.children.delete(child);
    const contents = tab.view.webContents;
    if (!contents.isDestroyed() && !tab.closing && this.tabs.has(tab.id)) {
      tab.releasing = true;
      try {
        try { if (contents.debugger.isAttached()) contents.debugger.detach(); } catch { /* not attached */ }
        if (!contents.isDestroyed() && !tab.closing && this.tabs.has(tab.id)) {
          try {
            contents.debugger.attach("1.3");
            if (this.chromeIdentity) {
              contents.debugger.sendCommand("Emulation.setUserAgentOverride", this.chromeIdentity.cdpParams).catch(() => {});
            }
          } catch { /* gone */ }
        }
      } finally {
        tab.releasing = false;
      }
    }
    this.emit("Target.detachedFromTarget", {sessionId, targetId: tab.targetId});
    this.changed();
  }

  async releaseAll() {
    for (const tab of [...this.tabs.values()]) await this.release(tab);
  }

  async adopt(tab, params) {
    this.children.set(params.sessionId, tab.id);
    const debuggee = tab.view.webContents.debugger;
    try {
      if (this.fetchPatterns) await debuggee.sendCommand("Fetch.enable", {patterns: this.fetchPatterns}, params.sessionId);
      await debuggee.sendCommand("Target.setAutoAttach", AUTO_ATTACH, params.sessionId);
    } catch { /* the frame went away */ }
    try { await debuggee.sendCommand("Runtime.runIfWaitingForDebugger", {}, params.sessionId); } catch { /* gone */ }
  }

  onEvent(tab, method, params, sessionId) {
    if (!tab.session) return;
    if (method === "Target.attachedToTarget") this.adopt(tab, params);
    else if (method === "Target.detachedFromTarget") this.children.delete(params.sessionId);
    else this.emit(method, params, sessionId || tab.session);
  }

  async tabCommand(sessionId, method, params) {
    if ((!DOMAINS.has(method.split(".")[0]) && !ALSO.has(method)) || BLOCKED.has(method))
      throw new Error(`${method} is not available to the assistant`);
    if (method === "Page.navigate" && !WEB.test(String(params.url || ""))) throw new Error("Only web pages can be opened");
    let tab, child;
    if (sessionId.startsWith("tab:")) {
      tab = this.tabs.get(Number(sessionId.slice(4)));
      if (!tab || tab.session !== sessionId) throw new Error("Session with given id not found");
    } else {
      tab = this.tabs.get(this.children.get(sessionId));
      if (!tab || !tab.session) throw new Error("Session with given id not found");
      child = sessionId;
    }
    if (method === "Page.bringToFront" && !child) {
      this.select(tab.id);
      this.onFront(tab.id);
      return {};
    }
    return tab.view.webContents.debugger.sendCommand(method, params, child);
  }

  async browserCommand(method, params) {
    switch (method) {
      case "Browser.getVersion":
        return {protocolVersion: "1.3", product: `CodeRunner app (Chrome/${process.versions.chrome})`, revision: "",
                userAgent: this.session.getUserAgent(), jsVersion: process.versions.v8};
      case "Browser.close":
        await this.releaseAll();        // the app stays open: the assistant only lets go of its tabs
        return {};
      case "Browser.setDownloadBehavior":
        this.denyDownloads = params.behavior === "deny";
        return {};
      case "Browser.setPermission":
        return {};                      // every permission the broker denies is refused by the session already
      case "Target.setDiscoverTargets":
        this.discover = Boolean(params.discover);
        if (this.discover) for (const tab of this.tabs.values()) await this.announce(tab);
        return {};
      case "Fetch.enable":
        this.fetchPatterns = params.patterns || [{urlPattern: "*"}];
        for (const tab of this.tabs.values()) {
          if (!tab.session) continue;
          await tab.view.webContents.debugger.sendCommand("Fetch.enable", {patterns: this.fetchPatterns});
        }
        for (const [child, id] of this.children) {
          try { await this.tabs.get(id)?.view.webContents.debugger.sendCommand("Fetch.enable", {patterns: this.fetchPatterns}, child); }
          catch { /* the frame went away */ }
        }
        return {};
      case "Target.getTargets":
        return {targetInfos: await Promise.all([...this.tabs.values()].filter((t) => t.targetId).map((t) => this.info(t)))};
      case "Target.getTargetInfo":
        return {targetInfo: await this.info(this.tabByTarget(params.targetId))};
      case "Target.createTarget": {
        if (!WEB.test(String(params.url || "about:blank"))) throw new Error("Only web pages can be opened");
        const tab = await this.createTab(params.url || "about:blank", {background: false});
        tab.assistant = true;
        return {targetId: tab.targetId};
      }
      case "Target.attachToTarget":
        return {sessionId: await this.attach(this.tabByTarget(params.targetId))};
      case "Target.detachFromTarget": {
        const tab = this.tabs.get(Number(String(params.sessionId || "").replace(/^tab:/, "")));
        if (tab) await this.release(tab);
        return {};
      }
      case "Target.closeTarget":
        this.closeTab(this.tabByTarget(params.targetId).id);
        return {success: true};
      case "Target.activateTarget": {
        const tab = this.tabByTarget(params.targetId);
        this.select(tab.id);
        this.onFront(tab.id);
        return {};
      }
      default:
        throw new Error(`${method} is not available to the assistant`);
    }
  }
}

// What the owner typed in the address field, as an address: a web address as it is (https:// added), anything
// else searched for.
function addressOf(text) {
  const typed = String(text || "").trim();
  if (!typed) return "";
  if (/^https?:\/\//i.test(typed) || typed === "about:blank") return typed;
  if (/^localhost(:\d+)?(\/|$)/.test(typed)) return `http://${typed}`;
  if (/^[^\s/]+\.[^\s]{2,}$/.test(typed)) return `https://${typed}`;
  return SEARCH + encodeURIComponent(typed);
}

module.exports = {AppBrowser, PARTITION, KEEP_DEFAULT, addressOf, desktopChromeIdentity};
