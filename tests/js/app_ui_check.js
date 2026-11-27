// The app's Browser tab and the "continue on your Mac" question, seen in the real app (app/main.js) without ever
// showing a window: CODERUNNER_APP_HIDDEN=1, a temporary CODERUNNER_HOME, and a small stand-in for the daemon that
// serves the app's page and canned answers (no daemon, models or sandbox are touched). Saves pictures of the page
// with the browser drawn where the app draws it, in light and dark.
// Usage: Electron tests/js/app_ui_check.js OUT_DIR
"use strict";
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const WEB = path.join(ROOT, "coderunner", "server", "web", "app");
const OUT = path.resolve(process.argv[process.argv.length - 1]);
const HOME = fs.mkdtempSync(path.join(os.tmpdir(), "cr-ui-"));
fs.mkdirSync(OUT, {recursive: true});

const now = () => Date.now() / 1000;
const ASK = {id: "a1", kind: "browser.mac", target: "shop.example", risk: "medium", status: "pending", origin: "agent",
             summary: "shop.example blocks the sandbox browser. Continue in CodeRunner's built-in browser on your Mac?",
             detail: {site: "shop.example", wall: "Cloudflare", sign_in: "chrome"}};
const ANSWERS = {
  "/health": {status: "ok"},
  "/api/setup": {configured: true},
  "/api/status": {models: {chat: "coderunner-bonsai-2-9b"}, browser: {mode: "computer"}},
  "/api/conversations": [{conversation: "c1", title: "Price of the Aeropress", status: "running", task: "t1", updated: now()}],
  "/api/conversations/c1": {tasks: [{id: "t1", prompt: "What does shop.example charge for the Aeropress Go?", status: "running", created: now()}],
                            events: [{task: "t1", seq: 1, type: "activity", event: "approval.requested", ts: now(), ...ASK}]},
  "/api/approvals": [ASK],
  "/api/computer": {available: false, running: false, state: "working", message: "Getting the computer ready…"},
  "/api/browser/mac": {available: true, sites: [{id: 1, site: "news.example"}], browsers: [{id: "chrome", name: "Chrome"}]},
};
const TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"};
const PAGE = `<!doctype html><title>Aeropress Go</title><style>body{font:16px -apple-system,sans-serif;margin:32px;color:#222}
h1{font-size:26px} .price{font-size:22px;color:#0a6}</style><h1>Aeropress Go</h1><p class="price">$39.95</p><p>In stock.</p>`;

const site = http.createServer((request, response) => { response.writeHead(200, {"Content-Type": "text/html"}); response.end(PAGE); });
const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://x");
  if (url.pathname.startsWith("/app/")) {
    const file = path.join(WEB, url.pathname.slice(5) || "index.html");
    if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { response.writeHead(404); return response.end(); }
    response.writeHead(200, {"Content-Type": TYPES[path.extname(file)] || "application/octet-stream"});
    return response.end(fs.readFileSync(file));
  }
  if (url.pathname === "/api/events") { response.writeHead(200, {"Content-Type": "text/event-stream"}); return; }   // held open
  response.writeHead(200, {"Content-Type": "application/json"});
  response.end(JSON.stringify(ANSWERS[url.pathname] ?? {}));
});

// The shell reads its settings as it loads, before the app is ready (its switches count only then), so the
// stand-in's port is chosen first.
const port = 20000 + (process.pid % 20000);
fs.mkdirSync(path.join(HOME, "assistant"), {recursive: true});
fs.writeFileSync(path.join(HOME, "assistant", "config.json"), JSON.stringify({gateway: {admin_port: port}}));
Object.assign(process.env, {CODERUNNER_HOME: HOME, CODERUNNER_APP_HIDDEN: "1", CODERUNNER_APP_DATA: path.join(HOME, "electron")});
const shell = require(path.join(ROOT, "app", "main.js"));
const {app, BrowserWindow, nativeTheme} = require("electron");
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const step = (text) => process.stderr.write(`step: ${text}\n`);
setTimeout(() => { step("gave up"); process.kill(process.pid, "SIGKILL"); }, 60000);
process.on("unhandledRejection", (error) => { step(`failed: ${error?.message || error}`); process.kill(process.pid, "SIGKILL"); });

server.listen(port, "127.0.0.1", () => {

  // A PNG of a page through its own protocol connection: in a window that is never shown, capturePage has no
  // surface to read, while the page's renderer still draws.
  async function shot(contents) {
    const debuggee = contents.debugger, mine = !debuggee.isAttached();
    if (mine) debuggee.attach("1.3");
    try { return (await debuggee.sendCommand("Page.captureScreenshot", {format: "png"})).data; }
    finally { if (mine) debuggee.detach(); }
  }

  async function picture(name) {
    const win = shell.mainWindow();
    const browser = shell.browser();
    const tab = browser.tabs.get(browser.active);
    const box = browser.bounds;
    const [width, height] = win.getContentSize();
    const parts = [`<img src="data:image/png;base64,${await shot(win.webContents)}" style="position:absolute;left:0;top:0;width:${width}px;height:${height}px">`];
    if (tab && browser.shown)
      parts.push(`<img src="data:image/png;base64,${await shot(tab.view.webContents)}" style="position:absolute;left:${box.x}px;top:${box.y}px;width:${box.width}px;height:${box.height}px">`);
    // The page and the browser view drawn together, as the window shows them, in another window nobody sees.
    const paper = new BrowserWindow({show: false, width, height, useContentSize: true, paintWhenInitiallyHidden: true});
    await paper.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(`<body style="margin:0">${parts.join("")}</body>`));
    await wait(500);
    fs.writeFileSync(path.join(OUT, name), Buffer.from(await shot(paper.webContents), "base64"));
    paper.destroy();
  }

  app.whenReady().then(async () => {
    step("ready");
    await wait(500); step("paper");
    const win = shell.mainWindow();
    win.setContentSize(1440, 900);
    while (!win.webContents.getURL().includes("/app/")) await wait(200);
    step("page loaded");
    await wait(1500);
    await new Promise((resolve) => site.listen(0, "127.0.0.1", resolve));
    await shell.browser().createTab(`http://127.0.0.1:${site.address().port}/shop`);
    await win.webContents.executeJavaScript(`location.hash = "#c=c1"; document.getElementById("viewBrowser").click(); true`);
    await wait(2500);
    const placed = shell.browser().shown;
    nativeTheme.themeSource = "light";
    await wait(1200);
    await picture("app-ui-light.png");
    nativeTheme.themeSource = "dark";
    await wait(1200);
    await picture("app-ui-dark.png");
    // The "..." menu beside the address, then its two forms, in light.
    nativeTheme.themeSource = "light";
    await wait(800);
    const press = (selector) => win.webContents.executeJavaScript(`document.querySelector(${JSON.stringify(selector)}).click(); true`);
    await press("#bMore");
    await wait(800);
    const menu = await win.webContents.executeJavaScript(`document.getElementById("bMenu").innerText`);
    await picture("app-ui-menu.png");
    await win.webContents.executeJavaScript(`[...document.querySelectorAll("#bMenu .b-item")].find((b) => b.textContent.startsWith("Import")).click(); true`);
    await wait(800);
    await picture("app-ui-import.png");
    await win.webContents.executeJavaScript(`document.querySelector("#bMenu .btn:not(.primary)").click(); true`);
    await wait(600);
    await press("#bMore");
    await wait(600);
    await win.webContents.executeJavaScript(`document.querySelector("#bMenu .b-item.danger").click(); true`);
    await wait(500);
    await picture("app-ui-clear.png");
    const covered = !shell.browser().shown;
    const text = await win.webContents.executeJavaScript(`[...document.querySelectorAll(".ask")].map((c) => c.innerText).join("\\n---\\n")`);
    console.log(JSON.stringify({placed, covered, menu, bounds: shell.browser().bounds, ask: text,
                                tabs: await win.webContents.executeJavaScript(`document.getElementById("bTabs").innerText`)}));
    server.close();
    site.close();
    fs.rmSync(HOME, {recursive: true, force: true});
    process.stdout.write("", () => process.kill(process.pid, "SIGKILL"));
  });
});
