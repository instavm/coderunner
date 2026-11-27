// The first-run screens, seen in the real app (app/main.js) without ever showing a window: CODERUNNER_APP_HIDDEN=1,
// a temporary CODERUNNER_HOME, and a stand-in for the daemon whose /api/setup answers walk through the plan, your
// own server, Claude, progress, a licence and the reach step. Saves pictures in light and dark.
// Usage: Electron tests/js/onboarding_ui_check.js PLAN_JSON OUT_DIR   (PLAN_JSON: a SetupJob.plan() result)
"use strict";
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const WEB = path.join(ROOT, "coderunner", "server", "web", "app");
const OUT = path.resolve(process.argv[process.argv.length - 1]);
const PLAN = JSON.parse(fs.readFileSync(process.argv[process.argv.length - 2], "utf8"));
const HOME = fs.mkdtempSync(path.join(os.tmpdir(), "cr-onboard-"));
fs.mkdirSync(OUT, {recursive: true});

const STAGES = [
  {key: "runtime", label: "llama.cpp runtime", state: "done"},
  {key: "model", label: "Bonsai 2.5 27B", state: "running", percent: 42, done_gb: 6.1, total_gb: 14.5},
  {key: "embed", label: "Search model", state: "pending"},
  {key: "check", label: "First check", state: "pending"},
];
const SETUP = {
  plan: {configured: false, plan: PLAN},
  running: {state: "running", stages: STAGES, lines: ["Fetching Bonsai 2.5 27B…"]},
  license: {state: "failed", lines: [], result: {needs_license: {model: "lfm3-omni-2b", link: "https://huggingface.co/LiquidAI/LFM3-Omni-2B",
            reason: "LFM Open License v1.0; review https://huggingface.co/LiquidAI/LFM3-Omni-2B and accept it once"}}},
  done: {state: "done", lines: [], stages: STAGES.map((s) => ({...s, state: "done"}))},
};
let setup = SETUP.plan;
const ANSWERS = {
  "/health": {status: "ok"},
  "/api/status": {models: {}, voice: []},
  "/api/settings": {settings: {"owner.onboarded": false},
                    suggestions: [{name: "Documents", path: "/Users/owner/Documents"}, {name: "Desktop", path: "/Users/owner/Desktop"},
                                  {name: "Downloads", path: "/Users/owner/Downloads"}]},
  "/api/connectors": {connectors: [{name: "calendar", label: "Calendar"}, {name: "notes", label: "Notes"}, {name: "mail", label: "Mail"}]},
  "/api/conversations": [], "/api/approvals": [], "/api/computer": {available: false},
  "/api/remote": {models: [{id: "claude-haiku-5-5"}], has_key: false,
                  warning: "Tasks go to Anthropic's servers with your key: not private like a model on this Mac."},
};
const TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"};

const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://x");
  if (url.pathname.startsWith("/app/")) {
    const file = path.join(WEB, url.pathname.slice(5) || "index.html");
    if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { response.writeHead(404); return response.end(); }
    response.writeHead(200, {"Content-Type": TYPES[path.extname(file)] || "application/octet-stream"});
    return response.end(fs.readFileSync(file));
  }
  if (url.pathname === "/api/events") { response.writeHead(200, {"Content-Type": "text/event-stream"}); return; }
  request.resume();
  request.on("end", () => {
    response.writeHead(200, {"Content-Type": "application/json"});
    const body = url.pathname === "/api/setup" ? (request.method === "POST" ? {...SETUP.plan.plan} : setup) : ANSWERS[url.pathname];
    response.end(JSON.stringify(body ?? {}));
  });
});

const port = 20000 + (process.pid % 20000);
fs.mkdirSync(path.join(HOME, "assistant"), {recursive: true});
fs.writeFileSync(path.join(HOME, "assistant", "config.json"), JSON.stringify({gateway: {admin_port: port}}));
Object.assign(process.env, {CODERUNNER_HOME: HOME, CODERUNNER_APP_HIDDEN: "1", CODERUNNER_APP_DATA: path.join(HOME, "electron")});
const shell = require(path.join(ROOT, "app", "main.js"));
const {app, nativeTheme} = require("electron");
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const step = (text) => process.stderr.write(`step: ${text}\n`);
setTimeout(() => { step("gave up"); process.kill(process.pid, "SIGKILL"); }, 120000);
process.on("unhandledRejection", (error) => { step(`failed: ${error?.message || error}`); process.kill(process.pid, "SIGKILL"); });

server.listen(port, "127.0.0.1", () => {
  async function shot(contents) {
    const debuggee = contents.debugger, mine = !debuggee.isAttached();
    if (mine) debuggee.attach("1.3");
    try { return (await debuggee.sendCommand("Page.captureScreenshot", {format: "png"})).data; }
    finally { if (mine) debuggee.detach(); }
  }
  app.whenReady().then(async () => {
    const win = shell.mainWindow();
    win.setContentSize(1152, 720);
    while (!win.webContents.getURL().includes("/app/")) await wait(200);
    await wait(1800);
    win.webContents.on("console-message", (event) => { if (event.level === "error") step(`console: ${event.message}`); });
    const js = (code) => win.webContents.executeJavaScript(code);
    const picture = async (name, tall = false) => {
      if (tall) {
        const h = await js(`Math.ceil(Math.max(document.querySelector(".setup")?.getBoundingClientRect().bottom || 0, 720) + 60)`);
        win.setContentSize(1152, Math.min(h, 2400)); await wait(500);
      }
      for (const theme of ["light", "dark"]) {
        nativeTheme.themeSource = theme;
        await wait(600);
        fs.writeFileSync(path.join(OUT, `${name}-${theme}.png`), Buffer.from(await shot(win.webContents), "base64"));
      }
      nativeTheme.themeSource = "light";
      if (tall) { win.setContentSize(1152, 720); await wait(300); }
    };
    const said = {};
    const text = () => js(`document.querySelector(".setup")?.innerText || document.body.innerText.slice(0, 600)`);
    const press = (pattern) => js(`(() => { const b = [...document.querySelectorAll("button, a, [role=button]")].find((b) => ${pattern}.test(b.textContent) && b.offsetParent); b?.click(); return b ? b.textContent : null; })()`);
    // Misaligned or overflowing pieces of the card, measured: anything wider than the card, or text cut off.
    const audit = () => js(`(() => {
      const card = document.querySelector(".setup"); if (!card) return [];
      const box = card.getBoundingClientRect(), out = [];
      for (const n of card.querySelectorAll("*")) {
        const r = n.getBoundingClientRect(); if (!r.width) continue;
        if (r.right > box.right + 1 || r.left < box.left - 1) out.push("outside the card: " + n.className + " " + (n.textContent || "").trim().slice(0, 40));
        if (n.scrollWidth > n.clientWidth + 1 && getComputedStyle(n).overflow !== "visible" && n.clientWidth) out.push("cut off: " + n.className + " " + (n.textContent || "").trim().slice(0, 40));
      }
      return [...new Set(out)].slice(0, 12);
    })()`);

    said.plan = await text(); said.planAudit = await audit();
    await picture("1-plan");
    await picture("1-plan-full", true);
    said.server = await press(/own model server/i); await wait(800);
    said.serverText = await text(); said.serverAudit = await audit();
    await picture("2-server");
    await js(`location.reload(); true`); await wait(2500);
    said.claude = await press(/anthropic key/i); await wait(800);
    said.claudeText = await text(); said.claudeAudit = await audit();
    await picture("3-claude");
    setup = SETUP.running; await js(`location.reload(); true`); await wait(2500);
    said.progress = await text(); said.progressAudit = await audit();
    await picture("4-progress");
    setup = SETUP.license; await js(`location.reload(); true`); await wait(2500);
    said.license = await text(); said.licenseAudit = await audit();
    await picture("5-license");
    setup = SETUP.running; await js(`location.reload(); true`); await wait(2500);
    setup = {configured: true};                  // set up: the next poll brings the reach step
    await wait(4000);
    said.reach = await text(); said.reachAudit = await audit();
    await picture("6-reach", true);
    console.log(JSON.stringify(said, null, 1));
    server.close();
    fs.rmSync(HOME, {recursive: true, force: true});
    process.stdout.write("", () => process.kill(process.pid, "SIGKILL"));
  });
});
