// Talk, spoken replies and the Voice settings, seen in the real app (app/main.js) without ever showing a window:
// CODERUNNER_APP_HIDDEN=1, a temporary CODERUNNER_HOME, a stand-in for the daemon with canned answers, a
// microphone made in the page, and a stand-in for the voice model's socket (no daemon, models or sandbox are touched). Saves
// pictures of each state in light and dark, and prints what the page said.
// Usage: Electron tests/js/voice_ui_check.js OUT_DIR
"use strict";
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const WEB = path.join(ROOT, "coderunner", "server", "web", "app");
const OUT = path.resolve(process.argv[process.argv.length - 1]);
const HOME = fs.mkdtempSync(path.join(os.tmpdir(), "cr-voice-ui-"));
fs.mkdirSync(OUT, {recursive: true});

const now = () => Date.now() / 1000;
const REPLY = "Your electricity bill for October is **₹2,140**, due on 5 November. It is 12% lower than September.";
const OMNI = {model: "lfm3-omni-2b", ready: true, running: false, exclusive: true, speak_tasks: false,
              voice: "female", language: "", label: "Ready. Speaks when the assistant is idle."};
const ANSWERS = {
  "/health": {status: "ok"},
  "/api/setup": {configured: true},
  "/api/status": {models: {chat: "coderunner-bonsai-2-9b"}, browser: {mode: "computer"}, voice: ["parakeet-coreml"],
                  voice_omni: OMNI},
  "/api/conversations": [{conversation: "c1", title: "Electricity bill", status: "done", task: "t1", updated: now()}],
  "/api/conversations/c1": {tasks: [{id: "t1", prompt: "How much is my electricity bill this month?", status: "done",
                                     created: now() - 40, finished: now() - 2, reply: REPLY}],
                            events: [{task: "t1", seq: 1, type: "started", prompt: "How much is my electricity bill this month?", ts: now() - 40},
                                     {task: "t1", seq: 2, type: "done", reply: REPLY, ts: now() - 2}]},
  "/api/approvals": [],
  "/api/computer": {available: false, running: false},
  "/api/voice/settings": {stt: "auto", language: "en-IN", speak_tasks: false, voice: "female",
                          status: {ready: true, engines: ["parakeet-coreml"]}, omni: OMNI},
  "/api/voice/speak": {id: "s1", text: "Your electricity bill for October is ₹2,140.", rate: 24000, kind: "listen"},
  "/api/voice": {text: "Check trains from Delhi to Mumbai tomorrow morning on IRCTC"},
  "/api/tasks": {id: "t2", conversation: "c1", status: "running"},
  "/api/voice/talk": {ok: true},
  "/api/tailscale": {}, "/api/devices": [], "/api/channels": [],
};
const TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"};
const posted = [];
const streams = [];
let seq = 10;
const push = (event) => { for (const s of streams) s.write(`data: ${JSON.stringify({seq: ++seq, ts: Date.now() / 1000, ...event})}\n\n`); };

const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://x");
  if (url.pathname.startsWith("/app/")) {
    const file = path.join(WEB, url.pathname.slice(5) || "index.html");
    if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { response.writeHead(404); return response.end(); }
    response.writeHead(200, {"Content-Type": TYPES[path.extname(file)] || "application/octet-stream"});
    return response.end(fs.readFileSync(file));
  }
  if (url.pathname === "/api/events") {      // held open; push() sends live events to the page
    response.writeHead(200, {"Content-Type": "text/event-stream"}); streams.push(response); return;
  }
  if (url.pathname.startsWith("/api/voice/speech/")) {
    // Two seconds of quiet, so a clip "plays" long enough to be seen.
    response.writeHead(200, {"Content-Type": "audio/pcm;rate=24000;encoding=signed-int;bits=16"});
    return response.end(Buffer.alloc(24000 * 2 * 2));
  }
  let body = "";
  request.on("data", (chunk) => { body += chunk; });
  request.on("end", () => {
    if (request.method === "POST") posted.push({path: url.pathname, body});
    response.writeHead(200, {"Content-Type": "application/json"});
    response.end(JSON.stringify(ANSWERS[url.pathname] ?? {}));
  });
});

const port = 20000 + (process.pid % 20000);
fs.mkdirSync(path.join(HOME, "assistant"), {recursive: true});
fs.writeFileSync(path.join(HOME, "assistant", "config.json"), JSON.stringify({gateway: {admin_port: port}}));
Object.assign(process.env, {CODERUNNER_HOME: HOME, CODERUNNER_APP_HIDDEN: "1", CODERUNNER_APP_DATA: path.join(HOME, "electron")});
const shell = require(path.join(ROOT, "app", "main.js"));
const {app, BrowserWindow, nativeTheme} = require("electron");
app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const step = (text) => process.stderr.write(`step: ${text}\n`);
setTimeout(() => { step("gave up"); process.kill(process.pid, "SIGKILL"); }, 90000);
process.on("unhandledRejection", (error) => { step(`failed: ${error?.message || error}`); process.kill(process.pid, "SIGKILL"); });

// A microphone made in the page: the real one is never asked for (macOS would show its own prompt even for a
// hidden window).
const FAKE_MIC = `
  navigator.mediaDevices.getUserMedia = async () => {
    const ctx = new AudioContext(), tone = ctx.createOscillator(), out = ctx.createMediaStreamDestination();
    tone.connect(out); tone.start();
    return out.stream;
  };
  true`;

server.listen(port, "127.0.0.1", () => {
  async function shot(contents) {
    const debuggee = contents.debugger, mine = !debuggee.isAttached();
    if (mine) debuggee.attach("1.3");
    try { return (await debuggee.sendCommand("Page.captureScreenshot", {format: "png"})).data; }
    finally { if (mine) debuggee.detach(); }
  }

  app.whenReady().then(async () => {
    step("ready");
    const win = shell.mainWindow();
    win.setContentSize(1200, 800);
    while (!win.webContents.getURL().includes("/app/")) await wait(200);
    step("page loaded");
    await wait(1500);
    win.webContents.on("console-message", (event) => { if (event.level === "error") step(`console: ${event.message}`); });
    const js = (code) => win.webContents.executeJavaScript(code).catch((error) => {
      step(`script failed: ${code.slice(0, 120)}`);
      throw error;
    });
    const picture = async (name) => {
      for (const theme of ["light", "dark"]) {
        nativeTheme.themeSource = theme;
        await wait(500);
        fs.writeFileSync(path.join(OUT, `${name}-${theme}.png`), Buffer.from(await shot(win.webContents), "base64"));
      }
      nativeTheme.themeSource = "light";
    };
    const said = {};
    await js(FAKE_MIC);
    await js(`location.hash = "#c=c1"; true`);
    await wait(2000);
    said.composer = await js(`[...document.querySelectorAll(".composer-bar button:not([hidden])")].map((b) => b.getAttribute("aria-label") || b.textContent.trim() || b.title)`);
    said.answerActions = await js(`[...document.querySelectorAll(".ans-actions button")].map((b) => b.textContent)`);
    await picture("1-thread");

    // Listen: the clip plays once, the composer says so with a way to stop it.
    await js(`[...document.querySelectorAll(".ans-actions button")].find((b) => b.textContent === "Listen")?.click(); true`);
    await wait(700);
    said.listening = await js(`document.querySelector(".composer-shell")?.innerText`);
    await picture("2-listen");
    await wait(2500);

    // Talk: ready, listening, then the spoken request runs as a task: working, needs you, speaking, ready.
    await js(`document.getElementById("talkBtn").click(); true`);
    await wait(300);
    said.ready = await js(`document.getElementById("talkStrip").innerText`);
    await picture("3-talk-ready");
    await js(`document.getElementById("talkHold").dispatchEvent(new PointerEvent("pointerdown", {bubbles: true, pointerId: 1})); true`);
    await wait(1500);
    said.held = await js(`document.getElementById("talkStrip").innerText`);
    await picture("4-talk-listening");
    await js(`document.getElementById("talkHold").dispatchEvent(new PointerEvent("pointerup", {bubbles: true, pointerId: 1})); true`);
    await wait(1200);
    said.sent = await js(`document.getElementById("talkStrip").innerText`);
    push({type: "started", task: "t2", conversation: "c1", prompt: "Check trains from Delhi to Mumbai tomorrow morning on IRCTC"});
    push({type: "note", task: "t2", conversation: "c1", text: "Opening IRCTC to look for trains. Then the fares."});
    await wait(500);
    said.working = await js(`document.getElementById("talkStrip").innerText`);
    await picture("5-talk-working");
    push({type: "activity", event: "approval.requested", task: "t2", conversation: "c1", id: "a9", kind: "browser.mac",
          summary: "irctc.co.in blocks the sandbox browser. Continue in the browser on your Mac?", status: "pending"});
    await wait(500);
    said.needsYou = await js(`document.getElementById("talkStrip").innerText`);
    await picture("6-talk-needs-you");
    push({type: "activity", event: "approval.decided", task: "t2", conversation: "c1", id: "a9", status: "approved"});
    push({type: "voice_utterance", conversation: "c1", live: true, seq: 0,
          utterance: {id: "u1", text: "Searching trains from New Delhi to Mumbai Central.", rate: 24000, kind: "progress"}});
    await wait(700);
    said.speaking = await js(`document.getElementById("talkStrip").innerText`);
    await picture("7-talk-speaking");
    await wait(2500);
    push({type: "done", task: "t2", conversation: "c1", reply: "Three trains leave before 9."});
    await wait(500);
    said.after = await js(`document.getElementById("talkStrip").innerText`);
    await js(`document.getElementById("talkEnd").click(); true`);
    await wait(300);

    // Settings, Voice & Phone.
    await js(`document.getElementById("modelsLine")?.click() || document.getElementById("panelModelsBtn")?.click(); true`);
    await wait(800);
    await js(`[...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Voice & Phone")?.click(); true`);
    await wait(1200);
    said.settings = await js(`document.querySelector(".set-pane-body")?.innerText`);
    await picture("8-settings");
    await js(`[...document.querySelectorAll(".set-pane-body button")].find((b) => b.textContent === "Remove")?.click(); true`);
    await wait(300);
    said.removeArmed = await js(`[...document.querySelectorAll(".set-pane-body button")].some((b) => b.textContent === "Remove model?")`);
    // Not set up yet: the licence is said beside the one button.
    ANSWERS["/api/voice/settings"] = {...ANSWERS["/api/voice/settings"], omni: {...OMNI, ready: false, label: "Not installed"}};
    await js(`[...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Voice & Phone")?.click(); true`);
    await wait(1000);
    said.notInstalled = await js(`[...document.querySelectorAll(".set-group")].find((g) => g.innerText.startsWith("TALK") || g.innerText.startsWith("Talk"))?.innerText`);
    await picture("9-settings-not-installed");
    said.posted = posted.map((p) => `${p.path} ${p.body.slice(0, 140)}`).filter((line) => /^\/api\/(voice|tasks)/.test(line));
    console.log(JSON.stringify(said, null, 1));
    server.close();
    fs.rmSync(HOME, {recursive: true, force: true});
    process.stdout.write("", () => process.kill(process.pid, "SIGKILL"));
  });
});
