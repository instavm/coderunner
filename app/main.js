// CodeRunner's desktop shell (macOS and Linux). The interface itself is served by the local daemon at /app/, so
// the same page works here and, once paired, in a browser on another device. This shell only:
//   - finds the daemon (CODERUNNER_HOME, default ~/.coderunner), starts it with `coderunner up` when it is not
//     running, and adds the owner's admin token to requests made to it, and to nothing else;
//   - opens every other address in the default browser, never inside the app;
//   - denies the page every permission except notifications;
//   - shows the system's folder dialog when the page asks where to keep the models (preload.js).
const {app, BrowserWindow, Menu, dialog, ipcMain, nativeTheme, session, shell} = require("electron");
const {execFile} = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const HOME = process.env.CODERUNNER_HOME || path.join(os.homedir(), ".coderunner");
const ASSISTANT = path.join(HOME, "assistant");

function readConfig() {
  try { return JSON.parse(fs.readFileSync(path.join(ASSISTANT, "config.json"), "utf8")); } catch { return {}; }
}

function daemonAddress() {
  const gateway = readConfig().gateway || {};
  let host = gateway.admin_bind || "127.0.0.1";
  if (host === "0.0.0.0" || host === "::") host = "127.0.0.1";
  return `${host}:${gateway.admin_port || 8224}`;
}

function adminToken() {
  try { return fs.readFileSync(path.join(ASSISTANT, "admin-token"), "utf8").trim(); } catch { return ""; }
}

const ADDRESS = daemonAddress();
const ORIGIN = `http://${ADDRESS}`;

const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// What the window shows until the daemon answers: the same look as the page's own setup screens, light or dark
// with the system.
function waitingPage(title, message) {
  const html = `<!doctype html><meta charset="utf-8"><title>CodeRunner</title><style>
  :root { color-scheme: light dark; --bg: #f6f6f7; --text: #17181c; --muted: #696c76; }
  @media (prefers-color-scheme: dark) { :root { --bg: #101114; --text: #e9eaee; --muted: #9093a0; } }
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, system-ui, sans-serif; display: grid; place-items: center;
    height: 100vh; margin: 0; background: var(--bg); color: var(--text); -webkit-app-region: drag; }
  div { max-width: 460px; padding: 0 24px; text-align: center; }
  i { display: inline-block; width: 44px; height: 44px; border-radius: 11px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 8l4 4-4 4M13 16h4'/%3E%3C/svg%3E") center / 74% no-repeat,
      #2563eb; }
  h1 { font-size: 25px; font-weight: 650; letter-spacing: -0.02em; margin: 14px 0 4px; }
  p { color: var(--muted); margin: 0; }
  </style><body><div><i></i><h1>${escapeHtml(title)}</h1><p>${escapeHtml(message)}</p></div></body>`;
  return "data:text/html;charset=utf-8," + encodeURIComponent(html);
}

async function daemonUp() {
  try { return (await fetch(`${ORIGIN}/health`, {signal: AbortSignal.timeout(2000)})).ok; } catch { return false; }
}

// The `coderunner` command: where the installer puts it, since an app opened from the Dock or a launcher gets a
// bare PATH.
function coderunnerCommand() {
  const places = [path.join(HOME, "venv", "bin", "coderunner"), path.join(os.homedir(), ".local", "bin", "coderunner"),
                  "/opt/homebrew/bin/coderunner", "/usr/local/bin/coderunner", "/usr/bin/coderunner"];
  return places.find((file) => { try { fs.accessSync(file, fs.constants.X_OK); return true; } catch { return false; } }) || null;
}

// `coderunner up` starts the sandbox and the daemon and returns; the daemon keeps running after this app quits,
// as the login service's does. Started once per launch: `failure` holds what it said when it could not.
let starting = false;
let failure = "";
function startDaemon() {
  const command = coderunnerCommand();
  if (!command) return false;
  if (starting) return true;
  starting = true;
  const tools = ["/opt/homebrew/bin", "/usr/local/bin", path.join(os.homedir(), ".local", "bin")];   // container, docker
  const env = {...process.env, CODERUNNER_HOME: HOME, PATH: [...tools, process.env.PATH || "/usr/bin:/bin"].join(":")};
  execFile(command, ["up"], {env, timeout: 180000}, (error, stdout, stderr) => {
    if (error) failure = `${stdout}${stderr}`.trim().split("\n").slice(-2).join(" ") || String(error.message);
  });
  return true;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1152, height: 720, minWidth: 760, minHeight: 520, title: "CodeRunner", show: false,
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#101114" : "#f6f6f7",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    trafficLightPosition: {x: 16, y: 14},
    webPreferences: {contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true,
                     preload: path.join(__dirname, "preload.js")},
  });
  win.once("ready-to-show", () => win.show());

  // Links to anywhere but the daemon open in the default browser.
  win.webContents.setWindowOpenHandler(({url}) => {
    if (/^https?:\/\//.test(url) && !url.startsWith(ORIGIN)) shell.openExternal(url);
    return {action: "deny"};
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(ORIGIN) && !url.startsWith("data:")) {
      event.preventDefault();
      if (/^https?:\/\//.test(url)) shell.openExternal(url);
    }
  });

  let shown = "";
  const show = (title, message) => {
    if (shown !== title + message) win.loadURL(waitingPage(title, message));
    shown = title + message;
  };
  const load = async () => {
    if (await daemonUp()) {
      win.loadURL(`${ORIGIN}/app/`);
      return;
    }
    if (!startDaemon()) {
      show("CodeRunner is not installed yet", "Run ./install.sh from the CodeRunner checkout, then open this app again.");
    } else if (failure) {
      show("The assistant did not start", `${failure} Run \`coderunner up\` in a terminal to see more.`);
    } else {
      show("CodeRunner is starting", `Starting the assistant at ${ADDRESS}…`);
    }
    setTimeout(load, 2000);
  };
  load();
  return win;
}

function buildMenu() {
  const template = [
    ...(process.platform === "darwin" ? [{role: "appMenu"}] : []),
    {label: "File", submenu: [
      {label: "New Task", accelerator: "CmdOrCtrl+N",
       click: (_, win) => win?.webContents.executeJavaScript("document.getElementById('newTask')?.click()")},
      {type: "separator"}, process.platform === "darwin" ? {role: "close"} : {role: "quit"},
    ]},
    {role: "editMenu"},
    {label: "View", submenu: [{role: "reload"}, {role: "toggleDevTools"}, {type: "separator"},
                               {role: "resetZoom"}, {role: "zoomIn"}, {role: "zoomOut"}, {role: "togglefullscreen"}]},
    {role: "windowMenu"},
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(() => {
  const token = adminToken();
  // The admin token rides only on requests to the daemon, websockets included (the computer's screen).
  session.defaultSession.webRequest.onBeforeSendHeaders(
    {urls: [`http://${ADDRESS}/*`, `ws://${ADDRESS}/*`]},
    (details, callback) => {
      if (token) details.requestHeaders["Authorization"] = `Bearer ${token}`;
      callback({requestHeaders: details.requestHeaders});
    });
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback) => {
    callback(permission === "notifications" && contents.getURL().startsWith(ORIGIN));
  });
  // The one thing the page asks of the shell: a folder picked by the owner (where to keep the models). Only the
  // daemon's own page may ask.
  ipcMain.handle("pick-folder", async (event, start) => {
    let asker = "";
    try { asker = new URL(event.senderFrame.url).origin; } catch { /* no frame, or no address: refused */ }
    if (asker !== ORIGIN) return null;
    const picked = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), {
      title: "Where to keep the models", buttonLabel: "Keep them here", defaultPath: start || undefined,
      properties: ["openDirectory", "createDirectory"],
    });
    return picked.canceled ? null : picked.filePaths[0];
  });
  buildMenu();
  createWindow();
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
