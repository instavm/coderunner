// CodeRunner's desktop shell (macOS and Linux). The interface itself is served by the local daemon at /app/, so
// the same page works here and, once paired, in a browser on another device. This shell only:
//   - finds the daemon (CODERUNNER_HOME, default ~/.coderunner) and adds the owner's admin token to requests made
//     to it, and to nothing else;
//   - opens every other address in the default browser, never inside the app;
//   - denies the page every permission except notifications.
const {app, BrowserWindow, Menu, session, shell} = require("electron");
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

function waitingPage(message) {
  const html = `<!doctype html><meta charset="utf-8"><title>CodeRunner</title>
  <body style="font:15px -apple-system,system-ui,sans-serif;display:grid;place-items:center;height:100vh;margin:0;
  background:#f6f5f2;color:#1d1c1a"><div style="max-width:420px;text-align:center"><h2>CodeRunner is starting</h2>
  <p style="color:#6f6c64">${message}</p></div></body>`;
  return "data:text/html;charset=utf-8," + encodeURIComponent(html);
}

async function daemonUp() {
  try { return (await fetch(`${ORIGIN}/health`, {signal: AbortSignal.timeout(2000)})).ok; } catch { return false; }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440, height: 900, minWidth: 760, minHeight: 520, title: "CodeRunner", show: false,
    backgroundColor: "#f6f5f2",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    trafficLightPosition: {x: 16, y: 14},
    webPreferences: {contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true},
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

  const load = async (attempt = 0) => {
    if (await daemonUp()) {
      win.loadURL(`${ORIGIN}/app/`);
      return;
    }
    if (attempt === 0) {
      win.loadURL(waitingPage(`Waiting for the assistant at ${ADDRESS}. Start it with <code>coderunner up</code>.`));
    }
    setTimeout(() => load(attempt + 1), 2000);
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
  buildMenu();
  createWindow();
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
