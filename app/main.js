// CodeRunner's desktop shell (macOS and Linux). The interface itself is served by the local daemon at /app/, so
// the same page works here and, once paired, in a browser on another device. This shell only:
//   - finds the daemon (CODERUNNER_HOME, default ~/.coderunner), starts it with `coderunner up` when it is not
//     running, and adds the owner's admin token to requests made to it, and to nothing else;
//   - opens every other address in the default browser, never inside the app;
//   - denies the page every permission except notifications;
//   - shows the system's folder dialog when the page asks where to keep the models (preload.js);
//   - brings the daemon back when it dies while the app is open, unless the owner stopped it (`coderunner down`
//     leaves a mark): then the page offers "Start it", which asks this shell;
//   - keeps a menu bar icon (macOS) that says whether tasks run, need the owner or finished unseen, with a small
//     popover of the latest threads and a quick-ask window on a global shortcut (both pages served by the daemon,
//     tray.html and ask.html), and tells the owner when a thread they are not looking at finishes or needs them.
//     Closing the window on a Mac leaves the app in the menu bar; Cmd+Q quits;
//   - keeps Developer Mode (Help menu, off unless turned on), in which the page offers a thread's trace;
//   - keeps the computer panel's Browser tab (browser.js): a browser on this Mac that a site blocking the sandbox's
//     browser continues in, with the owner's OK, driven by the daemon over a socket only the owner can open.
// CODERUNNER_APP_HIDDEN=1 runs it for a test: no window is ever shown, no menu bar icon, no shortcut, nothing focused.
const {app, BrowserWindow, Menu, Notification, Tray, dialog, globalShortcut, ipcMain, nativeImage, nativeTheme, screen,
       session, shell} = require("electron");
const {execFile} = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const {AppBrowser} = require("./browser.js");

const HIDDEN = process.env.CODERUNNER_APP_HIDDEN === "1";
if (HIDDEN) {
  if (process.env.CODERUNNER_APP_DATA) app.setPath("userData", process.env.CODERUNNER_APP_DATA);
  app.commandLine.appendSwitch("use-mock-keychain");        // a hidden Keychain prompt would hang every page
  if (process.platform === "darwin") { app.setActivationPolicy("accessory"); app.dock?.hide(); }
}

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
const MAC = process.platform === "darwin";

// The token, read again until there is one: on a first install the daemon writes it after the app has opened.
let token = "";
const currentToken = () => token || (token = adminToken());
// What the shell itself asks the daemon (the menu bar's state), with the same token the page's requests get.
const daemonHeaders = () => (currentToken() ? {Authorization: `Bearer ${currentToken()}`} : {});

const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// What the window shows until the daemon answers: the same look as the page's own setup screens, light or dark
// with the system.
function waitingPage(title, message) {
  const html = `<!doctype html><meta charset="utf-8"><title>CodeRunner</title><style>
  :root { color-scheme: light dark; --bg: #faf9f7; --text: #1f1e1b; --muted: #75736d; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1a1918; --text: #ecebe8; --muted: #9c9a94; } }
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
  if (HIDDEN) return false;
  const command = coderunnerCommand();
  if (!command) return false;
  if (starting) return true;
  starting = true;
  const tools = ["/opt/homebrew/bin", "/usr/local/bin", path.join(os.homedir(), ".local", "bin")];   // container, docker
  const env = {...process.env, CODERUNNER_HOME: HOME, PATH: [...tools, process.env.PATH || "/usr/bin:/bin"].join(":")};
  execFile(command, ["up"], {env, timeout: 180000}, (error, stdout, stderr) => {
    starting = false;               // it can be started again if it dies later
    failure = error ? `${stdout}${stderr}`.trim().split("\n").slice(-2).join(" ") || String(error.message) : "";
  });
  return true;
}

// The owner stopped the assistant themselves: it is not brought back behind their back.
function stoppedByOwner() {
  return fs.existsSync(path.join(ASSISTANT, "run", "stopped-by-owner"));
}

// While the app is open, a daemon that has died (a crash, a kill, a restart that did not come back) is started
// again. Two misses in a row, so a daemon restarting in place is left to it.
let misses = 0;
function watchDaemon() {
  setInterval(async () => {
    if (await daemonUp()) { misses = 0; return; }
    misses += 1;
    if (misses >= 2 && !stoppedByOwner()) startDaemon();
  }, 5000);
}

const PAGE_PREFERENCES = {contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true,
                          preload: path.join(__dirname, "preload.js")};

// Links to anywhere but the daemon open in the default browser, from every window of the app.
function keepToDaemon(win) {
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
}

// The main window, and the thread it shows (the page says which: thread-open). `conversation`: a thread to open
// once the page is there.
let mainWindow = null;
let viewed = "";
let appBrowser = null;          // browser.js: the Browser tab's tabs and the daemon's connection to them
// `open`: what to show once the page is there, as the address's hash ("c=<id>", "computer=expanded").
function createWindow(open = "") {
  const win = new BrowserWindow({
    width: 1152, height: 720, minWidth: 760, minHeight: 520, title: "CodeRunner", show: false,
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#1a1918" : "#faf9f7",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    trafficLightPosition: {x: 16, y: 14},
    webPreferences: PAGE_PREFERENCES,
  });
  win.once("ready-to-show", () => { if (!HIDDEN) win.show(); });
  win.on("focus", () => seen(viewed));
  // The Browser tab's views live in this window while it is open; a new page (a reload) hides them until it
  // says where they go.
  appBrowser?.setHost(win);
  win.webContents.on("did-start-navigation", (details) => { if (details.isMainFrame && !details.isSameDocument) appBrowser?.place(null, false); });
  win.on("closed", () => {
    if (mainWindow === win) { mainWindow = null; viewed = ""; }
    if (!MAC) app.quit();           // elsewhere the app is its window, as before
  });
  keepToDaemon(win);
  win.pending = open;

  let shown = "";
  const show = (title, message) => {
    if (shown !== title + message) win.loadURL(waitingPage(title, message));
    shown = title + message;
  };
  const load = async () => {
    const up = await daemonUp();
    if (win.isDestroyed()) return;              // closed while the daemon was starting
    if (up) {
      win.loadURL(`${ORIGIN}/app/` + (win.pending ? `#${win.pending}` : ""));
      win.pending = "";
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

// Shows the main window, on a thread when one is given, or with the assistant's computer as large as the window
// (computer): a window still opening is given it as it loads (#c=<id>, #computer=expanded), a page already
// there is told (show-thread, show-computer).
function showMain(conversation = "", computer = false) {
  const open = computer ? "computer=expanded" : conversation ? `c=${encodeURIComponent(conversation)}` : "";
  const tell = (contents) => (computer ? contents.send("show-computer") : contents.send("show-thread", conversation));
  if (!mainWindow || mainWindow.isDestroyed()) {
    mainWindow = createWindow(open);
  } else {
    const contents = mainWindow.webContents;
    if (open) {
      if (!contents.getURL().startsWith(`${ORIGIN}/app/`)) mainWindow.pending = open;   // the daemon is starting
      else if (contents.isLoading()) contents.once("did-finish-load", () => tell(contents));
      else tell(contents);
    }
    if (mainWindow.isMinimized()) mainWindow.restore();
    if (!HIDDEN) { mainWindow.show(); mainWindow.focus(); }
  }
  if (MAC && !HIDDEN) app.focus({steal: true});
  hidePanels();
}

function sendToMain(channel, payload) {
  showMain();
  const contents = mainWindow?.webContents;
  if (!contents) return;
  const fire = () => (payload === undefined ? contents.send(channel) : contents.send(channel, payload));
  if (contents.isLoading()) contents.once("did-finish-load", fire);
  else fire();
}

function buildMenu() {
  const appSubmenu = [
    {role: "about"},
    {type: "separator"},
    {label: "Settings…", accelerator: "CmdOrCtrl+,", click: () => sendToMain("open-settings", "")},
    {type: "separator"},
    {role: "services"},
    {type: "separator"},
    {role: "hide"}, {role: "hideOthers"}, {role: "unhide"},
    {type: "separator"},
    {role: "quit"},
  ];
  const template = [
    ...(process.platform === "darwin" ? [{label: app.name, submenu: appSubmenu}] : []),
    {label: "File", submenu: [
      {label: "New Task", accelerator: "CmdOrCtrl+N",
       click: () => {
         const open = mainWindow && !mainWindow.isDestroyed();
         showMain();
         if (open) mainWindow.webContents.executeJavaScript("document.getElementById('newTask')?.click()");
       }},
      ...(process.platform !== "darwin" ? [{label: "Settings…", accelerator: "CmdOrCtrl+,", click: () => sendToMain("open-settings", "")}] : []),
      {type: "separator"}, process.platform === "darwin" ? {role: "close"} : {role: "quit"},
    ]},
    {role: "editMenu"},
    {label: "View", submenu: [{role: "reload"}, {role: "toggleDevTools"}, {type: "separator"},
                               {role: "resetZoom"}, {role: "zoomIn"}, {role: "zoomOut"}, {role: "togglefullscreen"}]},
    {role: "windowMenu"},
    {role: "help", submenu: [
      {label: "Test this Mac", click: () => sendToMain("test-this-mac")},
      {type: "separator"},
      // Off unless turned on: the open thread then offers "Export trace", a file for the developers.
      {label: "Developer Mode", type: "checkbox", checked: developerMode(), click: (item) => setDeveloperMode(item.checked)},
    ]},
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---- the menu bar icon -------------------------------------------------------------------------------------

// The >_ mark drawn for the menu bar (app/tray, @1x and @2x): a template image the system tints for a light or
// dark bar, the same with a dot (working, or finished unseen), and in colour for "needs you".
const trayImage = (name) => {
  const image = nativeImage.createFromPath(path.join(__dirname, "tray", `${name}.png`));
  if (name.endsWith("Template")) image.setTemplateImage(true);
  return image;
};
let tray = null;
// The latest threads as the daemon lists them (/api/tray), how many run and how many wait for the owner, and
// those that finished while the owner was not looking at them (a dot until the popover or that thread opens).
const trayState = {ready: false, at: 0, running: 0, waiting: 0, computer: false, threads: new Map(), unseen: new Set(),
                   image: ""};

function drawTray() {
  if (!tray || tray.isDestroyed()) return;
  const {running, waiting, unseen} = trayState;
  if (MAC) {
    // Needs you is the one state meant to catch the eye: the mark in the bar's own colour with an amber dot.
    const name = waiting ? (nativeTheme.shouldUseDarkColors ? "trayNeedsYouDark" : "trayNeedsYou")
      : running || unseen.size ? "trayDotTemplate" : "trayTemplate";
    if (name !== trayState.image) { tray.setImage(trayImage(name)); trayState.image = name; }
    tray.setTitle(running >= 2 ? String(running) : "", {fontType: "monospacedDigit"});
  }
  tray.setToolTip(waiting ? "CodeRunner needs you" : running ? `CodeRunner: ${running} working` : "CodeRunner");
}

// The owner is looking at this thread: the main window shows it and has the focus.
const looking = (conversation) => Boolean(mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible() &&
                                         mainWindow.isFocused() && viewed === conversation);

function seen(conversation) {
  if (conversation && trayState.unseen.delete(conversation)) drawTray();
}

const RUNNING = ["running", "queued"];
async function refreshTray() {
  let data;
  try {
    const response = await fetch(`${ORIGIN}/api/tray`, {headers: daemonHeaders(), signal: AbortSignal.timeout(4000)});
    if (!response.ok) return;
    data = await response.json();
  } catch { return; }       // the daemon is away: the icon keeps what it said
  const before = trayState.threads, now = new Map();
  for (const thread of data.threads || []) {
    now.set(thread.conversation, thread);
    if (!trayState.ready) continue;               // the first look only learns where things stand
    // A thread not seen before that ended, or asks, since the last look ran in between.
    const was = before.get(thread.conversation) ||
      ((thread.finished || 0) * 1000 > trayState.at || thread.waiting ? {status: "running", waiting: false, task: thread.task} : null);
    if (!was || looking(thread.conversation)) continue;
    const name = thread.title || "CodeRunner";
    if (thread.waiting && !was.waiting) notify(name, "Needs you", thread.conversation);
    if (["done", "failed"].includes(thread.status) && (RUNNING.includes(was.status) || was.task !== thread.task)) {
      trayState.unseen.add(thread.conversation);
      notify(name, thread.status === "failed" ? "Failed" : thread.reply || "Done", thread.conversation);
    }
  }
  Object.assign(trayState, {threads: now, ready: true, at: Date.now(), running: data.running || 0, waiting: data.waiting || 0,
                            computer: Boolean(data.computer)});
  drawTray();
  sizePopover();
}

// Looked again just after anything that changes what the icon says, and every few seconds while a task runs
// (slowly when none does), in case the event stream is down.
let refreshTimer = null;
function refreshSoon(delay = 250) {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    await refreshTray();
    refreshSoon(trayState.running || trayState.waiting ? 4000 : 15000);
  }, delay);
}

const CHANGES = ["started", "queued", "resumed", "paused", "done", "failed", "cancelled", "title"];
async function followEvents() {
  for (;;) {
    try {
      const response = await fetch(`${ORIGIN}/api/events?since=-1`, {headers: daemonHeaders()});
      if (response.ok && response.body) {
        refreshSoon();
        const decoder = new TextDecoder();
        let buffer = "";
        for await (const chunk of response.body) {
          buffer += decoder.decode(chunk, {stream: true});
          for (let end = buffer.indexOf("\n\n"); end >= 0; end = buffer.indexOf("\n\n")) {
            const data = buffer.slice(0, end).split("\n").filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trim()).join("\n");
            buffer = buffer.slice(end + 2);
            let e = null;
            try { e = data ? JSON.parse(data) : null; } catch { /* a keep-alive, or not an event */ }
            const event = e?.type === "activity" ? String(e.event || "") : "";
            if (e && (CHANGES.includes(e.type) || event.startsWith("approval.") ||
                      (!trayState.computer && /^(browser|driver)\./.test(event))))     // the computer got busy
              refreshSoon();
          }
        }
      }
    } catch { /* the daemon is away: tried again below */ }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

// A notification for a thread the owner is not looking at; a click opens that thread.
const notes = new Set();          // kept, or a notification's click is lost when it is collected
function notify(title, body, conversation) {
  if (!Notification.isSupported()) return;
  const note = new Notification({title, body: String(body).slice(0, 160)});
  notes.add(note);
  if (notes.size > 20) notes.delete(notes.values().next().value);
  note.on("click", () => { notes.delete(note); showMain(conversation); });
  note.on("close", () => notes.delete(note));
  note.show();
}

function trayMenu() {
  return Menu.buildFromTemplate([
    {label: "Open CodeRunner", click: () => showMain()},
    {label: "Quick Ask", click: () => toggleAsk()},
    {label: "Settings…", click: () => sendToMain("open-settings", "")},
    {type: "separator"},
    {label: "Quit CodeRunner", role: "quit"},
  ]);
}

function createTray() {
  try {
    if (MAC) {
      tray = new Tray(trayImage("trayTemplate"));
      trayState.image = "trayTemplate";
      tray.on("click", () => togglePopover());
      tray.on("right-click", () => tray.popUpContextMenu(trayMenu()));
    } else {                      // elsewhere: the app's icon with a menu
      tray = new Tray(nativeImage.createFromPath(path.join(__dirname, "icon.png")).resize({width: 22, height: 22}));
      tray.setContextMenu(trayMenu());
    }
    tray.setToolTip("CodeRunner");
  } catch { tray = null; }        // no menu bar or tray here
}

// ---- the popover and the quick ask -------------------------------------------------------------------------

// A small window over the others: frameless and rounded; on a Mac a panel, which takes the keyboard without
// bringing the app forward, with the system's material behind the page. Gone when it loses the focus.
// clear: a see-through window the page draws itself into, its rounded bar and soft shadow (the quick ask).
function panelWindow(width, height, clear = false) {
  const look = clear ? {transparent: true, backgroundColor: "#00000000", hasShadow: false}
    : MAC ? {vibrancy: "popover", visualEffectState: "active", backgroundColor: "#00000000", roundedCorners: true, hasShadow: true}
      : {backgroundColor: nativeTheme.shouldUseDarkColors ? "#222120" : "#ffffff"};
  const win = new BrowserWindow({
    width, height, show: false, frame: false, resizable: false, minimizable: false, maximizable: false,
    fullscreenable: false, skipTaskbar: true, alwaysOnTop: true, title: "CodeRunner",
    ...(MAC ? {type: "panel"} : {}), ...look,
    webPreferences: PAGE_PREFERENCES,
  });
  keepToDaemon(win);
  win.on("blur", () => {
    if (win.webContents.isDevToolsOpened()) return;
    win.hide();
    win.hiddenAt = Date.now();
  });
  win.webContents.on("did-fail-load", () => { win.failed = true; });
  return win;
}

// One quiet line where a panel's page would be, while the daemon does not answer.
function panelNote(text) {
  const html = `<!doctype html><meta charset="utf-8"><title>CodeRunner</title><style>
  :root { color-scheme: light dark; --text: #75736d; }
  @media (prefers-color-scheme: dark) { :root { --text: #9c9a94; } }
  body { font: 13px -apple-system, BlinkMacSystemFont, system-ui, sans-serif; margin: 0; height: 100vh;
    display: grid; place-items: center; color: var(--text); background: transparent; }
  </style><body>${escapeHtml(text)}</body>`;
  return "data:text/html;charset=utf-8," + encodeURIComponent(html);
}

async function loadPanel(win, page) {
  if (win.webContents.getURL().startsWith(`${ORIGIN}/app/${page}`) && !win.failed) return;
  win.failed = false;
  const up = await daemonUp();
  try { await win.loadURL(up ? `${ORIGIN}/app/${page}` : panelNote("The assistant is not running")); }
  catch { win.failed = true; }
  if (!up) win.failed = true;     // asked again next time
}

// Taller while the assistant's computer is busy: the popover shows its screen live above the threads.
let popover = null;
const POPOVER = {width: 360, height: 420, withScreen: 640};
function sizePopover() {
  if (!popover || popover.isDestroyed()) return;
  const height = trayState.computer ? POPOVER.withScreen : POPOVER.height;
  if (popover.getSize()[1] !== height) popover.setSize(POPOVER.width, height);
}

async function togglePopover() {
  if (!popover || popover.isDestroyed()) popover = panelWindow(POPOVER.width, POPOVER.height);
  if (popover.isVisible()) { popover.hide(); return; }
  if (Date.now() - (popover.hiddenAt || 0) < 300) return;      // the same click took its focus and hid it
  await loadPanel(popover, "tray.html");
  sizePopover();
  placeUnderTray(popover);
  popover.show();
  popover.focus();
  popover.webContents.send("panel-shown");
  if (trayState.unseen.size) { trayState.unseen.clear(); drawTray(); }
}

function placeUnderTray(win) {
  const [width] = win.getSize();
  const icon = tray && !tray.isDestroyed() ? tray.getBounds() : {width: 0};
  const at = icon.width ? {x: icon.x + icon.width / 2, y: icon.y + icon.height} : screen.getCursorScreenPoint();
  const area = screen.getDisplayNearestPoint({x: Math.round(at.x), y: Math.round(at.y)}).workArea;
  const x = Math.min(Math.max(Math.round(at.x - width / 2), area.x + 8), area.x + area.width - width - 8);
  win.setPosition(x, Math.max(Math.round(at.y + 6), area.y + 6));
}

// The quick ask: a 680 x 64 bar, drawn by its page with room around it for its shadow.
const ASK = {bar: 680, height: 64, room: 32};
let asker = null;
async function toggleAsk() {
  if (!asker || asker.isDestroyed()) {
    asker = panelWindow(ASK.bar + 2 * ASK.room, ASK.height + 2 * ASK.room, true);
    if (MAC) asker.setVisibleOnAllWorkspaces(true, {visibleOnFullScreen: true, skipTransformProcessType: true});
  }
  if (asker.isVisible()) { asker.hide(); return; }
  await loadPanel(asker, "ask.html");
  const area = screen.getDisplayNearestPoint(screen.getCursorScreenPoint()).workArea;
  const [width] = asker.getSize();
  asker.setPosition(Math.round(area.x + (area.width - width) / 2), Math.round(area.y + area.height * 0.24 - ASK.room));
  asker.show();
  asker.focus();
  asker.webContents.send("panel-shown");
}

function hidePanels() {
  for (const win of [popover, asker]) if (win && !win.isDestroyed() && win.isVisible()) win.hide();
}

// The app's own settings (userData/settings.json). quickAskShortcut: Alt+Space unless the owner sets another, ""
// for none; written with its default the first time, so there is a line to change.
function settings() {
  const file = path.join(app.getPath("userData"), "settings.json");
  let saved = {};
  try { saved = JSON.parse(fs.readFileSync(file, "utf8")) || {}; }
  catch (error) { if (error.code !== "ENOENT") return {quickAskShortcut: "Alt+Space"}; }   // not JSON: left as it is
  if (!("quickAskShortcut" in saved)) {
    saved = {...saved, quickAskShortcut: "Alt+Space"};
    try { fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, JSON.stringify(saved, null, 2) + "\n"); }
    catch { /* used for this run only */ }
  }
  return saved;
}

// Developer mode (settings.json developerMode, off unless the owner turns it on in the Help menu): the page shows
// the open thread's "Export trace". Kept here so the page knows it when it loads, and told to it when it changes.
let developerThisRun = null;     // what the menu set, when the file could not keep it
function developerMode() { return developerThisRun ?? settings().developerMode === true; }

// One setting written to settings.json beside the others; a file that is not JSON is left as it is.
function saveSetting(key, value) {
  const file = path.join(app.getPath("userData"), "settings.json");
  let saved = {};
  try { saved = JSON.parse(fs.readFileSync(file, "utf8")) || {}; }
  catch (error) { saved = error.code === "ENOENT" ? {} : null; }
  if (!saved) return;
  try {
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, JSON.stringify({...saved, [key]: value}, null, 2) + "\n");
  } catch { /* for this run only */ }
}

function setDeveloperMode(on) {
  developerThisRun = Boolean(on);
  saveSetting("developerMode", Boolean(on));
  for (const win of BrowserWindow.getAllWindows())
    if (!win.isDestroyed() && win.webContents.getURL().startsWith(`${ORIGIN}/app/`)) win.webContents.send("developer-mode", Boolean(on));
}

// A shortcut another app holds, or one Electron does not know, leaves the quick ask without one.
function registerQuickAsk() {
  const shortcut = settings().quickAskShortcut;
  if (typeof shortcut !== "string" || !shortcut) return;
  try { globalShortcut.register(shortcut, () => toggleAsk()); } catch { /* none */ }
}

// One app at a time: opening it again shows the window it already has (it stays in the menu bar on a Mac).
const primary = app.requestSingleInstanceLock();
if (!primary) app.quit();
app.on("second-instance", () => showMain());

app.whenReady().then(() => {
  if (!primary) return;
  // macOS's About window (CodeRunner, About CodeRunner): the product, its maker, the version.
  if (typeof app.setAboutPanelOptions === "function")
    app.setAboutPanelOptions({applicationName: "CodeRunner", applicationVersion: app.getVersion(), version: "",
                              credits: "by InstaVM", copyright: "© 2026 InstaVM"});
  // The admin token rides only on requests to the daemon, websockets included (the computer's screen).
  session.defaultSession.webRequest.onBeforeSendHeaders(
    {urls: [`http://${ADDRESS}/*`, `ws://${ADDRESS}/*`]},
    (details, callback) => {
      if (currentToken()) details.requestHeaders["Authorization"] = `Bearer ${currentToken()}`;
      callback({requestHeaders: details.requestHeaders});
    });
  // The page may show notifications and, for dictation, hear the microphone: sound only, never the camera,
  // and only the daemon's own page. Everything else is refused.
  const own = (url) => { try { return new URL(url).origin === ORIGIN; } catch { return false; } };
  const soundOnly = (details) => {
    const kinds = details?.mediaTypes || (details?.mediaType ? [details.mediaType] : []);
    return kinds.length > 0 && kinds.every((kind) => kind === "audio");
  };
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    const asker = own(details?.requestingUrl || contents.getURL());
    callback(asker && (permission === "notifications" || (permission === "media" && soundOnly(details))));
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission, origin, details) =>
    own(origin || details?.requestingUrl || "") &&
    (permission === "notifications" || (permission === "media" && (!details?.mediaType || details.mediaType === "audio"))));
  // The one thing the page asks of the shell: a folder picked by the owner (where to keep the models). Only the
  // daemon's own page may ask.
  const fromDaemon = (event) => { try { return new URL(event.senderFrame.url).origin === ORIGIN; } catch { return false; } };
  ipcMain.handle("pick-folder", async (event, startOrOpts) => {
    let asker = "";
    try { asker = new URL(event.senderFrame.url).origin; } catch { /* no frame, or no address: refused */ }
    if (asker !== ORIGIN) return null;
    const opts = typeof startOrOpts === "object" && startOrOpts ? startOrOpts : {start: startOrOpts};
    const picked = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), {
      title: typeof opts.title === "string" && opts.title ? opts.title : "Choose a folder",
      buttonLabel: typeof opts.buttonLabel === "string" && opts.buttonLabel ? opts.buttonLabel : "Choose",
      defaultPath: typeof opts.start === "string" && opts.start ? opts.start : undefined,
      properties: ["openDirectory", "createDirectory"],
    });
    return picked.canceled ? null : picked.filePaths[0];
  });
  // The page's "Start it", shown when the assistant has been away for a while (stopped by the owner, say).
  ipcMain.handle("start-assistant", (event) => {
    let asker = "";
    try { asker = new URL(event.senderFrame.url).origin; } catch { /* refused */ }
    return asker === ORIGIN ? startDaemon() : false;
  });
  ipcMain.handle("show-in-finder", (event, target) => {
    if (!fromDaemon(event) || typeof target !== "string" || !target) return false;
    try { shell.showItemInFolder(target); return true; } catch { return false; }
  });
  const ALLOWED_PANES = new Set(["Privacy_Automation", "Privacy_AllFiles", "Privacy_Microphone", "Privacy_Notifications"]);
  ipcMain.handle("open-system-settings", (event, pane) => {
    if (!MAC || !fromDaemon(event) || !ALLOWED_PANES.has(String(pane || ""))) return false;
    shell.openExternal(`x-apple.systempreferences:com.apple.preference.security?${pane}`);
    return true;
  });
  ipcMain.handle("app-settings", (event, update) => {
    if (!fromDaemon(event)) return null;
    const cur = settings();
    if (update && typeof update === "object") {
      const next = {...cur, developerMode: developerMode()};
      if (typeof update.quickAskShortcut === "string") {
        const candidate = update.quickAskShortcut.trim();
        if (MAC) {
          globalShortcut.unregisterAll();
          if (candidate) {
            let registered = false;
            try { registered = Boolean(globalShortcut.register(candidate, () => toggleAsk())); }
            catch { registered = false; }
            if (!registered) {
              if (typeof cur.quickAskShortcut === "string" && cur.quickAskShortcut) {
                try { globalShortcut.register(cur.quickAskShortcut, () => toggleAsk()); } catch { /* best effort */ }
              }
              return {ok: false, error: "Shortcut is already in use or invalid", settings: {...cur, developerMode: developerMode()}};
            }
          }
        }
        next.quickAskShortcut = candidate;
      }
      if (typeof update.developerMode === "boolean") next.developerMode = update.developerMode;
      const file = path.join(app.getPath("userData"), "settings.json");
      try { fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, JSON.stringify(next, null, 2) + "\n"); }
      catch { /* best effort */ }
      if (typeof update.developerMode === "boolean") setDeveloperMode(update.developerMode);
      return {ok: true, settings: next};
    }
    return {ok: true, settings: {...cur, developerMode: developerMode()}};
  });
  // The popover's and the quick ask's: open a thread, or the app; put the panel away; a task was started (the
  // icon looks again at once). The main page's: which thread it shows. Only the daemon's own pages are heard.
  ipcMain.on("open-thread", (event, conversation) => { if (fromDaemon(event)) showMain(String(conversation || "")); });
  ipcMain.on("open-app", (event) => { if (fromDaemon(event)) showMain(); });
  ipcMain.on("open-computer", (event) => { if (fromDaemon(event)) showMain("", true); });
  ipcMain.on("open-settings", (event, section) => { if (fromDaemon(event)) sendToMain("open-settings", typeof section === "string" ? section : ""); });
  ipcMain.on("hide-panel", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (fromDaemon(event) && win && win !== mainWindow) win.hide();
  });
  ipcMain.on("task-started", (event) => { if (fromDaemon(event)) refreshSoon(0); });
  ipcMain.on("thread-open", (event, conversation) => {
    if (!fromDaemon(event) || !mainWindow || mainWindow.isDestroyed() || event.sender !== mainWindow.webContents) return;
    viewed = String(conversation || "");
    seen(viewed);
  });
  // Whether developer mode is on, asked by the page as it loads.
  ipcMain.handle("developer-mode", (event) => fromDaemon(event) && developerMode());
  // The Browser tab, from the main window's page only: where its view goes (in the page's pixels, so zoom counts)
  // and whether it shows; the owner's tab, address and button presses; what it shows now.
  const mainPage = (event) => fromDaemon(event) && mainWindow && !mainWindow.isDestroyed() && event.sender === mainWindow.webContents;
  ipcMain.on("browser-view", (event, bounds, shown) => {
    if (!fromDaemon(event) || !mainPage(event) || !appBrowser) return;
    const zoom = mainWindow.webContents.getZoomFactor(), box = bounds && typeof bounds === "object" ? bounds : {};
    appBrowser.place({x: box.x * zoom, y: box.y * zoom, width: box.width * zoom, height: box.height * zoom}, shown === true);
  });
  ipcMain.handle("browser-act", (event, request) => (mainPage(event) && appBrowser ? appBrowser.act(request) : null));
  ipcMain.handle("browser-state", (event) => (mainPage(event) && appBrowser ? appBrowser.state() : null));
  // Keep cookies (the Browser tab's menu): settings.json browserKeepCookies, "always" unless set to "quit".
  let keepThisRun = null;
  appBrowser = new AppBrowser({assistantDir: ASSISTANT, daemonOrigin: ORIGIN,
                               keep: () => keepThisRun ?? (settings().browserKeepCookies === "quit" ? "quit" : "always"),
                               setKeep: (mode) => { keepThisRun = mode; saveSetting("browserKeepCookies", mode); }});
  const toPage = (channel, ...args) => {
    const contents = mainWindow && !mainWindow.isDestroyed() ? mainWindow.webContents : null;
    if (contents && contents.getURL().startsWith(`${ORIGIN}/app/`)) contents.send(channel, ...args);
  };
  appBrowser.onState = (state) => toPage("browser-state", state);
  appBrowser.onFront = () => toPage("browser-front");
  appBrowser.connect();
  buildMenu();
  mainWindow = createWindow();
  watchDaemon();
  if (!HIDDEN) createTray();
  if (MAC && !HIDDEN) registerQuickAsk();
  followEvents();
  refreshSoon(0);
  nativeTheme.on("updated", () => { trayState.image = ""; drawTray(); });
  app.on("activate", (_, visible) => { if (!mainWindow || mainWindow.isDestroyed() || !visible) showMain(); });
});

app.on("will-quit", () => globalShortcut.unregisterAll());
app.on("will-quit", () => appBrowser?.stop());
// Keep cookies "until quit": the Browser tab's cookies go as the app quits (and, should that not finish, when it
// next starts, before any page loads).
let cookiesCleared = false;
app.on("before-quit", (event) => {
  if (cookiesCleared || !appBrowser || appBrowser.keep() !== "quit") return;
  event.preventDefault();
  cookiesCleared = true;
  Promise.race([appBrowser.clearCookies(), new Promise((resolve) => setTimeout(resolve, 2000))]).finally(() => app.quit());
});
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });

// For driving the app from a test (node --inspect): the same actions as the icon's click and the shortcut.
module.exports = {togglePopover, toggleAsk, showMain, refreshTray, trayState, browser: () => appBrowser,
                  mainWindow: () => mainWindow};
