// The menu bar's popover in the desktop app: while a task uses the assistant's computer, its screen live (view
// only; the expand button opens it as large as the main window); a quick ask; then the latest threads (the ones
// that need the owner first, then those running, then the most recent), each with one live line. A thread opens
// in the main window; nothing is approved here. Every word from a task goes in as text.
import RFB from "./novnc/core/rfb.js";
import {el} from "./render.js";

const $ = (id) => document.getElementById(id);
const shell = window.coderunnerShell || {};
const HEADERS = {"Content-Type": "application/json", "X-CodeRunner": "1"};
// The system's material shows through on a Mac; elsewhere the window is plain.
if (!window.coderunnerShell || !/Mac/i.test(navigator.platform || "")) document.documentElement.classList.add("opaque");

function icon(d) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), line = document.createElementNS(svg.namespaceURI, "path");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  line.setAttribute("d", d);
  svg.appendChild(line);
  return svg;
}
$("traySend").appendChild(icon("M12 19V5 M5.5 11.5L12 5l6.5 6.5"));
$("liveExpand").appendChild(icon("M14 4h6v6 M10 20H4v-6 M20 4l-6.5 6.5 M4 20l6.5-6.5"));
$("trayInput").addEventListener("input", () => { $("traySend").disabled = !$("trayInput").value.trim(); });

// The computer's screen: connected only while the popover is shown and a task uses the computer, through the
// same relay as the app's (/api/computer/screen), never to type or click.
const live = {rfb: null, want: false, timer: null};
async function connectLive() {
  clearTimeout(live.timer);
  if (!live.want || live.rfb) return;
  let info = null;
  try { info = await api("/api/computer"); } catch { /* the daemon is restarting */ }
  if (!live.want || live.rfb) return;
  if (!info?.available || !info?.running) { live.timer = setTimeout(connectLive, 4000); return; }
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/computer/screen`;
  const rfb = live.rfb = new RFB($("liveScreen"), url, {credentials: {password: info.password}, wsProtocols: ["binary"], shared: true});
  rfb.viewOnly = true;
  rfb.focusOnClick = false;
  rfb.scaleViewport = true;
  rfb.resizeSession = false;
  rfb.background = "#0d0d0c";
  rfb.addEventListener("credentialsrequired", () => rfb.sendCredentials({password: info.password}));
  rfb.addEventListener("disconnect", () => {
    if (live.rfb === rfb) live.rfb = null;
    if (live.want) live.timer = setTimeout(connectLive, 3000);
  });
}
function showLive(busy) {
  $("live").hidden = !busy;
  live.want = busy && !document.hidden;
  if (live.want) { connectLive(); return; }
  clearTimeout(live.timer);
  const rfb = live.rfb;
  live.rfb = null;
  rfb?.disconnect();
}
$("liveExpand").onclick = () => shell.openComputer?.();

async function api(path, options = {}) {
  const response = await fetch(path, {credentials: "same-origin", headers: HEADERS, ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "That did not work. Try again.");
  return data;
}

const pad = (n) => String(n).padStart(2, "0");
function clock(seconds) {
  seconds = Math.max(0, Math.floor(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
function ago(seconds) {
  const gone = Date.now() / 1000 - seconds;
  if (gone < 60) return "just now";
  if (gone < 3600) return `${Math.floor(gone / 60)} min ago`;
  if (gone < 86400) return `${Math.floor(gone / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString([], {month: "short", day: "numeric"});
}

// The thread's dot and its one line.
function status(thread) {
  if (thread.waiting) return ["waiting", "Needs you"];
  if (thread.status === "running") return ["running", thread.line || "Working…"];
  if (thread.status === "queued") return ["queued", "Queued"];
  if (thread.status === "failed") return ["failed", "Failed"];
  if (thread.status === "cancelled") return ["cancelled", "Stopped"];
  return ["done", thread.finished ? `Done ${ago(thread.finished)}` : "Done"];
}

let threads = [];
function draw() {
  const list = $("threads"), focused = document.activeElement?.dataset?.conversation;
  list.replaceChildren();
  if (!threads.length) { list.appendChild(el("div", "tray-empty muted", "No tasks yet")); return; }
  for (const thread of threads.slice(0, 5)) {
    const [kind, line] = status(thread);
    const row = el("button", "tray-row"), words = el("span", "tray-words");
    row.type = "button";
    row.dataset.conversation = thread.conversation;
    words.append(el("span", "tray-name", thread.title || "Attached files"), el("span", `tray-line ${kind}`, line));
    row.append(el("span", `dot ${kind}`), words);
    if (kind === "running" && thread.started) {
      const time = el("span", "tray-time", clock(Date.now() / 1000 - thread.started));
      time.dataset.started = thread.started;
      row.appendChild(time);
    }
    row.title = thread.title || "";
    row.onclick = () => shell.openThread?.(thread.conversation);
    list.appendChild(row);
    if (focused === thread.conversation) row.focus();
  }
}

async function refresh() {
  let data;
  try { data = await api("/api/tray"); } catch { return; }      // the daemon is restarting: what is shown stays
  threads = data.threads || [];
  draw();
  showLive(Boolean(data.computer));
}

let noteTimer = null;
function note(text) {
  $("trayNote").textContent = text;
  $("trayNote").hidden = !text;
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => { $("trayNote").hidden = true; }, 6000);
}

// Return starts the task in a new thread; the box clears for the next one.
$("trayAsk").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("trayInput"), message = input.value.trim();
  if (!message || input.disabled) return;
  input.disabled = true;
  try {
    await api("/api/tasks", {method: "POST", body: JSON.stringify({message, conversation: "", attachments: []})});
    input.value = "";
    $("traySend").disabled = true;
    note("");
    shell.taskStarted?.();
    refresh();
  } catch (err) { note(err.message); }
  finally { input.disabled = false; input.focus(); }
};

$("openApp").onclick = () => (shell.openApp ? shell.openApp() : location.assign("/app/"));
$("openSettings").onclick = () => (shell.openSettings ? shell.openSettings("models") : location.assign("/app/"));
document.addEventListener("keydown", (event) => { if (event.key === "Escape") shell.hidePanel?.(); });
shell.onShown?.(() => { refresh(); $("trayInput").focus(); });
// Put away: the screen's connection closes with it.
document.addEventListener("visibilitychange", () => { if (document.hidden) showLive(false); else refresh(); });

setInterval(() => { if (!document.hidden) refresh(); }, 2000);
setInterval(() => {
  for (const time of document.querySelectorAll(".tray-time")) time.textContent = clock(Date.now() / 1000 - Number(time.dataset.started));
}, 1000);
refresh();
