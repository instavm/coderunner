// CodeRunner app: tasks run in the background on the daemon; this page shows what each one does as it happens,
// and the sandbox computer's screen, which the owner can watch or take over.
import RFB from "./novnc/core/rfb.js";

const $ = (id) => document.getElementById(id);
const HEADERS = {"Content-Type": "application/json", "X-CodeRunner": "1"};

async function api(path, options = {}) {
  const response = await fetch(path, {credentials: "same-origin", headers: HEADERS, ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

// Text with its web addresses as links (never HTML from the model or a page).
function richText(node, text) {
  const parts = String(text).split(/(https?:\/\/[^\s<>()]+)/g);
  for (const part of parts) {
    if (/^https?:\/\//.test(part)) {
      const a = el("a", "", part); a.href = part; a.target = "_blank"; a.rel = "noopener noreferrer";
      node.appendChild(a);
    } else if (part) node.appendChild(document.createTextNode(part));
  }
  return node;
}

// A small, safe subset of Markdown (paragraphs, lists, **bold**, `code`, links), built as DOM nodes.
function inline(node, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  for (const part of String(text).split(pattern)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) node.appendChild(el("strong", "", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) node.appendChild(el("code", "", part.slice(1, -1)));
    else if (part.startsWith("[")) {
      const m = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      const a = el("a", "", m[1]); a.href = m[2]; a.target = "_blank"; a.rel = "noopener noreferrer";
      node.appendChild(a);
    } else richText(node, part);
  }
  return node;
}
function markdown(node, text) {
  node.classList.add("md");
  let list = null, para = null;
  for (const raw of String(text ?? "").split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/), numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const kind = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== kind) {
        list = el(kind); if (numbered) list.start = Number(numbered[1]); node.appendChild(list);
      }
      inline(list.appendChild(el("li")), bullet ? bullet[1] : numbered[2]);
      para = null;
    } else if (!line.trim()) { list = null; para = null; }
    else if (/^#{1,4}\s+/.test(line)) { list = null; para = null; inline(node.appendChild(el("div", "md-h")), line.replace(/^#+\s+/, "")); }
    else {
      list = null;
      if (para) para.appendChild(document.createElement("br")); else para = node.appendChild(el("p"));
      inline(para, line);
    }
  }
  return node;
}

// Tool results wrap page content in <untrusted> tags for the model; the owner sees the words only.
const plain = (s) => String(s ?? "").replace(/<\/?untrusted[^>]*>/g, "").trim();
const trunc = (s, n) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
const host = (u) => { try { return new URL(u).host.replace(/^www\./, ""); } catch { return trunc(u, 60); } };
const base = (p) => String(p ?? "").split("/").pop();
const clock = (seconds) => {
  seconds = Math.max(0, Math.round(seconds));
  const m = Math.floor(seconds / 60), s = seconds % 60;
  return m ? `${m}:${String(s).padStart(2, "0")}` : `${s}s`;
};
function args(raw) {
  if (raw && typeof raw === "object") return raw;
  try { return JSON.parse(raw || "{}"); } catch { return {}; }
}

function describeTool(name, raw) {
  const a = args(raw);
  switch (name) {
    case "browser_open": case "browser_navigate": return `Opening ${host(a.url)}`;
    case "browser_snapshot": return "Reading the page";
    case "browser_list": return "Reading the results on the page";
    case "browser_click": return "Clicking";
    case "browser_type": return a.text ? `Typing “${trunc(a.text, 40)}”` : "Typing";
    case "browser_batch": return "Filling in the page";
    case "browser_find": return `Looking for “${trunc(a.text || a.query, 40)}” on the page`;
    case "browser_select": return `Choosing “${trunc(a.option, 40)}”`;
    case "browser_press": return `Pressing ${a.key}`;
    case "browser_scroll": return "Scrolling";
    case "browser_back": return "Going back";
    case "browser_tabs": return "Checking the open tabs";
    case "browser_screenshot": return "Looking at the screen";
    case "browser_handoff": return "Asking you to take over";
    case "browser_task": return `Working in the browser: ${trunc(a.goal || a.task || "", 90)}`;
    case "run_shell": return `Running ${trunc(a.command, 70)}`;
    case "shell_job": return "Checking a running job";
    case "run_python": return "Running Python";
    case "files_search": return `Searching your files for “${trunc(a.query, 50)}”`;
    case "file_read": return `Reading ${base(a.path)}`;
    case "file_view": return `Looking at ${base(a.path)}`;
    case "file_listen": return `Listening to ${base(a.path)}`;
    case "memory_search": return "Checking what I remember";
    case "memory_save": return "Remembering this";
    case "plan_update": return "Updating the plan";
    case "schedule_create": return "Scheduling it";
    case "send_message": return "Sending a message";
    default: return name.replace(/_/g, " ");
  }
}

function describeActivity(e) {
  const ok = e.ok !== false;
  switch (e.event) {
    case "browser.started": return "Started the computer's browser";
    case "browser.open": case "browser.navigate": return ok ? `Opened ${host(e.url || e.page)}` : null;
    case "browser.click": return e.label ? `Clicked ${trunc(e.label, 80)}` : "Clicked";
    case "browser.type": return e.label ? `Typed into ${trunc(e.label, 80)}` : "Typed";
    case "browser.select": return `Chose “${trunc(e.option, 60)}”`;
    case "browser.press": return `Pressed ${e.key}`;
    case "browser.click_at": return "Clicked on the screen";
    case "browser.type_focused": return "Typed";
    case "browser.scroll": case "browser.wheel": return "Scrolled";
    case "driver.start": return `The browser model (${e.model}) is working`;
    case "driver.finish": return "The browser model finished";
    default: return null;
  }
}

// ---- state ------------------------------------------------------------------

const state = {
  conversation: null,           // "app:…" or null for a new task
  views: new Map(),             // task id -> TaskView (current conversation only)
  approvals: new Map(),         // approval id -> {conversation, view}
  pending: new Set(),           // approval ids still waiting
  running: new Map(),           // task id -> {conversation, label}
  lastSeq: 0,
};

// ---- a task in the thread ----------------------------------------------------

class TaskView {
  constructor(task) {
    this.id = task.id;
    this.started = task.created || Date.now() / 1000;
    this.finished = task.finished || null;
    this.status = task.status || "running";
    this.seq = 0;
    this.root = el("article", "task");
    this.root.appendChild(el("div", "bubble-user", task.prompt));
    this.flow = el("div", "flow");
    this.root.appendChild(this.flow);
    this.segment = null;          // the open group of steps
    this.steps = new Map();       // tool call id -> step row
    this.current = null;          // the step activity is attached to
    $("thread").appendChild(this.root);
  }

  workSegment() {
    if (this.segment) return this.segment;
    const box = el("div", "work" + (this.status === "running" ? " open" : ""));
    const head = el("div", "work-head");
    const chev = el("span", "chev", "›");
    const what = el("span", "what", "Working");
    const time = el("span", "time", "");
    head.append(chev, what, time);
    const steps = el("div", "steps");
    box.append(head, steps);
    head.onclick = () => box.classList.toggle("open");
    this.flow.appendChild(box);
    this.segment = {box, what, time, steps, count: 0, began: null};
    if (this.status === "running") this.addStop(head);
    return this.segment;
  }

  addStop(head) {
    const stop = el("button", "stop", "Stop");
    stop.onclick = async (event) => {
      event.stopPropagation();
      stop.disabled = true;
      await api(`/api/tasks/${this.id}/cancel`, {method: "POST", body: "{}"}).catch(() => {});
    };
    head.appendChild(stop);
    this.stopButton = stop;
  }

  closeSegment() {
    if (!this.segment) return;
    this.segment.box.classList.remove("open");
    this.segment.box.querySelector(".stop")?.remove();
    this.segment = null;
    this.current = null;
  }

  addStep(text, state, key, ts) {
    const seg = this.workSegment();
    seg.began ??= ts;
    const row = el("div", "step " + state);
    row.append(el("span", "ic", state === "run" ? "•" : state === "ok" ? "✓" : state === "err" ? "!" : "·"),
               el("div", "txt", text));
    seg.steps.appendChild(row);
    seg.count += 1;
    seg.what.textContent = text;
    seg.last = ts;
    if (key) this.steps.set(key, row);
    seg.steps.scrollTop = seg.steps.scrollHeight;
    return row;
  }

  markStep(row, ok, detail) {
    row.className = "step " + (ok ? "ok" : "err");
    row.querySelector(".ic").textContent = ok ? "✓" : "!";
    if (detail) row.querySelector(".txt").appendChild(el("div", "sub", detail));
  }

  apply(e) {
    if (e.seq && e.seq <= this.seq) return;
    if (e.seq) this.seq = e.seq;
    switch (e.type) {
      case "started": break;
      case "note":
        this.closeSegment();
        this.flow.appendChild(markdown(el("div", "note"), e.text));
        break;
      case "plan":
        break;
      case "tool": {
        if (e.name === "plan_update") break;
        this.current = this.addStep(describeTool(e.name, e.arguments), "run", e.id, e.ts);
        if (this.status === "running") setNow(this.current.querySelector(".txt").textContent);
        break;
      }
      case "tool_done": {
        const row = this.steps.get(e.id);
        if (row) this.markStep(row, e.ok, e.ok ? "" : trunc(plain(e.preview), 200));
        break;
      }
      case "activity": this.activity(e); break;
      case "done": case "failed": case "cancelled": this.finish(e); break;
    }
    this.tick();
  }

  activity(e) {
    if (e.event === "approval.requested") { this.ask(e); return; }
    const text = describeActivity(e);
    if (!text) return;
    const step = this.current && this.current.classList.contains("run") ? this.current : null;
    if (step && e.ok !== false && ["browser.click", "browser.type", "browser.select"].includes(e.event) &&
        !step.dataset.named) {
      // The step says what it did ("Clicked Add to Cart") rather than "Clicking" with the detail below.
      step.dataset.named = "1";
      step.querySelector(".txt").firstChild.textContent = text;
      if (this.segment) this.segment.what.textContent = text;
    } else if (step) {
      step.querySelector(".txt").appendChild(el("div", "sub", text + (e.ok === false && e.error ? ` — ${trunc(e.error, 120)}` : "")));
    } else {
      this.addStep(text, e.ok === false ? "err" : "ok", null, e.ts);
    }
    if (this.status === "running") setNow(text);
  }

  ask(e) {
    this.closeSegment();
    const handoff = e.kind === "browser.handoff";
    const card = el("div", "ask");
    card.appendChild(el("div", "ask-title", handoff ? "Your turn on the computer" : "Needs your approval"));
    card.appendChild(richText(el("div", ""), e.summary || e.kind));
    const row = el("div", "row");
    const decide = async (allow) => {
      row.querySelectorAll("button").forEach((b) => b.disabled = true);
      try { await api(`/api/approvals/${e.id}`, {method: "POST", body: JSON.stringify({allow, scope: "once"})}); }
      catch (err) { row.querySelectorAll("button").forEach((b) => b.disabled = false); alert(err.message); return; }
      closeAsk(e.id, allow);
    };
    if (handoff) {
      const take = el("button", "btn primary", "Take control");
      take.onclick = () => { showComputer(true); setControl(true); setExpanded(true); };
      const done = el("button", "btn", "I'm done");
      done.onclick = () => { setControl(false); decide(true); };
      const skip = el("button", "btn danger", "Skip");
      skip.onclick = () => decide(false);
      row.append(take, done, skip);
      showComputer(true);
    } else {
      const allow = el("button", "btn primary", "Allow");
      allow.onclick = () => decide(true);
      const deny = el("button", "btn danger", "Deny");
      deny.onclick = () => decide(false);
      row.append(allow, deny);
    }
    card.appendChild(row);
    this.flow.appendChild(card);
    state.approvals.set(e.id, {conversation: state.conversation, card, row});
    if (!state.pending.has(e.id) && e.ts && Date.now() / 1000 - e.ts > 5) closeAsk(e.id, null);
    card.scrollIntoView({block: "nearest", behavior: "smooth"});
  }

  finish(e) {
    this.status = e.type;
    this.finished = e.ts;
    this.closeSegment();
    const answer = el("div", "answer" + (e.type === "failed" ? " failed" : ""));
    markdown(answer, e.reply || "");
    this.flow.appendChild(answer);
    this.stopButton?.remove();
    state.running.delete(this.id);
    refreshNow();
  }

  tick() {
    const seg = this.segment;
    if (!seg) return;
    const end = this.status === "running" ? Date.now() / 1000 : (this.finished || seg.last);
    seg.time.textContent = `${seg.count} step${seg.count === 1 ? "" : "s"} · ${clock(end - (seg.began || this.started))}`;
  }
}

function closeAsk(id, allowed) {
  const entry = state.approvals.get(id);
  state.pending.delete(id);
  if (!entry || entry.card.classList.contains("closed")) return;
  entry.card.classList.add("closed");
  entry.row.replaceChildren(el("span", "muted small", allowed === null ? "Answered" : allowed ? "Done" : "Declined"));
}

// ---- conversations ----------------------------------------------------------

async function loadConversations() {
  const list = await api("/api/conversations").catch(() => []);
  const nav = $("conversations");
  nav.replaceChildren();
  for (const c of list) {
    const button = el("button", "conv" + (c.conversation === state.conversation ? " on" : ""));
    const t = el("div", "t");
    const waiting = [...state.approvals.values()].some((a) => a.conversation === c.conversation &&
                                                       !a.card.classList.contains("closed"));
    const status = waiting ? "waiting" : c.status;
    t.append(el("span", "dot " + status), el("span", "label", trunc(c.title, 60)));
    const sub = status === "running" ? "Working…" : status === "waiting" ? "Needs you" :
                status === "failed" ? "Failed" : status === "cancelled" ? "Stopped" :
                new Date(c.updated * 1000).toLocaleString([], {dateStyle: "medium", timeStyle: "short"});
    button.append(t, el("div", "s", sub));
    button.onclick = () => openConversation(c.conversation, c.title);
    nav.appendChild(button);
  }
}

async function openConversation(id, title) {
  if (state.setup) return;        // the first-run screen holds the page until a model is set up
  state.conversation = id;
  state.views.clear();
  $("thread").replaceChildren();
  $("title").textContent = id ? trunc(title || "Task", 90) : "New task";
  if (!id) { $("thread").appendChild(emptyState); loadConversations(); $("input").focus(); return; }
  const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
  for (const task of data.tasks) state.views.set(task.id, new TaskView(task));
  for (const e of data.events) state.views.get(e.task)?.apply(e);
  loadConversations();
  scrollDown();
}

function scrollDown() { const t = $("thread"); t.scrollTop = t.scrollHeight; }

// ---- live events ---------------------------------------------------------------

function listen() {
  const source = new EventSource(`/api/events?since=${state.lastSeq || -1}`);
  source.onmessage = (message) => {
    const e = JSON.parse(message.data);
    state.lastSeq = Math.max(state.lastSeq, e.seq || 0);
    if (e.type === "started") state.running.set(e.task, {conversation: e.conversation, label: "Starting"});
    if (["done", "failed", "cancelled"].includes(e.type)) state.running.delete(e.task);
    if (e.conversation === state.conversation) {
      let view = state.views.get(e.task);
      if (!view && e.type === "started") {
        view = new TaskView({id: e.task, prompt: e.prompt, created: e.ts, status: "running"});
        state.views.set(e.task, view);
      }
      const nearBottom = $("thread").scrollHeight - $("thread").scrollTop - $("thread").clientHeight < 160;
      view?.apply(e);
      if (nearBottom) scrollDown();
    } else if (e.type === "activity" && e.event === "approval.requested") {
      notify("CodeRunner needs you", e.summary || "A task is waiting for you");
    }
    if (["started", "done", "failed", "cancelled"].includes(e.type) ||
        (e.type === "activity" && e.event === "approval.requested")) {
      if (e.type !== "started" && e.conversation !== state.conversation) notify("Task finished", trunc(e.reply, 120));
      loadConversations();
    }
    refreshNow();
  };
  source.onerror = () => { source.close(); setTimeout(listen, 2000); };
}

function notify(title, body) {
  if (document.hasFocus() || !("Notification" in window)) return;
  if (Notification.permission === "granted") new Notification(title, {body});
}

async function pollApprovals() {
  try {
    const items = await api("/api/approvals");
    const ids = new Set(items.map((a) => a.id));
    for (const id of state.approvals.keys()) if (!ids.has(id)) closeAsk(id, null);
    state.pending = ids;
  } catch { /* the daemon restarts now and then */ }
}

// ---- composer ------------------------------------------------------------------

const emptyState = $("empty");
$("composer").onsubmit = async (event) => {
  event.preventDefault();
  const message = $("input").value.trim();
  if (!message) return;
  $("send").disabled = true;
  try {
    const task = await api("/api/tasks", {method: "POST",
      body: JSON.stringify({message, conversation: state.conversation || ""})});
    $("input").value = ""; grow();
    if (task.conversation !== state.conversation) {
      await openConversation(task.conversation, message);
    }
  } catch (err) { alert(err.message); }
  finally { $("send").disabled = false; }
};
function grow() { const i = $("input"); i.style.height = "auto"; i.style.height = Math.min(i.scrollHeight, 200) + "px"; }
$("input").addEventListener("input", grow);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
});
document.querySelectorAll(".example").forEach((b) => b.onclick = () => { $("input").value = b.textContent; grow(); $("input").focus(); });
$("newTask").onclick = () => openConversation(null);
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") { e.preventDefault(); openConversation(null); }
  if (e.key === "Escape" && $("shell").classList.contains("expanded") && !control) setExpanded(false);
});

// ---- the computer ----------------------------------------------------------------

let rfb = null;
let control = false;
let retry = null;

function setNote(text) { $("screenNote").textContent = text; }

async function connectScreen() {
  clearTimeout(retry);
  let info;
  try { info = await api("/api/computer", {method: "POST", body: "{}"}); }
  catch (err) { setNote(`The computer is not reachable: ${err.message}`); retry = setTimeout(connectScreen, 5000); return; }
  if (!info.available) {
    setNote(info.error || "This computer has no screen yet.");
    $("computerState").textContent = "No screen";
    watchComputerSetup();           // offers to set it up, or shows a setup that is running
    return;
  }
  if (computerBox) computerBox.box.hidden = true;
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/computer/screen`;
  rfb = new RFB($("screen"), url, {credentials: {password: info.password}, wsProtocols: ["binary"], shared: true});
  rfb.scaleViewport = true;
  rfb.resizeSession = false;
  rfb.background = "#0d0d0c";
  rfb.viewOnly = !control;
  rfb.focusOnClick = control;
  rfb.addEventListener("connect", () => { setNote(""); refreshNow(); });
  rfb.addEventListener("credentialsrequired", () => rfb.sendCredentials({password: info.password}));
  rfb.addEventListener("disconnect", () => {
    rfb = null;
    setNote("Reconnecting to the screen…");
    $("computerState").textContent = "Reconnecting";
    retry = setTimeout(connectScreen, 2500);
  });
}

// The sandbox has no desktop yet: `coderunner browser computer` from here, with its progress. The daemon
// starts again when it is done, and the screen connects.
let computerTimer = null;
let computerBox = null;         // made here, not in the page: a browser may still hold an older index.html
function computerSetupBox() {
  if (computerBox) return computerBox;
  const box = el("div", "panel-section"), button = el("button", "btn primary", "Set up the computer");
  box.id = "computerSetupBox";
  button.type = "button";
  button.onclick = async () => {
    button.disabled = true;
    await api("/api/computer/setup", {method: "POST", body: "{}"}).catch(() => {});      // 409: one is running
    watchComputerSetup();
  };
  box.append(button, el("div", "muted small setup-line"));
  $("screenWrap").after(box);
  return computerBox = {box, button, line: box.lastChild};
}

async function watchComputerSetup() {
  clearTimeout(computerTimer);
  let job;
  try { job = await api("/api/computer/setup"); }
  catch { computerTimer = setTimeout(watchComputerSetup, 2500); return; }
  const {box, button, line} = computerSetupBox();
  box.hidden = false;
  const last = job.lines[job.lines.length - 1] || "";
  if (job.state === "running") {
    button.disabled = true;
    button.textContent = "Setting up the computer…";
    line.textContent = last || "Installing a desktop and Chrome in the sandbox; this takes a few minutes.";
    computerTimer = setTimeout(watchComputerSetup, 2500);
  } else if (job.state === "done") {
    button.disabled = true;
    button.textContent = "The computer is ready";
    line.textContent = "Starting the assistant with it…";
    retry = setTimeout(connectScreen, 8000);
  } else {
    button.disabled = false;
    button.textContent = job.state === "failed" ? "Try again" : "Set up the computer";
    line.textContent = job.state === "failed" ? last : "A desktop and Chrome inside the sandbox, for the assistant to work on while you watch.";
  }
}
function setControl(on) {
  control = on;
  $("watching").classList.toggle("on", !on);
  $("takeControl").classList.toggle("on", on);
  $("screenWrap").classList.toggle("control", on);
  if (rfb) { rfb.viewOnly = !on; rfb.focusOnClick = on; if (on) rfb.focus(); }
  refreshNow();
}
function setExpanded(on) {
  $("shell").classList.toggle("expanded", on);
  $("expand").textContent = on ? "⤡" : "⤢";
  $("expand").title = on ? "Shrink" : "Expand";
}
function showComputer(on) { $("shell").classList.toggle("no-computer", !on); }

$("watching").onclick = () => setControl(false);
$("takeControl").onclick = () => setControl(true);
$("expand").onclick = () => setExpanded(!$("shell").classList.contains("expanded"));
$("closeComputer").onclick = () => { setExpanded(false); showComputer(false); };
$("toggleComputer").onclick = () => showComputer($("shell").classList.contains("no-computer"));

function setNow(text) { state.nowText = text; refreshNow(); }
function refreshNow() {
  const busy = state.running.size > 0;
  $("computerState").textContent = control ? "You're in control" : busy ? "Working" : rfb ? "Idle" : "Connecting";
  $("nowLine").textContent = busy ? (state.nowText || "Working…") : "Idle";
}

async function loadModels() {
  try {
    const status = await api("/api/status");
    const models = status.models || {};
    const rows = [["Planning", models.chat || status.model?.model], ["Browser", models.driver],
                  ["Decisions", models.decision], ["Files", models.embedding || status.model?.embedding],
                  ["Voice", (status.voice || []).join(", ")], ["Browser runs on", status.browser?.mode === "computer" ? "this computer's VM" : status.browser?.mode]];
    const dl = $("models");
    dl.replaceChildren();
    for (const [k, v] of rows) if (v) dl.append(el("dt", "", k), el("dd", "", v));
    $("modelsLine").textContent = models.chat ? `Local · ${models.chat}` : "";
  } catch { /* shown once the daemon answers */ }
}

// ---- first run: no model yet ----------------------------------------------------------
// What `coderunner setup` does in a terminal, from here: the plan for this computer and what it downloads, then
// setup itself with its progress. The daemon starts again with the model when setup is done.

const gb = (n) => n == null ? "size not known" : n < 0.1 ? `${Math.round(n * 1000)} MB` : `${n >= 10 ? n.toFixed(0) : n.toFixed(1)} GB`;
let setupTimer = null;
let setupDone = 0;

function setupCard(title) {
  state.setup = true;
  const card = el("section", "setup");
  card.appendChild(el("h1", "", title));
  $("thread").replaceChildren(card);
  $("composer").hidden = true;
  $("title").textContent = "Set up";
  showComputer(false);            // nothing runs on the computer until there is a model
  return card;
}

function setupButton(label, action) {
  const button = el("button", "btn primary setup-go", label);
  button.onclick = () => { button.disabled = true; action(); };
  return button;
}

async function showPlan() {
  const card = setupCard("Set up your assistant");
  card.appendChild(el("p", "muted", "Looking at this computer and the models that suit it. The first look of a day takes about half a minute…"));
  try { renderPlan(await api("/api/setup", {method: "POST", body: JSON.stringify({action: "plan"})})); }
  catch (err) { card.append(el("p", "setup-error", err.message), setupButton("Look again", showPlan)); }
}

function renderPlan(plan) {
  const card = setupCard("Set up your assistant");
  const hw = plan.hardware;
  card.appendChild(el("p", "muted", `This computer: ${hw.chip}, ${hw.ram_gb} GB of memory${hw.gpu === "none" ? ", no graphics processor found" : ""}. ` +
                                    "The assistant's models run on it; nothing you ask is sent anywhere else."));
  if (!plan.model) {
    card.append(el("p", "setup-error", plan.notes.join(" ") || "No model fits this computer."), setupButton("Look again", showPlan));
    return;
  }
  const pick = el("div", "setup-box"), model = plan.model;
  // In a sentence first; the planner's own reasons stay a click away.
  const facts = [model.need_gb ? `uses ${model.need_gb.toFixed(1)} of this computer's ${hw.ram_gb} GB of memory` : "",
                 model.step_s ? `about ${Math.round(model.step_s)} seconds for each step of a task` : "",
                 model.vision ? "reads pictures and screenshots" : "text only"].filter(Boolean).join(" · ");
  const why = el("details", "setup-why");
  why.append(el("summary", "", "Why this one"), el("div", "muted small", model.reasons.join(" · ")));
  pick.append(el("div", "section-title", "The model for this computer"), el("div", "setup-model", model.id),
              el("div", "muted small", facts.charAt(0).toUpperCase() + facts.slice(1)), why);
  const list = el("div", "setup-box");
  list.appendChild(el("div", "section-title", "To download"));
  for (const item of plan.downloads) {
    const row = el("div", "setup-row");
    row.append(el("span", "", item.what), el("span", "muted", gb(item.gb)));
    list.appendChild(row);
  }
  const total = el("div", "setup-row total");
  total.append(el("span", "", plan.downloads.length ? "In all" : "Nothing: everything it needs is already here"),
               el("span", "", `${plan.downloads.length ? gb(plan.download_gb) + " · " : ""}${gb(plan.free_gb)} free on the disk`));
  list.appendChild(total);
  card.append(pick, list);
  for (const note of plan.notes) card.appendChild(el("p", "setup-note", note));
  card.appendChild(setupButton(plan.downloads.length ? "Download and set up" : "Set up", async () => {
    await api("/api/setup", {method: "POST", body: JSON.stringify({action: "start"})}).catch(() => {});   // 409: one is running
    watchSetup();
  }));
}

function renderProgress(job) {
  const title = job.state === "running" ? "Setting up…" : job.state === "done" ? "Ready" : "Setup did not finish";
  const card = setupCard(title);
  if (job.state === "running") {
    card.appendChild(el("p", "muted", "Downloading the model and testing it on this computer. It carries on if you close this window."));
  }
  const log = el("div", "setup-log");
  for (const line of job.lines) log.appendChild(el("div", "", line));
  card.appendChild(log);
  log.scrollTop = log.scrollHeight;
  if (job.state === "failed") {
    card.append(el("p", "setup-error", job.result?.error || "Setup failed."), setupButton("Look again", showPlan));
  } else if (job.state === "done") {
    card.appendChild(el("p", "muted", "Starting the assistant with its model…"));
  }
}

async function watchSetup() {
  clearTimeout(setupTimer);
  let job;
  try { job = await api("/api/setup"); }
  catch { setupTimer = setTimeout(watchSetup, 1500); return; }      // the daemon is starting again
  if (job.state === "failed") { renderProgress(job); return; }
  if (job.state === "running" || (job.state === "done" && setupDone++ < 40)) {
    renderProgress(job);
    setupTimer = setTimeout(watchSetup, 1500);
    return;
  }
  location.reload();              // set up, and the daemon that answers now was started with the model
}

// True when the page shows the first-run screen instead of tasks.
async function checkSetup() {
  let job;
  try { job = await api("/api/setup"); } catch { return false; }
  if (job.state === "running") { renderProgress(job); watchSetup(); return true; }
  if (job.configured) return false;
  if (job.state === "failed") renderProgress(job);
  else if (job.plan) renderPlan(job.plan);
  else showPlan();
  return true;
}

// ---- start -------------------------------------------------------------------------

setInterval(() => state.views.forEach((v) => v.tick()), 1000);
setInterval(pollApprovals, 2500);
if ("Notification" in window && Notification.permission === "default") Notification.requestPermission().catch(() => {});
pollApprovals();
loadModels();
checkSetup().then((firstRun) => { if (!firstRun) openConversation(null); });
listen();
connectScreen();
