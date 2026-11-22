// CodeRunner app: tasks run in the background on the daemon; this page shows what each one does as it happens,
// and the sandbox computer's screen, which the owner can watch or take over.
import RFB from "./novnc/core/rfb.js";

const $ = (id) => document.getElementById(id);
// In the desktop app on a Mac the window's own buttons sit over the page's top-left corner.
if (window.coderunnerShell && /Mac/i.test(navigator.platform || "")) document.documentElement.classList.add("mac-shell");
const HEADERS = {"Content-Type": "application/json", "X-CodeRunner": "1"};

// The daemon restarts now and then (a setting changed, the Mac slept, it was stopped). While it does not
// answer the app keeps what is on screen, says one calm thing, tries again quietly and picks everything up
// by itself when it is back. No request's own error text is ever shown for this.
const link = {down: false, since: 0, timer: null, wait: 1000, asked: false};

class Offline extends Error {
  constructor() { super(""); this.offline = true; }
}

async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, {credentials: "same-origin", headers: HEADERS, ...options}); }
  catch { setOffline(true); throw new Offline(); }
  if (response.status >= 502 && response.status <= 504) { setOffline(true); throw new Offline(); }
  setOffline(false);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "That did not work. Try again.");
  return data;
}

function setOffline(down) {
  if (down === link.down) return;
  link.down = down;
  clearTimeout(link.timer);
  if (down) {
    link.since = Date.now(); link.wait = 1000; link.asked = false;
    $("offlineText").textContent = "Reconnecting…";
    document.querySelector("#offline button")?.remove();
    $("offline").hidden = false;
    document.body.classList.add("is-offline");
    link.timer = setTimeout(probe, link.wait);
    return;
  }
  $("offline").hidden = true;
  document.body.classList.remove("is-offline");
  recover();
}

async function probe() {
  let up = false;
  try { up = (await fetch("/health", {cache: "no-store"})).ok; } catch { /* still away */ }
  if (up) { setOffline(false); return; }
  link.wait = Math.min(link.wait * 1.6, 8000);
  // Gone for a while: in the desktop app, where the assistant can be started from here, say so and offer it.
  if (Date.now() - link.since > 12000 && window.coderunnerShell?.startAssistant && !link.asked) {
    link.asked = true;
    $("offlineText").textContent = "The assistant is not running.";
    const start = el("button", "linkish", "Start it");
    start.type = "button";
    start.onclick = () => { start.remove(); $("offlineText").textContent = "Starting…"; window.coderunnerShell.startAssistant(); };
    $("offline").appendChild(start);
  }
  link.timer = setTimeout(probe, link.wait);
}

// Back: the models, the task list, the thread on screen (a task may have ended meanwhile) and the screen.
function recover() {
  if (!link.since) return;
  link.since = 0;
  if (state.setup) { checkSetup(); return; }
  loadModels();
  pollApprovals();
  if (state.conversation && state.conversation !== "check") openConversation(state.conversation, $("title").textContent, true);
  else loadConversations();
  if (!rfb) connectScreen();
}

// Something the owner should know that is not part of a task: said above the task box, in the app's own
// style, and gone by itself.
let noticeTimer = null;
function notice(text) {
  if (!text) return;                  // an Offline error: the reconnecting line already says it
  const box = $("notice");
  box.textContent = text;
  box.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { box.hidden = true; }, 7000);
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
      // An address reads as where it leads: the site and the start of its path. The whole of it is the link.
      const url = part.replace(/[.,;:!?]+$/, ""), rest = part.slice(url.length);
      const a = el("a", "link", linkLabel(url)); a.href = url; a.title = url; a.target = "_blank"; a.rel = "noopener noreferrer";
      node.appendChild(a);
      if (rest) node.appendChild(document.createTextNode(rest));
    } else if (part) node.appendChild(document.createTextNode(part));
  }
  return node;
}

function linkLabel(url) {
  try {
    const u = new URL(url), site = u.host.replace(/^www\./, "");
    const path = decodeURIComponent(u.pathname).replace(/\/$/, "");
    const tail = path.length > 28 ? path.slice(0, 27) + "…" : path + (u.search ? "…" : "");
    return site + (tail === "…" && !path ? "" : tail);
  } catch { return url.length > 48 ? url.slice(0, 47) + "…" : url; }
}

// A small, safe subset of Markdown (paragraphs, lists, tables, **bold**, `code`, links), built as DOM nodes.
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
  const lines = String(text ?? "").split("\n").map((raw) => raw.trimEnd());
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  const isRow = (line) => /^\s*\|.*\|\s*$/.test(line || "");
  const isRule = (line) => isRow(line) && cells(line).every((cell) => /^:?-{2,}:?$/.test(cell));
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (isRow(line) && isRule(lines[i + 1])) {             // a table: its head, a rule, then its rows
      const table = el("table", "md-table"), head = table.createTHead().insertRow();
      for (const cell of cells(line)) inline(head.appendChild(el("th")), cell);
      const body = table.createTBody();
      for (i += 2; i < lines.length && isRow(lines[i]); i++) {
        const row = body.insertRow();
        for (const cell of cells(lines[i])) inline(row.insertCell(), cell);
      }
      i -= 1;
      node.appendChild(el("div", "md-table-wrap")).appendChild(table);
      list = null; para = null;
      continue;
    }
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
    case "browser_search": return `Searching the web for “${String(a.query || "").slice(0, 60)}”`;
    case "browser_snapshot": return "Reading the page";
    case "browser_list": return "Reading the results on the page";
    case "browser_click": return "Working on the page";
    case "browser_type": return a.text ? `Typing “${trunc(a.text, 40)}”` : "Typing";
    case "browser_batch": return "Filling in the page";
    case "browser_find": return `Looking for “${trunc(a.text || a.query, 40)}” on the page`;
    case "browser_select": return `Choosing “${trunc(a.option, 40)}”`;
    case "browser_press": return "Working on the page";
    case "browser_scroll": return "Reading further down the page";
    case "browser_back": return "Going back";
    case "browser_tabs": return "Looking at the open pages";
    case "browser_screenshot": return "Looking at the screen";
    case "browser_handoff": return "Asking you to take over";
    case "browser_task": return `Working in the browser: ${trunc(a.goal || a.task || "", 90)}`;
    case "run_shell": return "Working on the computer";
    case "shell_job": return "Waiting for a job on the computer";
    case "run_python": return "Working something out";
    case "files_search": return `Searching your files for “${trunc(a.query, 50)}”`;
    case "file_read": return `Reading ${base(a.path)}`;
    case "file_view": return `Looking at ${base(a.path)}`;
    case "file_listen": return `Listening to ${base(a.path)}`;
    case "memory_search": return "Checking what I remember";
    case "memory_save": return "Remembering this";
    case "plan_update": return "Updating the plan";
    case "tell_owner": return "Working";
    case "schedule_create": return "Scheduling it";
    case "send_message": return "Sending a message";
    default: return "Working";
  }
}

// What was clicked or typed into, as it reads on the page: without the kind of control or the page's own text.
const plainLabel = (label) => trunc(String(label || "").replace(/^(button|link|option|textbox|searchbox|combobox|checkbox|radio|tab|menuitem|switch|listitem|heading)\s+/i, "")
  .replace(/\s*\(page text .*$/i, "").replace(/^"(.*)"$/, "$1").replace(/"\s*->.*$/, "").replace(/^"|"$/g, ""), 60);

function describeActivity(e) {
  const ok = e.ok !== false;
  switch (e.event) {
    case "browser.started": return "Started the computer's browser";
    case "browser.open": case "browser.navigate": return ok ? `Opened ${host(e.url || e.page)}` : null;
    case "browser.click": return e.label ? `Chose “${plainLabel(e.label)}”` : "Working on the page";
    case "browser.type": return e.label ? `Filling in “${plainLabel(e.label)}”` : "Filling in the page";
    case "browser.select": return `Chose “${trunc(e.option, 60)}”`;
    case "browser.press": return "Working on the page";
    case "browser.click_at": return "Working on the page";
    case "browser.type_focused": return "Filling in the page";
    case "browser.scroll": case "browser.wheel": return "Reading further down the page";
    case "driver.start": return "Working through the page";
    case "driver.finish": return "Reading what came of it";
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

// A task reads as a conversation: what the assistant says along the way, one line for what it is doing now, its
// plan ticking off, a picture of the page where that says more than words, and the answer. The calls it makes
// (which tool, with what arguments) are not shown; they are in the audit trail for whoever wants them.
class TaskView {
  constructor(task) {
    this.id = task.id;
    this.started = task.created || Date.now() / 1000;
    this.finished = task.finished || null;
    this.status = task.status || "running";
    this.seq = 0;
    this.steps = 0;
    this.root = el("article", "task");
    this.bubble = el("div", "bubble-user", task.prompt);
    this.bubble.hidden = !task.prompt;          // files alone, with no words
    this.root.appendChild(this.bubble);
    this.flow = el("div", "flow");
    this.root.appendChild(this.flow);
    this.now = null;              // the line that says what is happening
    this.plan = null;             // the plan, kept in one place and updated
    $("thread").appendChild(this.root);
    if (this.status === "running") this.setNow("Thinking");
  }

  // Something to read, placed above the line that says what is happening now.
  say(node) {
    if (this.now) this.flow.insertBefore(node, this.now.row); else this.flow.appendChild(node);
    return node;
  }

  setNow(text) {
    if (this.status !== "running" || !text) return;
    if (!this.now) {
      const row = el("div", "now"), stop = el("button", "stop", "Stop");
      const what = el("span", "now-what"), time = el("span", "now-time");
      stop.onclick = async () => {
        stop.disabled = true;
        await api(`/api/tasks/${this.id}/cancel`, {method: "POST", body: "{}"}).catch(() => {});
      };
      row.append(el("span", "now-dot"), what, time, stop);
      this.flow.appendChild(row);
      this.now = {row, what, time};
    }
    this.now.what.textContent = text.replace(/…$/, "") + "…";
    setNow(text);
  }

  showPlan(steps) {
    if (!Array.isArray(steps) || !steps.length) return;
    if (!this.plan) this.plan = this.say(el("div", "plan"));
    this.plan.replaceChildren();
    for (const item of steps) {
      const state = item.status === "done" ? "done" : ["doing", "in_progress", "active"].includes(item.status) ? "doing" : "todo";
      const row = el("div", "plan-step " + state);
      row.append(el("span", "plan-mark", state === "done" ? "✓" : state === "doing" ? "•" : ""), el("span", "", item.step || item.content || ""));
      this.plan.appendChild(row);
    }
  }

  // What the owner attached, above their words: pictures and videos as themselves, anything else by name.
  showAttached(files) {
    if (!Array.isArray(files) || !files.length || this.attached) return;
    this.attached = el("div", "attached");
    for (const file of files) this.attached.appendChild(attachedFile(file));
    this.root.insertBefore(this.attached, this.bubble);
  }

  // What the task made (a chart, a sheet, a document its code wrote), under its answer.
  showProduced(files) {
    if (!Array.isArray(files) || !files.length || this.produced) return;
    this.produced = el("div", "produced");
    for (const file of files) this.produced.appendChild(attachedFile(file));
    this.flow.insertBefore(this.produced, this.flow.querySelector(".took"));
  }

  showPicture(e) {
    const figure = el("figure", "shot"), image = el("img");
    image.src = `/api/shots/${encodeURIComponent(e.file)}`;
    image.alt = `The page at ${host(e.page)}`;
    image.loading = "lazy";
    figure.append(image, el("figcaption", "muted small", host(e.page)));
    figure.onclick = () => figure.classList.toggle("large");
    image.onload = () => { image.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`; };
    image.onerror = () => figure.remove();          // a picture cleared away since: the words still stand
    this.say(figure);
  }

  apply(e) {
    if (e.seq && e.seq <= this.seq) return;
    if (e.seq) this.seq = e.seq;
    switch (e.type) {
      case "started": this.showAttached(e.attachments); break;
      case "note": this.say(markdown(el("div", "note"), e.text)); break;
      case "plan": this.showPlan(e.steps); break;
      case "tool":
        if (e.name === "plan_update" || e.name === "tell_owner") break;
        this.steps += 1;
        this.setNow(describeTool(e.name, e.arguments));
        break;
      case "tool_done": break;
      case "shot": this.showPicture(e); break;
      case "activity":
        if (e.event === "approval.requested") this.ask(e);
        else if (e.event === "approval.decided") closeAsk(e.id, e.status === "approved" ? true : e.status === "denied" ? false : "late");
        else if (e.ok !== false) this.setNow(describeActivity(e));
        break;
      case "done": case "failed": case "cancelled": this.finish(e); break;
      case "files": this.showProduced(e.files); break;
    }
    this.tick();
  }

  ask(e) {
    const handoff = e.kind === "browser.handoff";
    const card = el("div", "ask");
    card.appendChild(el("div", "ask-title", handoff ? "Your turn on the computer" : "Needs your approval"));
    // One sentence says what is asked; the step itself (a script, a form) is shown small under it.
    const [first, ...rest] = String(e.summary || e.kind).split(/\n+/);
    const share = e.kind === "data.share";
    // What it holds is said as the policy names it: your files, your memory, your mail.
    const holds = (/holds data from your ([^;]+);/.exec(first) || [])[1] || "files";
    // A step on a page is asked as a person would ask it: the control's own name, cut where its description
    // starts, and the site. The whole of what the page calls it is under the question.
    const act = /^(Click|Double-click|Triple-click) (.+) on ([\w.-]+)$/s.exec(first);
    const named = act && plainLabel(act[2].replace(/&amp;/g, "&")).split(/, |\s[·|-]\s/)[0];
    // The owner's turn is asked in the assistant's own words, once.
    const turn = handoff && first.replace(/^Please finish in the CodeRunner browser: /, "").replace(/^please /i, "")
      .replace(/^./, (c) => c.toUpperCase());
    const question = handoff ? turn : share ? `This task holds data from your ${holds}. Allow it to use that data on ${e.target}?` :
      act ? `${act[1]} “${named}” on ${act[3].replace(/^www\./, "")}?` : first;
    card.appendChild(richText(el("div", ""), question));
    const whole = act && plainLabel(act[2]) !== named ? act[2].replace(/&amp;/g, "&") : "";
    const step = (share ? rest.join("\n").replace(/^First step: /, "") : [whole, ...rest].filter(Boolean).join("\n")).trim();
    // What a high-risk step would send or change (a message, a setting, a call) is shown whole, as written:
    // nothing is allowed on the strength of its first two lines. Anything else is two small lines that open.
    const grave = e.risk === "high" && !act;
    if (grave) this.showWhole(card, String(e.summary || ""));
    else if (step) {
      const more = el("div", "ask-step", step);
      more.title = "Show all of it";
      more.onclick = () => more.classList.toggle("open");
      card.appendChild(more);
    }
    const row = el("div", "row"), before = this.now?.what.textContent || "Working";
    const decide = async (allow, scope = "once") => {
      row.querySelectorAll("button").forEach((b) => b.disabled = true);
      try { await api(`/api/approvals/${e.id}`, {method: "POST", body: JSON.stringify({allow, scope})}); }
      catch (err) { row.querySelectorAll("button").forEach((b) => b.disabled = false); notice(err.message); return; }
      closeAsk(e.id, allow);
      this.setNow(before);            // no longer waiting for the owner
    };
    if (handoff) {
      const take = el("button", "btn primary", "Take control");
      const done = el("button", "btn", "I'm done");
      const finish = () => { setControl(false); setExpanded(false); decide(true); };
      // One button leads: taking the screen first, handing it back after.
      take.onclick = () => { showComputer(true); setControl(true); setExpanded(true); take.remove(); done.classList.add("primary"); };
      done.onclick = finish;
      const skip = el("button", "btn danger", "Skip");
      skip.onclick = () => decide(false);
      row.append(take, done, skip);
      showComputer(true);
      $("turnText").textContent = turn;
      $("turnDone").onclick = finish;
      $("turn").hidden = false;
      $("turn").dataset.ask = e.id;
    } else {
      const allow = el("button", "btn primary", "Allow");
      allow.onclick = () => decide(true);
      const deny = el("button", "btn danger", "Deny");
      deny.onclick = () => decide(false);
      row.append(allow);
      if (share) {                    // a task that reads several sites asks once, not once a site
        const all = el("button", "btn", "Allow for this task");
        all.title = "Every site this task opens, until it ends";
        all.onclick = () => decide(true, "task");
        row.append(all);
      }
      row.append(deny);
    }
    card.appendChild(row);
    this.say(card);
    this.setNow("Waiting for you");
    state.approvals.set(e.id, {conversation: state.conversation, card, row});
    if (!state.pending.has(e.id) && e.ts && Date.now() / 1000 - e.ts > 5) closeAsk(e.id, null);
    card.scrollIntoView({block: "nearest", behavior: "smooth"});
  }

  // What a message, a call or a setting change will be, in full: who it reaches on lines of their own, then
  // every word as written (it scrolls inside the card when long), and a caution about a recipient the owner
  // never named set apart under it.
  showWhole(card, summary) {
    let rest = summary.includes("\n") ? summary.slice(summary.indexOf("\n") + 1) : "";
    const check = /\n\n(Check [^\n]+: not in what you asked, nor in your contacts\.)\s*$/.exec(rest);
    if (check) rest = rest.slice(0, check.index);
    if (rest.trim()) {
      const body = el("div", "ask-body");
      const lines = rest.split("\n");
      while (lines.length && /^(To|Cc|Bcc|From|Subject|Number|Name): /.test(lines[0])) {
        const [, label, value] = /^(\w+): (.*)$/.exec(lines.shift());
        const line = el("div", "ask-field");
        line.append(el("span", "ask-label", label), el("span", "", value));
        body.appendChild(line);
      }
      if (lines[0] === "Body:") lines.shift();
      const text = lines.join("\n").replace(/^\n+|\s+$/g, "");
      if (text) body.appendChild(el("div", "ask-text", text));
      card.appendChild(body);
    }
    if (check) card.appendChild(el("div", "ask-caution", check[1]));
  }

  finish(e) {
    this.status = e.type;
    this.finished = e.ts;
    this.now?.row.remove();
    this.now = null;
    const answer = el("div", "answer" + (e.type === "failed" ? " failed" : ""));
    markdown(answer, e.reply || "");
    this.flow.appendChild(answer);
    const took = clock((this.finished || Date.now() / 1000) - this.started);
    this.flow.appendChild(el("div", "took muted small", e.type === "cancelled" ? `Stopped after ${took}` :
                             e.type === "failed" ? `Failed after ${took}` : `Done in ${took}`));
    state.running.delete(this.id);
    refreshNow();
  }

  tick() {
    if (this.now) this.now.time.textContent = clock(Date.now() / 1000 - this.started);
  }
}

function closeAsk(id, allowed) {
  const entry = state.approvals.get(id);
  state.pending.delete(id);
  if ($("turn").dataset.ask === id) $("turn").hidden = true;
  if (!entry || entry.card.classList.contains("closed")) return;
  entry.card.classList.add("closed");
  entry.row.replaceChildren(el("span", "muted small", allowed === "late" ? "Not answered in time" : allowed === null ? "Answered" :
                                                      allowed ? "Allowed" : "Declined"));
}

// ---- conversations ----------------------------------------------------------

async function loadConversations() {
  const list = await api("/api/conversations").catch(() => []);
  const nav = $("conversations");
  nav.replaceChildren();
  for (const c of list) {
    // The check's own tasks are on the Check screen; here only the one running now, or the one being read.
    if (c.conversation.startsWith("app:check-") && c.status !== "running" && c.conversation !== state.conversation) continue;
    const button = el("button", "conv" + (c.conversation === state.conversation ? " on" : ""));
    const t = el("div", "t");
    const waiting = [...state.approvals.values()].some((a) => a.conversation === c.conversation &&
                                                       !a.card.classList.contains("closed"));
    const status = waiting ? "waiting" : c.status;
    t.append(el("span", "dot " + status), el("span", "label", trunc(c.title || "Attached files", 60)));
    const sub = status === "running" ? "Working…" : status === "waiting" ? "Needs you" :
                status === "failed" ? "Failed" : status === "cancelled" ? "Stopped" :
                new Date(c.updated * 1000).toLocaleString([], {dateStyle: "medium", timeStyle: "short"});
    button.append(t, el("div", "s", sub));
    button.onclick = () => openConversation(c.conversation, c.title || "Attached files");
    nav.appendChild(button);
  }
}

async function openConversation(id, title, again) {
  if (state.setup) return;        // the first-run screen holds the page until a model is set up
  let data = null;
  if (id && again) {              // the same thread, read again after a reconnect: what is shown stays until it is here
    try { data = await api(`/api/conversations/${encodeURIComponent(id)}`); } catch { return; }
    if (state.conversation !== id) return;
  }
  const kept = again ? $("thread").scrollTop : null;
  state.conversation = id;
  state.views.clear();
  $("thread").replaceChildren();
  $("title").textContent = id ? trunc(title || "Task", 90) : "New task";
  if (!id) { $("thread").appendChild(emptyState); loadConversations(); $("input").focus(); return; }
  try { data = data || await api(`/api/conversations/${encodeURIComponent(id)}`); }
  catch (err) { notice(err.message); return; }
  if (kept !== null) setTimeout(() => { $("thread").scrollTop = kept; }, 0);
  for (const task of data.tasks) state.views.set(task.id, new TaskView(task));
  for (const e of data.events) state.views.get(e.task)?.apply(e);
  loadConversations();
  scrollDown();
}

// ---- check: each kind of task done once on this computer --------------------------

let checkTimer = null;
let checkSet = "check";             // "check": this install's own tasks; "real": the twenty on real websites
const CHECK_MARKS = {passed: "✓", failed: "✕", blocked: "!", needs_you: "→", skipped: "–", waiting: "", running: ""};
const CHECK_LEADS = {check: "Each kind of task, done once on this computer and checked.",
                     real: "Twenty everyday tasks and five forms on real websites. Nothing is signed in to, paid for or submitted."};

async function showCheck() {
  if (state.setup) return;
  state.conversation = "check";             // not a conversation: no task's events belong to it
  state.views.clear();
  $("title").textContent = "Check";
  loadConversations();
  const box = el("section", "checkup");
  box.append(el("h1", "", "Check"), el("p", "muted", CHECK_LEADS[checkSet]));
  $("thread").replaceChildren(box);
  const run = async (body) => {
    try { draw(await api("/api/check", {method: "POST", body: JSON.stringify(body)})); }
    catch (err) { await poll(); if (err.message) box.appendChild(el("p", "setup-error", err.message)); return; }
    poll();
  };
  const draw = (data) => {
    const real = checkSet === "real", running = data.state === "running" && (data.suite || "check") === checkSet;
    const other = data.state === "running" && !running;       // the other set is running: this one waits
    const result = real ? data.real?.result : data.suite === "real" ? data.check : data.result || data.check;
    const names = real ? data.real?.names || [] : data.names.filter((n) => !n.long);
    const rows = running ? data.tasks : result ? result.tasks : names.map((n) => ({...n, state: "waiting"}));
    const tabs = el("div", "tabs");
    for (const [key, label] of [["check", "This install"], ["real", "Real websites"]]) {
      const tab = el("button", "tab" + (key === checkSet ? " on" : ""), label);
      tab.type = "button";
      tab.onclick = () => { if (checkSet !== key) { checkSet = key; draw(data); } };
      tabs.appendChild(tab);
    }
    box.replaceChildren(el("h1", "", "Check"), el("p", "muted", CHECK_LEADS[checkSet]), tabs);
    const list = el("div", "check-list");
    let forms = false;
    for (const task of rows) {
      if (task.kind === "forms" && !forms) { forms = true; list.appendChild(el("div", "check-group", "Forms")); }
      const row = el(task.conversation ? "button" : "div", "check-row " + task.state);
      const named = task.what.replace(/: CAPTCHA$/, "");
      const what = el("span", "check-what", named.charAt(0).toUpperCase() + named.slice(1));
      if (task.why && task.state !== "passed") what.appendChild(el("span", "check-why", task.why));
      const took = task.seconds != null ? clock(task.seconds) : "";
      row.append(el("span", "check-mark", CHECK_MARKS[task.state]), what,
                 el("span", "check-time", took + (task.approvals ? ` · ${task.approvals} asked` : "")));
      row.title = [task.why, task.sites?.length ? task.sites.join(", ") : ""].filter(Boolean).join(" · ");
      if (task.conversation) { row.type = "button"; row.onclick = () => openConversation(task.conversation, task.what); }
      list.appendChild(row);
    }
    box.appendChild(list);
    const foot = el("div", "check-foot");
    if (running) {
      const stop = el("button", "btn", "Stop");
      stop.onclick = () => { stop.disabled = true; run({action: "stop"}); };
      foot.append(el("span", "muted", "Running. You can leave this screen."), stop);
    } else {
      if (result) {
        const when = new Date(result.date).toLocaleString([], {month: "short", day: "numeric", hour: "numeric", minute: "2-digit"});
        foot.appendChild(el("span", "muted", `${result.passed} of ${result.ran} passed` +
                                              (result.blocked ? `, ${result.blocked} blocked by a site` : "") +
                                              (result.needs_you ? `, ${result.needs_you} waiting for you` : "") +
                                              (result.skipped ? `, ${result.skipped} skipped` : "") + ` · ${modelName(result.model)} · ${when}`));
      }
      const go = el("button", "btn primary", result ? "Run again" : real ? "Run all" : "Run check");
      go.disabled = other;
      go.onclick = () => { go.disabled = true; run(real ? {real: true} : {}); };
      foot.appendChild(go);
      if (!real) {
        const more = el("button", "linkish", "Include the long tasks");
        more.type = "button"; more.disabled = other;
        more.title = "A flight search, a booking up to payment, and three more: a few more minutes";
        more.onclick = () => { more.disabled = true; run({long: true}); };
        foot.appendChild(more);
      }
    }
    box.appendChild(foot);
  };
  const poll = async () => {
    clearTimeout(checkTimer);
    if (state.conversation !== "check") return;
    let data;
    try { data = await api("/api/check"); } catch { checkTimer = setTimeout(poll, 4000); return; }
    if (state.conversation !== "check") return;
    draw(data);
    if (data.state === "running") checkTimer = setTimeout(poll, 2000);
  };
  await poll();
}
$("openCheck").onclick = showCheck;

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

// ---- files attached to a task -----------------------------------------------------

const FILE_ICON = "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z M14 3v5h5";
const CLIP_ICON = "M20.5 11.5l-8.2 8.2a5 5 0 0 1-7.1-7.1l8.5-8.5a3.4 3.4 0 0 1 4.8 4.8l-8.4 8.4a1.8 1.8 0 0 1-2.6-2.6l7.6-7.6";
const MAX_ATTACHED = 8;
const sizeText = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e5 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(Math.round(n / 1000), 1)} KB`;
const mediaKind = (type) => /^image\/(?!svg)/.test(type) ? "image" : /^video\//.test(type) ? "video" : /^audio\//.test(type) ? "audio" : "file";

// A picture at full size over the app; a click or Escape puts it away.
function lightbox(src, alt) {
  const box = el("div", "lightbox"), image = el("img");
  image.src = src; image.alt = alt || "";
  box.appendChild(image);
  const close = () => { box.remove(); document.removeEventListener("keydown", onKey, true); };
  const onKey = (event) => { if (event.key === "Escape") { event.stopPropagation(); close(); } };
  box.onclick = close;
  document.addEventListener("keydown", onKey, true);
  document.body.appendChild(box);
}

// A file by its name and size, for what cannot be shown as itself.
function fileCard(name, size, href) {
  const card = el(href ? "a" : "div", "att-file"), words = el("span", "att-words");
  words.append(el("span", "att-name", name), el("span", "att-size", sizeText(size)));
  card.append(icon(FILE_ICON), words);
  card.title = name;
  if (href) { card.href = href; card.download = name; }
  return card;
}

// One attached file in a task's thread: {name, kind, size, url, inline}.
function attachedFile(file) {
  if (file.kind === "image" && file.inline) {
    const image = el("img", "att-image");
    image.src = file.url; image.alt = file.name; image.title = file.name; image.loading = "lazy";
    image.onclick = () => lightbox(file.url, file.name);
    image.onerror = () => image.replaceWith(fileCard(file.name, file.size));
    return image;
  }
  if (file.kind === "video" && file.inline) {
    const video = el("video", "att-video");
    video.src = `${file.url}#t=0.1`; video.controls = true; video.preload = "metadata"; video.title = file.name;
    video.onerror = () => video.replaceWith(fileCard(file.name, file.size, file.url));   // a format this browser cannot play
    return video;
  }
  if (file.kind === "audio" && file.inline) {
    const box = el("div", "att-audio"), audio = el("audio");
    audio.src = file.url; audio.controls = true; audio.preload = "metadata";
    box.append(el("span", "att-name", file.name), audio);
    return box;
  }
  return fileCard(file.name, file.size, file.url);
}

// What is waiting in the composer: each file is sent to the daemon as soon as it is chosen, and shown small
// with how far it has got. {file, id, tile, request, failed}
const waiting = [];

function refreshComposer() {
  $("attachments").hidden = !waiting.length;
  $("send").disabled = waiting.some((item) => !item.id && !item.failed);
}

function dropWaiting(item) {
  item.request?.abort();
  if (item.preview) URL.revokeObjectURL(item.preview);
  item.tile.remove();
  waiting.splice(waiting.indexOf(item), 1);
  refreshComposer();
}

function attach(files) {
  for (const file of files) {
    if (waiting.length >= MAX_ATTACHED) { notice(`At most ${MAX_ATTACHED} files to a message.`); break; }
    const kind = mediaKind(file.type), name = file.name || `pasted.${(file.type.split("/")[1] || "bin").replace(/\W.*/, "")}`;
    const item = {file, id: "", failed: "", preview: kind === "image" || kind === "video" ? URL.createObjectURL(file) : ""};
    const tile = item.tile = el("div", "wait " + kind), remove = el("button", "wait-remove", "×"), bar = el("span", "wait-bar");
    remove.type = "button"; remove.title = "Remove"; remove.setAttribute("aria-label", `Remove ${name}`);
    remove.onclick = () => dropWaiting(item);
    if (kind === "image") { const image = el("img"); image.src = item.preview; image.alt = name; tile.appendChild(image); }
    else if (kind === "video") {
      const video = el("video"); video.src = `${item.preview}#t=0.1`; video.muted = true; video.preload = "metadata";
      tile.append(video, el("span", "wait-play"));
    } else tile.appendChild(fileCard(name, file.size));
    tile.title = `${name} · ${sizeText(file.size)}`;
    tile.append(bar, remove);
    $("attachments").appendChild(tile);
    waiting.push(item);
    const request = item.request = new XMLHttpRequest();
    request.open("POST", "/api/uploads");
    request.setRequestHeader("X-CodeRunner", "1");
    request.setRequestHeader("X-File-Name", encodeURIComponent(name));
    request.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    request.upload.onprogress = (event) => { if (event.lengthComputable) bar.style.width = `${event.loaded / event.total * 100}%`; };
    const fail = (why) => { item.failed = why; tile.classList.add("failed"); tile.title = `${name}: ${why}`; refreshComposer(); };
    request.onload = () => {
      let data = {};
      try { data = JSON.parse(request.responseText); } catch { /* not ours */ }
      if (request.status >= 300 || !data.id) return fail(data.error || "It was not kept.");
      item.id = data.id;
      tile.classList.add("kept");
      refreshComposer();
    };
    request.onerror = () => fail("The upload did not go through.");
    request.send(file);
  }
  refreshComposer();
  $("input").focus();
}

$("attach").appendChild(icon(CLIP_ICON));
$("attach").onclick = () => $("filePick").click();
$("filePick").onchange = () => { attach([...$("filePick").files]); $("filePick").value = ""; };
$("input").addEventListener("paste", (event) => {
  const files = [...(event.clipboardData?.files || [])];
  if (files.length) { event.preventDefault(); attach(files); }
});
// Files dragged over the window are attached wherever they are dropped.
let dragging = 0;
const carriesFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files") && !$("composer").hidden;
window.addEventListener("dragenter", (event) => { if (carriesFiles(event)) { dragging += 1; $("main").classList.add("dropping"); } });
window.addEventListener("dragleave", (event) => { if (carriesFiles(event) && --dragging <= 0) { dragging = 0; $("main").classList.remove("dropping"); } });
window.addEventListener("dragover", (event) => { if (carriesFiles(event)) event.preventDefault(); });
window.addEventListener("drop", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragging = 0;
  $("main").classList.remove("dropping");
  attach([...event.dataTransfer.files]);
});

// ---- composer ------------------------------------------------------------------

const emptyState = $("empty");
$("composer").onsubmit = async (event) => {
  event.preventDefault();
  const message = $("input").value.trim();
  const failed = waiting.find((item) => item.failed);
  if (failed) { notice(`${failed.file.name || "A file"} was not attached: ${failed.failed} Remove it to go on.`); return; }
  const attachments = waiting.filter((item) => item.id).map((item) => item.id);
  if ((!message && !attachments.length) || waiting.some((item) => !item.id)) return;
  $("send").disabled = true;
  try {
    const task = await api("/api/tasks", {method: "POST",
      body: JSON.stringify({message, conversation: state.conversation || "", attachments})});
    $("input").value = ""; grow();
    for (const item of [...waiting]) dropWaiting(item);
    if (task.conversation !== state.conversation) {
      await openConversation(task.conversation, message || "Attached files");
    }
  } catch (err) { notice(err.message); }
  finally { refreshComposer(); }
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
  catch {                             // the daemon is away: the line above the thread says so, and it comes back
    setNote("Reconnecting to the screen…");
    $("computerState").textContent = "Reconnecting";
    retry = setTimeout(connectScreen, 4000);
    return;
  }
  $("startComputer")?.remove();
  if (info.reachable === false) {   // no sandbox answering: coming up, stopped, or never installed
    const starting = info.state === "starting";
    setNote(starting ? "Starting the computer…" : info.error);
    $("computerState").textContent = starting ? "Starting" : info.state === "absent" ? "Not installed" : "Stopped";
    if (computerBox) computerBox.box.hidden = true;
    if (info.state === "stopped") {
      const start = el("button", "btn primary", "Start it");
      start.id = "startComputer"; start.type = "button";
      start.onclick = async () => {
        start.remove(); setNote("Starting the computer…"); $("computerState").textContent = "Starting";
        clearTimeout(retry);
        try { await api("/api/computer", {method: "POST", body: JSON.stringify({start: true})}); } catch { /* said above the thread */ }
        connectScreen();
      };
      $("screenNote").appendChild(start);
    }
    retry = setTimeout(connectScreen, starting ? 3000 : 10000);
    return;
  }
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

// A model's id as an owner reads it: coderunner-bonsai-2-27b -> Bonsai 2 27B.
const MODEL_NAMES = {"embeddinggemma-2": "EmbeddingGemma 2", "embeddinggemma-300m": "EmbeddingGemma 300M", "tev1-4b": "tev1 4B"};
function modelName(id) {
  if (!id) return "";
  const claude = /^claude-([a-z]+)-(\d+)(?:-(\d+))?$/.exec(String(id));       // claude-haiku-5-5 -> Claude Haiku 5.5
  if (claude) return `Claude ${claude[1][0].toUpperCase()}${claude[1].slice(1)} ${claude[2]}${claude[3] ? "." + claude[3] : ""}`;
  const base = String(id).split("/").pop().split(":")[0].replace(/^coderunner-/, "").replace(/^hf-/, "");
  if (MODEL_NAMES[base]) return MODEL_NAMES[base];
  return base.split("-").filter(Boolean).map((word) => /^[ae]?\d+(\.\d+)?[bm]$/i.test(word) ? word.toUpperCase() :
    /^\d+bit$/.test(word) ? word.replace("bit", "-bit") : word[0].toUpperCase() + word.slice(1)).join(" ");
}
const VOICES = {apple: "Apple's on-device speech", parakeet: "Parakeet", "parakeet-coreml": "Parakeet Ultra", "qwen3-asr": "Qwen3 ASR", whisper: "Whisper"};
const BROWSERS = {computer: "The assistant's own computer", chrome: "Your Chrome, in its own tab group", dedicated: "A Chrome window of its own"};

async function loadModels() {
  try {
    const status = await api("/api/status");
    const models = status.models || {}, remote = status.remote;
    // In remote mode Claude plans, works the web pages and judges the steps; file search and voice stay here.
    const here = remote ? " · on this computer" : "";
    const voice = (status.voice || []).slice(0, 1).map((name) => VOICES[name] || name).join("");   // the one in use; the rest stand by
    const search = modelName(models.embedding || status.model?.embedding);
    const rows = [["Planning", modelName(models.chat || status.model?.model)], ["Web pages", modelName(models.driver)],
                  ["Safety check", modelName(models.decision)], ["File search", search && search + here],
                  ["Voice", voice && voice + here],
                  ["Speed", remote?.step_s ? `${remote.step_s} s a step, measured` : ""],
                  ["Browser", BROWSERS[status.browser?.mode] || status.browser?.mode]];
    const dl = $("models");
    dl.replaceChildren();
    for (const [k, v] of rows) if (v) dl.append(el("dt", "", k), el("dd", "", v));
    const line = $("modelsLine");
    line.replaceChildren();
    if (remote) {
      const where = el("span", "where", "Anthropic");
      where.appendChild(el("span", "not-private", "Not private"));
      line.append(el("span", "", modelName(remote.model)), where);
    }
    else if (models.chat) line.append(el("span", "", `Local · ${modelName(models.chat)}`));
    line.title = remote ? "Your requests, the pages it reads and the files it opens are sent to Anthropic." : "";
    const lead = document.querySelector("#empty p");
    if (lead) lead.textContent = remote ?
      "Planned by Claude on Anthropic's API, and carried out on a private Linux machine with its own browser that you can watch and take over." :
      "Tasks run on your own computer: local models, and a private Linux machine with its own browser that you can watch and take over.";
    loadRemote();
  } catch { /* shown once the daemon answers */ }
}

// Remote mode, under the models: on with its warning and the way back, or the offer to turn it on with a key.
async function loadRemote() {
  const box = $("remoteBox");
  let remote;
  try { remote = await api("/api/remote"); } catch { box.replaceChildren(); return; }
  box.replaceChildren();
  const change = async (body, button) => {
    button.disabled = true;
    try { await api("/api/remote", {method: "POST", body: JSON.stringify(body)}); }
    catch (err) { box.appendChild(el("p", "setup-error", err.message)); button.disabled = false; return; }
    box.replaceChildren(el("p", "muted small", "Switching…"));
    whenDaemonReturns(() => location.reload());
  };
  if (remote.on) {
    box.appendChild(el("p", "remote-warning", remote.warning));
    const back = el("button", "btn", remote.local ? "Use the local model" : "Set up a local model");
    back.onclick = () => change({action: "off"}, back);
    box.appendChild(back);
    return;
  }
  const offer = el("button", "linkish small", "Use an Anthropic key");
  offer.type = "button";
  offer.onclick = () => { box.replaceChildren(remoteForm(remote, change, loadRemote)); box.querySelector("input, button.primary")?.focus(); };
  box.appendChild(offer);
}

// The form that turns remote mode on: what it means in one sentence, the key (unless one is kept), one button.
function remoteForm(remote, change, cancel) {
  const form = el("form", "remote-form"), key = el("input"), go = el("button", "btn primary", "Use Claude"), no = el("button", "linkish small", "Cancel");
  key.type = "password"; key.placeholder = "Anthropic API key"; key.autocomplete = "off"; key.spellcheck = false;
  key.setAttribute("aria-label", "Anthropic API key");
  no.type = "button"; no.onclick = () => cancel();
  form.appendChild(el("p", "remote-warning", remote.warning));
  if (!remote.has_key) form.appendChild(key);
  const row = el("div", "row"); row.append(go, no);
  form.appendChild(row);
  form.onsubmit = (event) => {
    event.preventDefault();
    if (!remote.has_key && !key.value.trim()) { key.focus(); return; }
    change({action: "on", ...(remote.has_key ? {} : {key: key.value.trim()})}, go);
  };
  return form;
}

// The daemon starts again in place after a change of model: wait for it to go, then to answer.
function whenDaemonReturns(then, gone = false, tries = 0) {
  setTimeout(async () => {
    let up = true;
    try { await api("/api/setup"); } catch { up = false; }
    if ((gone || tries > 12) && up) { then(); return; }
    whenDaemonReturns(then, gone || !up, tries + 1);
  }, 1000);
}

// ---- first run: no model yet ----------------------------------------------------------
// What `coderunner setup` does in a terminal, from here: the plan for this computer and what it downloads, then
// setup itself with its progress. The daemon starts again with the model when setup is done.

const gb = (n) => n == null ? "size not known" : n < 0.1 ? `${Math.round(n * 1000)} MB` : `${n >= 100 ? n.toFixed(0) : n.toFixed(1).replace(/\.0$/, "")} GB`;
let setupTimer = null;
let setupDone = 0;

// Behind the setup screens: the owner's own things, drawn small in hairlines and laid at three depths, as if on a
// desk seen from above. A document with the one line that was looked for, a web page, a day in the calendar, a
// voice memo, a conversation, a sheet, a photo, a letter. The near ones are sharp and at the edges, the far ones
// small and out of focus; all drift slowly and lean a little with the pointer. The middle, where the words are,
// is left clear. Each drawing: [width, height, the lines, what is marked in the brand colour].
const SHEETS = {
  doc: [60, 76, "M12 16h22 M12 27h36 M12 49h36 M12 60h20", "M12 38h30"],
  page: [84, 60, "M0 14h84 M9 7h.01 M15 7h.01 M21 7h.01 M34 7h38 M10 24h28v24h-28z M46 26h28 M46 34h22", "M46 46h16"],
  day: [64, 64, "M0 16h64 M18 5v6 M46 5v6 M14 28h.01 M26 28h.01 M38 28h.01 M50 28h.01 M14 40h.01 M26 40h.01 M50 40h.01 M14 52h.01 M26 52h.01 M38 52h.01", "M38 40h.01"],
  wave: [84, 44, "M12 18v8 M20 12v20 M28 16v12 M44 14v16 M52 10v24 M60 17v10 M68 14v16", "M36 8v28"],
  chat: [76, 60, "M10 10h34v15h-34z M16 17.5h20", "M34 35h32v15h-32z"],
  sheet: [76, 60, "M0 15h76 M0 30h76 M0 45h76 M25 0v60 M50 0v60", "M30 22.5h14"],
  photo: [76, 60, "M8 48l18-22l12 14l8-8l22 16 M56 13a5 5 0 1 0 .01 0", ""],
  mail: [76, 54, "M0 0l38 27l38-27", ""],
};
// [layer, left %, top %, width px, tilt deg, drawing]
const FLOATS = [
  ["near", 6.5, 15, 104, -9, "doc"], ["near", 84, 9, 132, 7, "page"], ["near", 8, 70, 96, 8, "day"], ["near", 85, 73, 116, -6, "chat"],
  ["mid", 19.5, 7, 66, 12, "wave"], ["mid", 78.5, 40, 54, -12, "sheet"], ["mid", 17, 47, 50, -6, "photo"], ["mid", 78, 91, 58, 9, "mail"],
  ["mid", 3, 45, 38, 16, "doc"], ["mid", 94.5, 44, 44, -10, "day"],
  ["far", 14, 35, 20, 20, "doc"], ["far", 40, 2, 22, -12, "sheet"], ["far", 62, 4, 22, 26, "doc"], ["far", 34, 95, 24, -18, "page"],
  ["far", 59, 96, 18, 8, "doc"], ["far", 2, 6, 24, -22, "mail"], ["far", 97, 28, 18, 14, "doc"], ["far", 3, 92, 20, 30, "photo"],
  ["far", 96, 94, 22, -10, "sheet"], ["far", 16, 64, 16, 12, "doc"], ["far", 84, 57, 20, -26, "wave"], ["far", 73, 24, 14, 16, "doc"],
];

function floatField() {
  const NS = "http://www.w3.org/2000/svg";
  const field = el("div", "float-bg"), layers = {};
  field.setAttribute("aria-hidden", "true");
  for (const name of ["far", "mid", "near"]) field.appendChild(layers[name] = el("div", `float-layer ${name}`));
  FLOATS.forEach(([layer, left, top, width, tilt, kind], i) => {
    const [w, h, lines, mark] = SHEETS[kind], sheet = el("div", "ft"), svg = document.createElementNS(NS, "svg");
    sheet.style.cssText = `left:${left}%;top:${top}%;width:${width}px;--r:${tilt}deg;--dr:${i % 2 ? 4 : -4}deg;` +
      `--dx:${(i % 3 - 1) * 6}px;--dy:${i % 2 ? 12 : -10}px;--d:${17 + (i * 7) % 12}s;--delay:${-((i * 5) % 17)}s`;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    for (const [d, cls] of [[lines, "ink"], [mark, "mark"]]) {
      if (!d) continue;
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("class", cls);
      svg.appendChild(path);
    }
    sheet.appendChild(svg);
    layers[layer].appendChild(sheet);
  });
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    addEventListener("pointermove", (event) => {
      field.style.setProperty("--px", (event.clientX / innerWidth - 0.5).toFixed(3));
      field.style.setProperty("--py", (event.clientY / innerHeight - 0.5).toFixed(3));
    }, {passive: true});
  }
  return field;
}

// The first-run screens have the page to themselves: no task list, no computer panel, one thing to read.
function setupCard(title, lead) {
  state.setup = true;
  const card = el("section", "setup"), logo = el("span", "logo setup-logo");
  logo.setAttribute("aria-hidden", "true");
  card.append(logo, el("h1", "", title));
  if (lead) card.appendChild(el("p", "setup-lead", lead));
  $("thread").replaceChildren(card);
  $("composer").hidden = true;
  $("title").textContent = "";
  $("shell").classList.add("first-run");
  if (!$("main").querySelector(".float-bg")) $("main").prepend(floatField());
  showComputer(false);            // nothing runs on the computer until there is a model
  return card;
}

function setupButton(label, action) {
  const button = el("button", "btn primary setup-go", label);
  button.onclick = () => { button.disabled = true; action(); };
  return button;
}

// A small line icon, drawn with strokes on a 24 by 24 grid.
function icon(d) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg"), path = document.createElementNS(NS, "path");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "ico");
  svg.setAttribute("aria-hidden", "true");
  path.setAttribute("d", d);
  svg.appendChild(path);
  return svg;
}

const ROLE_ICONS = {
  planner: "M9 7h10 M9 12h10 M9 17h10 M5 7h.01 M5 12h.01 M5 17h.01",
  eyes: "M3 12s3.5-6 9-6s9 6 9 6s-3.5 6-9 6s-9-6-9-6z M12 9.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 1 0 0-5",
  search: "M11 5a6 6 0 1 0 0 12a6 6 0 1 0 0-12 M20 20l-4.6-4.6",
  voice: "M6 10v4 M9 7v10 M12 5v14 M15 8v8 M18 10v4",
  checker: "M12 4l7 3v5c0 4-3 7-7 8c-4-1-7-4-7-8V7z M9 12l2 2l4-4",
};

// The planning model's name as a button that lists the models that fit, best first: how long a step takes with
// each, and what it would still download. Choosing one asks for the plan again, made for it.
function modelPicker(plan, choose) {
  const box = el("span", "role-model pick"), button = el("button", "pick-button"), menu = el("div", "pick-menu");
  button.type = "button";
  button.append(el("span", "", plan.model.name), icon("M7 10l5 5l5-5"));
  button.setAttribute("aria-haspopup", "listbox");
  menu.setAttribute("role", "listbox");
  menu.hidden = true;
  for (const choice of plan.choices) {
    const item = el("button", "pick-item" + (choice.id === plan.model.id ? " on" : ""));
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(choice.id === plan.model.id));
    const name = el("span", "pick-name", choice.name);
    if (choice.id === plan.recommended) name.appendChild(el("i", "", "Recommended"));
    item.append(name, el("span", "pick-meta", `~${Math.round(choice.step_s)} s a step`),
                el("span", "pick-meta", choice.download_gb === 0 ? "Installed" : gb(choice.download_gb)));
    item.onclick = () => { menu.hidden = true; if (choice.id !== plan.model.id) { button.disabled = true; choose(choice.id); } };
    menu.appendChild(item);
  }
  const close = (event) => {
    if (box.contains(event.target)) return;
    menu.hidden = true;
    button.setAttribute("aria-expanded", "false");
    document.removeEventListener("click", close);
  };
  button.onclick = () => {
    menu.hidden = !menu.hidden;
    button.setAttribute("aria-expanded", String(!menu.hidden));
    if (!menu.hidden) document.addEventListener("click", close);
  };
  box.onkeydown = (event) => { if (event.key === "Escape") { menu.hidden = true; button.focus(); } };
  box.append(button, menu);
  return box;
}

async function showPlan() {
  const card = setupCard("Set up your assistant", "Looking at this computer and the models that suit it…");
  const bar = el("div", "bar busy looking");
  bar.appendChild(el("span"));
  card.appendChild(bar);
  try { renderPlan(await api("/api/setup", {method: "POST", body: JSON.stringify({action: "plan"})})); }
  catch (err) { bar.remove(); card.append(el("p", "setup-error", err.message), setupButton("Try again", showPlan)); }
}

// The one screen before anything is fetched: this computer in four figures, a model for each job, the space it
// takes and where, and one button. How it was all chosen is a click away, not on the screen.
function renderPlan(plan, problem) {
  const hw = plan.hardware, speed = plan.speed, here = hw.os === "mac" ? "this Mac" : "this computer";
  const card = setupCard("Set up your assistant", `Private AI that runs on ${here}. Nothing you ask leaves it.`);

  const tiles = el("div", "tiles");
  const tile = (value, label, hint) => {
    const box = el("div", "tile");
    box.append(el("div", "tile-value", value), el("div", "tile-label", label));
    box.title = hint || value;
    tiles.appendChild(box);
  };
  tile(hw.chip, "Chip");
  tile(`${hw.ram_gb} GB`, "Memory", hw.budget_gb ? `${hw.budget_gb} GB of it can go to models` : "");
  tile(hw.gpu === "apple" ? "Apple GPU" : hw.gpu === "none" ? "None" : `${hw.gpu === "nvidia" ? "NVIDIA" : "AMD"} ${hw.vram_gb} GB`, "Graphics",
       hw.gpu === "apple" ? "Apple GPU and Neural Engine, sharing the memory" : hw.gpu === "none" ? "Models will run on the processor, slowly" : "");
  if (speed) tile(`~${Math.round(speed.step_s)} s`, speed.measured ? "Per step, measured" : "Per step, estimated",
                  "How long each step of a task takes here");
  card.appendChild(tiles);
  if (!plan.model) {
    card.append(el("p", "setup-error", plan.notes.join(" ") || "No model fits this computer."), setupButton("Try again", showPlan));
    return;
  }

  const own = plan.recommended && plan.model.id !== plan.recommended;      // the owner chose another model
  const replan = async (extra) => {
    try { renderPlan(await api("/api/setup", {method: "POST", body: JSON.stringify({model: plan.model.id, ...extra})})); }
    catch (err) { renderPlan(plan, err.message); }
  };
  card.appendChild(el("div", "setup-label", own ? `Your choice for ${here}` : `Recommended for ${here}`));
  const roles = el("div", "roles");
  for (const role of plan.roles || []) {
    const row = el("div", "role" + (role.shared ? " shared" : "") + (role.missing ? " missing" : ""));
    row.title = [role.job, role.on].filter(Boolean).join(" · ");
    let model = el("span", "role-model", role.shared ? "Same model" : role.model);
    if (role.key === "planner" && (plan.choices || []).length > 1) model = modelPicker(plan, (id) => replan({action: "plan", model: id}));
    row.append(icon(ROLE_ICONS[role.key] || ROLE_ICONS.planner), el("span", "role-job", role.short || role.job), model,
               el("span", "role-size", role.shared || role.missing ? "" : role.download_gb ? gb(role.download_gb) : "Installed"));
    roles.appendChild(row);
  }
  card.appendChild(roles);

  // The free space as one bar, with what this takes of it. Then where it all goes.
  const disk = el("div", "disk"), line = el("div", "disk-line"), sizes = el("span", "disk-sizes");
  const tight = plan.download_gb > plan.room_gb;          // it would take the disk under what is always left free
  if (plan.download_gb && plan.free_gb > 0) {
    const bar = el("div", "diskbar" + (tight ? " tight" : "")), take = el("span", "take");
    take.style.width = `${Math.min(plan.download_gb / plan.free_gb * 100, 100)}%`;
    bar.appendChild(take);
    bar.title = `${gb(plan.free_gb)} free` + (plan.disk_gb ? ` of ${gb(plan.disk_gb)}` : "");
    disk.appendChild(bar);
  }
  sizes.append(el("i", "dot" + (tight ? " tight" : "")),
               el("b", "", plan.downloads.length ? `${gb(plan.download_gb)} to download` : "Nothing to download"),
               el("span", "muted", tight ? ` · ${gb(plan.free_gb)} free` : ` · ${gb(Math.max(plan.after_gb, 0))} free afterwards`));
  const place = el("span", "place"), change = el("button", "linkish", "Change"), standard = el("button", "linkish", "Reset");
  change.type = standard.type = "button";
  standard.title = "Use the usual folder";
  const path = el("span", "place-path", plan.models_dir_short || plan.models_dir);
  path.title = plan.models_dir;
  place.append(path, change);
  if (!plan.models_dir_default) place.appendChild(standard);
  line.append(sizes, place);
  disk.appendChild(line);
  const form = el("form", "place-form"), box = el("input"), save = el("button", "btn", "Keep them here");
  box.type = "text"; box.value = plan.models_dir; box.setAttribute("aria-label", "Folder for the models"); box.spellcheck = false;
  form.append(box, save);
  form.hidden = true;
  const move = (path) => {
    change.disabled = save.disabled = standard.disabled = true;
    return replan({action: "location", path});
  };
  change.onclick = async () => {
    if (window.coderunnerShell?.pickFolder) {
      const picked = await window.coderunnerShell.pickFolder(plan.models_dir);
      if (picked) move(picked);
    } else { form.hidden = !form.hidden; if (!form.hidden) box.focus(); }
  };
  form.onsubmit = (event) => { event.preventDefault(); move(box.value); };
  standard.onclick = () => move("");
  disk.appendChild(form);
  if (problem) disk.appendChild(el("p", "setup-error", problem));
  if (tight) {
    disk.appendChild(el("p", "setup-error", `Not enough room here: ${gb(plan.floor_gb)} is always left free. ` +
                                            `Free ${gb(plan.download_gb - plan.room_gb)} or keep the models on another disk.`));
  }
  card.appendChild(disk);

  if (plan.memory?.short) {
    card.appendChild(el("p", "setup-note", `${plan.model.name} needs ${gb(plan.memory.need_gb)} of memory and ` +
                                           `${gb(plan.memory.free_gb)} is free right now. Close a few apps before you start.`));
  }
  for (const note of plan.notes) card.appendChild(el("p", "setup-note", note));
  const go = setupButton(plan.downloads.length ? "Download and set up" : "Set up", async () => {
    await api("/api/setup", {method: "POST", body: JSON.stringify({action: "start", model: own ? plan.model.id : ""})})
      .catch(() => {});                                                                             // 409: one is running
    watchSetup();
  });
  go.disabled = tight;
  card.appendChild(go);

  const why = el("details", "setup-why"), list = el("dl", "why");
  const say = (name, text) => list.append(el("dt", "", name), el("dd", "", text));
  say(plan.model.name, plan.model.reasons.join(" · "));
  if (speed) {
    const rates = `reads ${Math.round(speed.read_tok_s)} and writes ${Math.round(speed.tok_s)} tokens a second`;
    const tasks = speed.tasks ? ` It finished ${speed.tasks.passed} of ${speed.tasks.total} sample tasks.` : "";
    say("Speed", speed.measured ? `Measured here${speed.when ? ` (${speed.when})` : ""}: it ${rates}.${tasks}` :
        `Estimated from this chip's memory speed (${Math.round(speed.bandwidth_gbs)} GB/s): it ${rates}. ` +
        "Setup measures it for real, and picks a smaller model if this one is too slow.");
  }
  if (hw.budget_gb) say("Memory", `${hw.budget_gb} of ${hw.ram_gb} GB can go to models; the rest stays with the system and your apps.`);
  const others = (plan.alternatives || []).map((a) => `${a.name} (${Math.round(a.step_s)} s a step)`).join(", ");
  if (others) say("Also fit", others);
  if (plan.downloads.length) say("Downloads", plan.downloads.map((d) => `${d.label || d.what} ${gb(d.gb)}`).join(" · "));
  say("Disk", `${gb(plan.free_gb)} free now; ${plan.floor_gb} GB is always left free.`);
  why.append(el("summary", "", "How these were chosen"), list);
  why.addEventListener("toggle", () => { if (why.open) why.scrollIntoView({behavior: "smooth", block: "end"}); });
  card.appendChild(why);
  const other = el("button", "linkish setup-other", "Use an Anthropic key instead");
  other.type = "button";
  other.onclick = () => renderRemoteSetup(plan);
  card.appendChild(other);
}

// The other way to start: Claude on Anthropic's API with the owner's key. Nothing to download, and not private,
// which this screen says in place of the promise the local one makes.
async function renderRemoteSetup(plan, problem) {
  const here = plan.hardware.os === "mac" ? "this Mac" : "this computer";
  const card = setupCard("Use Claude with your key", "Nothing to download. Tasks take a second or two a step.");
  let remote;
  try { remote = await api("/api/remote"); } catch (err) { card.append(el("p", "setup-error", err.message), setupButton("Back", () => renderPlan(plan))); return; }
  const model = modelName(remote.models[0].id);
  card.appendChild(el("div", "setup-label", "With an Anthropic key"));
  const roles = el("div", "roles");
  for (const [key, job, name, size] of [["planner", "Planning", model, "Anthropic"], ["eyes", "Vision", "Same model", ""],
                                         ["checker", "Safety check", "Same model", ""],
                                         ["search", "File search", `On ${here}`, "Set up later"], ["voice", "Voice", `On ${here}`, "Set up later"]]) {
    const row = el("div", "role" + (name === model ? "" : " shared"));
    row.append(icon(ROLE_ICONS[key]), el("span", "role-job", job), el("span", "role-model", name), el("span", "role-size", size));
    roles.appendChild(row);
  }
  card.appendChild(roles);
  card.appendChild(el("p", "setup-note", remote.warning));
  const form = el("form", "place-form remote-key"), key = el("input");
  key.type = "password"; key.placeholder = "Anthropic API key"; key.autocomplete = "off"; key.spellcheck = false;
  key.setAttribute("aria-label", "Anthropic API key");
  if (!remote.has_key) form.appendChild(key);
  card.appendChild(form);
  if (problem) card.appendChild(el("p", "setup-error", problem));
  const start = async () => {
    if (!remote.has_key && !key.value.trim()) { go.disabled = false; key.focus(); return; }
    try { await api("/api/remote", {method: "POST", body: JSON.stringify({action: "on", ...(remote.has_key ? {} : {key: key.value.trim()})})}); }
    catch (err) { renderRemoteSetup(plan, err.message); return; }
    go.textContent = "Starting…";
    whenDaemonReturns(() => location.reload());
  };
  const go = setupButton("Use Claude", start);
  form.onsubmit = (event) => { event.preventDefault(); go.click(); };
  card.appendChild(go);
  const back = el("button", "linkish setup-other", `Run a model on ${here} instead`);
  back.type = "button";
  back.onclick = () => renderPlan(plan);
  card.appendChild(back);
  key.focus();
}

// How long a download has left, from how its percentage has moved: [[seconds, percent], ...] at each change,
// oldest first, and the time now. Nothing until it has moved for a while, so the first guess is not a wild
// one; a download that stalls is given longer as it stalls.
function timeLeft(seen, now) {
  if (seen.length < 2) return "";
  const [t0, p0] = seen[0], p1 = seen[seen.length - 1][1];
  if (p1 >= 100 || p1 - p0 < 2 || now - t0 < 20) return "";
  const seconds = (100 - p1) * (now - t0) / (p1 - p0);
  if (seconds < 50) return "less than a minute left";
  if (seconds < 3300) return `about ${Math.max(1, Math.round(seconds / 60))} min left`;
  const hours = seconds / 3600;
  if (hours >= 24) return "more than a day left";
  return `about ${hours < 9.5 ? Math.round(hours * 2) / 2 : Math.round(hours)} hr left`;
}

// The stages of a setup, one row each, kept and filled in as setup goes on (so a bar moves instead of being
// drawn again): done, running (with a bar, counted while something downloads) or failed.
function showStages(box, job) {
  const rows = box.rows || (box.rows = new Map());
  for (const stage of job.stages || []) {
    let r = rows.get(stage.key);
    if (!r) {
      r = {row: el("div", "stage"), mark: el("span", "stage-mark"), label: el("div", "stage-label"), amount: el("span"),
           left: el("span"), meta: el("div", "stage-meta"), bar: el("div", "bar"), fill: el("span"),
           detail: el("div", "muted small"), seen: []};
      const body = el("div", "stage-body");
      r.meta.append(r.amount, r.left);
      r.bar.appendChild(r.fill);
      body.append(r.label, r.bar, r.meta, r.detail);
      r.row.append(r.mark, body);
      box.appendChild(r.row);
      rows.set(stage.key, r);
    }
    const running = stage.state === "running", counted = running && stage.percent != null;
    r.row.className = "stage " + stage.state;
    r.mark.textContent = stage.state === "done" ? "✓" : stage.state === "failed" ? "!" : "";
    if (r.label.textContent !== stage.label) { r.label.textContent = stage.label; r.seen = []; }   // another download
    let left = "";
    if (counted) {
      const now = Date.now() / 1000, last = r.seen[r.seen.length - 1];
      if (last && stage.percent < last[1]) r.seen = [];
      if (!last || stage.percent !== last[1]) r.seen.push([now, stage.percent]);
      while (r.seen.length > 2 && now - r.seen[1][0] > 180) r.seen.shift();     // the last three minutes' pace
      left = timeLeft(r.seen, now);
    }
    r.amount.textContent = counted ? `${stage.percent}%` + (stage.of_gb ? ` of ${gb(stage.of_gb)}` : "") : "";
    r.left.textContent = left;
    r.meta.hidden = !counted;
    r.bar.hidden = !running;
    r.bar.classList.toggle("busy", running && !counted);
    r.fill.style.width = counted ? `${stage.percent}%` : "";
    r.detail.hidden = true;                         // setup's own last line is under Details, not here
  }
  box.hidden = !rows.size;
}

// While setup runs: the app itself, in small. A request is typed and sent, the assistant says what it is doing,
// what it found and its answer come in as they would in a real task, and the next request slides over. What it
// shows is what the assistant is for; none of it is said in a caption.
const DEMOS = [
  {title: "Your files", icon: "M8 4h6l4 4v12H8z M14 4v4h4", ask: "When does my lease end?", doing: ["Searching your files", "Reading Lease-2024.pdf"],
   show: [{file: "Lease-2024.pdf", where: "Documents › Home"}],
   answer: "It ends on **31 March 2027**. Notice is two months, so tell them by 31 January."},
  {title: "The web", icon: "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16 M4 12h16 M12 4c3 3 3 13 0 16 M12 4c-3 3-3 13 0 16",
   ask: "Cheapest Pixel 8 on Amazon or Flipkart?", doing: ["Opening amazon.in", "Opening flipkart.com"],
   show: [{page: "flipkart.com", rows: [["Flipkart", "₹47,999", "4.8★"], ["Amazon.in", "₹49,999", "4.6★"]]}],
   answer: "**Flipkart**, by ₹2,000. The same 128 GB model, a year's warranty on both."},
  {title: "Asks before it sends", icon: "M12 4l7 3v5c0 4-3 7-7 8c-4-1-7-4-7-8V7z M9 12l2 2l4-4",
   ask: "Email Sam that I'll be ten minutes late", doing: ["Writing the email"],
   show: [{approve: "Send this to Sam Lee?", text: "Running ten minutes late, please start without me."}],
   answer: "Sent to Sam."},
  {title: "Your day", icon: "M5 7h14v12H5z M5 10h14 M9 5v3 M15 5v3", ask: "What's on tomorrow?", doing: ["Checking your calendar"],
   show: [{list: [["10:00", "Design review"], ["13:00", "Lunch with Priya"], ["17:30", "Dentist"]]}],
   answer: "Three things. The dentist runs into your 5 pm call: shall I move the call?"},
  {title: "By voice", icon: "M6 10v4 M9 7v10 M12 5v14 M15 8v8 M18 10v4", voice: true, ask: "Remind me to call mum at six", doing: ["Setting a reminder"], show: [],
   answer: "Done. I'll remind you at **6:00 pm**."},
];

function demoItem(item) {
  if (item.file) {
    const row = el("div", "demo-file");
    const text = el("div", "");
    text.append(el("div", "demo-strong", item.file), el("div", "muted", item.where));
    row.append(icon("M8 4h6l4 4v12H8z M14 4v4h4"), text);
    return row;
  }
  const table = (rows) => {
    const made = el("table", "md-table demo-table");
    rows.forEach((cells, i) => {
      const row = made.insertRow();
      if (i === 0) row.className = "best";
      for (const cell of cells) row.insertCell().textContent = cell;
    });
    return made;
  };
  if (item.page) {            // a page's picture, as a task's thread carries one, beside what was read off it
    const row = el("div", "demo-page"), figure = el("div", ""), sheet = el("div", "demo-sheet");
    sheet.append(el("i", "bar-a"), el("i", "tile"), el("i", "tile"), el("i", "tile"), el("i", "line"), el("i", "line short"));
    figure.append(sheet, el("div", "demo-host muted", item.page));
    row.append(figure, table(item.rows));
    return row;
  }
  if (item.list) {
    const list = el("div", "demo-list");
    for (const [time, what] of item.list) { const row = el("div", ""); row.append(el("span", "muted", time), el("span", "", what)); list.appendChild(row); }
    return list;
  }
  const card = el("div", "ask demo-ask"), row = el("div", "row");
  row.append(el("span", "btn primary", "Allow"), el("span", "btn", "Deny"));
  card.append(el("div", "ask-title", item.approve), el("div", "muted", `“${item.text}”`), row);
  return card;
}

function showcase() {
  const wrap = el("div", "showcase"), box = el("div", "demo"), stage = el("div", "demo-stage");
  const composer = el("div", "demo-composer"), typed = el("span", "demo-typed"), send = el("span", "demo-send", "↑");
  const mic = el("span", "demo-mic"), dots = el("div", "demo-dots");
  for (let i = 0; i < 4; i++) mic.appendChild(el("i"));
  composer.append(mic, typed, el("span", "demo-caret"), send);
  DEMOS.forEach(() => dots.appendChild(el("i")));
  const bar = el("div", "demo-bar"), label = el("span", "demo-label");
  bar.appendChild(label);
  box.append(bar, stage, composer);
  wrap.append(box, dots);
  wrap.setAttribute("aria-hidden", "true");       // decoration: setup's progress above is what is read out
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const add = (panel, node) => { if (!calm) node.classList.add("demo-in"); panel.appendChild(node); return node; };

  // The answer comes in word by word; **bold** is made as it arrives, not shown as asterisks first.
  async function stream(node, text) {
    const parts = text.split("**");
    for (let i = 0; i < parts.length; i++) {
      const into = i % 2 ? node.appendChild(el("strong")) : node;
      for (const word of parts[i].split(/(?<= )/)) {
        if (!word) continue;
        into.appendChild(document.createTextNode(word));
        if (!calm) await wait(55);
      }
    }
  }

  async function play(index) {
    const demo = DEMOS[index], panel = el("div", "demo-panel enter"), before = stage.firstElementChild;
    [...dots.children].forEach((dot, i) => dot.classList.toggle("on", i === index));
    label.replaceChildren(icon(demo.icon), el("span", "", demo.title));
    stage.appendChild(panel);
    if (before) { before.classList.add("leave"); setTimeout(() => before.remove(), 650); }
    panel.getBoundingClientRect();                 // placed off to the right first, so that it slides in
    panel.classList.remove("enter");
    composer.classList.toggle("voice", Boolean(demo.voice));
    typed.textContent = "";
    await wait(calm ? 200 : 500);
    for (const piece of demo.voice ? demo.ask.split(/(?<= )/) : demo.ask) {       // spoken: a word at a time
      typed.textContent += piece;
      if (!calm) await wait(demo.voice ? 190 : 24 + Math.random() * 34);
      if (!wrap.isConnected) return;
    }
    await wait(420);
    send.classList.add("go");
    await wait(170);
    send.classList.remove("go");
    typed.textContent = "";
    composer.classList.remove("voice");
    add(panel, el("div", "bubble-user", demo.ask));
    await wait(520);
    const now = add(panel, el("div", "now"));
    now.append(el("span", "now-dot"), el("span", "now-what"));
    for (const doing of demo.doing) { now.lastChild.textContent = `${doing}…`; await wait(1150); }
    for (const item of demo.show) {
      const node = add(panel, demoItem(item));
      panel.insertBefore(node, now);                // what is found goes above the line that says what is happening
      await wait(850);
      if (item.approve) {                           // the owner's yes, before anything is sent
        now.lastChild.textContent = "Waiting for you…";
        await wait(900);
        node.querySelector(".btn.primary").classList.add("pressed");
        await wait(380);
        node.classList.add("closed");
        node.querySelector(".row").replaceChildren(el("span", "muted", "Allowed"));
        await wait(350);
      }
    }
    now.remove();
    await stream(add(panel, el("div", "answer")), demo.answer);
    await wait(3600);
  }

  (async () => {
    await wait(80);                                 // the card is on the page by now
    for (let i = 0; wrap.isConnected; i = (i + 1) % DEMOS.length) await play(i);
  })();
  return wrap;
}

// While setup runs the same card is kept and only what changed is filled in, so what is on it does not flicker.
let progressView = null;

function renderProgress(job) {
  const running = job.state === "running";
  if (running && progressView?.card.isConnected) {
    showStages(progressView.stages, job);
    progressView.log.replaceChildren(...job.lines.map((line) => el("div", "", line)));
    progressView.details.hidden = !job.lines.length;
    if (progressView.details.open) progressView.log.scrollTop = progressView.log.scrollHeight;
    return;
  }
  const license = job.state === "failed" ? job.result?.needs_license : null;
  if (license) { progressView = null; renderLicense(license); return; }
  const short = job.state === "failed" ? job.result?.memory_short : null;
  if (short) { progressView = null; renderMemory(short, job.result.next); return; }
  const card = setupCard(running ? "Setting up your assistant" : job.state === "done" ? "Ready" : "Setup did not finish",
                         running ? "This takes a while the first time. It keeps going if you close the window." :
                         job.state === "done" ? "Starting the assistant with its model…" : "");
  const stages = el("div", "stages");
  showStages(stages, job);
  // What setup itself printed, for whoever wants it; open when there are no stages to show instead.
  const details = el("details", "setup-why"), log = el("div", "setup-log");
  details.open = !stages.children.length && job.lines.length > 0;
  details.hidden = !job.lines.length;
  log.append(...job.lines.map((line) => el("div", "", line)));
  details.append(el("summary", "", "Details"), log);
  card.appendChild(stages);
  if (running) {
    card.append(showcase(), details);
    progressView = {card, stages, details, log};
    return;
  }
  progressView = null;
  if (job.state === "failed") {
    // What went wrong, once, and the ways on: the same again, the next choice where setup named one, or back.
    const again = (model) => async () => {
      await api("/api/setup", {method: "POST", body: JSON.stringify({action: "start", ...(model ? {model} : {})})}).catch(() => {});
      watchSetup();
    };
    const next = job.result?.next, back = el("button", "linkish setup-other", "Back");
    back.type = "button";
    back.onclick = showPlan;
    card.append(el("p", "setup-error", job.result?.error || "Setup failed."), setupButton("Try again", again(null)));
    if (next) {
      const other = el("button", "btn setup-next", `Use ${next.name || modelName(next.model)}` +
                                                   (next.download_gb ? ` (${gb(next.download_gb)} download)` : ""));
      other.onclick = () => { other.disabled = true; again(next.model)(); };
      card.appendChild(other);
    }
    card.append(el("div", "", ""), back);
  }
  card.appendChild(details);
}

// Setup stopped at a model whose licence asks to be agreed to: say which and why, link to it, and go on from
// here on a yes.
function renderLicense(license) {
  const card = setupCard("One thing to agree to", `${modelName(license.model)} comes with a licence that asks for your agreement.`);
  const box = el("div", "roles license");
  const reason = (license.reason || "").replace(/;?\s*review \S+ and accept it once/, "");
  box.appendChild(el("div", "", reason ? `${reason}.` : "Its licence has terms to accept before it is downloaded."));
  if (/^https?:\/\//.test(license.link || "")) {
    const link = el("a", "linkish", "Read the licence");
    link.href = license.link; link.target = "_blank"; link.rel = "noopener";
    box.appendChild(link);
  }
  card.appendChild(box);
  const agree = setupButton("Agree and continue", async () => {
    await api("/api/setup", {method: "POST", body: JSON.stringify({action: "start", model: license.model, accept_license: true})}).catch(() => {});
    watchSetup();
  });
  const other = el("button", "linkish setup-other", "Back");
  other.type = "button";
  other.onclick = showPlan;
  card.append(agree, el("div", "", ""), other);
}

// The model suits this computer but not this moment: other apps hold the memory it needs. Say how much, and
// offer the two ways on: make room and try again, or take the smaller model that fits now.
function renderMemory(short, next) {
  const card = setupCard("Not enough memory free right now",
                         `${modelName(short.model)} needs ${gb(short.need_gb)} and ${gb(short.free_gb)} is free. Close a few apps, then try again.`);
  const start = (model) => async () => {
    await api("/api/setup", {method: "POST", body: JSON.stringify({action: "start", ...(model ? {model} : {})})}).catch(() => {});
    watchSetup();
  };
  const bar = el("div", "diskbar membar"), have = el("span", "take");
  have.style.width = `${Math.min(short.free_gb / short.need_gb * 100, 100)}%`;
  bar.appendChild(have);
  bar.title = `${gb(short.free_gb)} free of the ${gb(short.need_gb)} it needs`;
  card.append(bar, setupButton("Try again", start(null)));
  if (next) {
    const other = el("button", "linkish setup-other", `Use ${next.name || modelName(next.model)} instead` +
                                                      (next.download_gb ? ` (${gb(next.download_gb)} download)` : ""));
    other.type = "button";
    other.onclick = () => { other.disabled = true; start(next.model)(); };
    card.append(el("div", "", ""), other);
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
