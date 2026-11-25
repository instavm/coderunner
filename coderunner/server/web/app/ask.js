// The quick ask in the desktop app (a global shortcut): one line, Return starts the task in a new thread and the
// window goes away; Escape puts it away. Files chosen or dropped go with the task, uploaded as in the app
// (/api/uploads). The menu bar icon says the task is working; there is nothing more to confirm here.
import {el} from "./render.js";

const $ = (id) => document.getElementById(id);
const shell = window.coderunnerShell || {};
const HEADERS = {"Content-Type": "application/json", "X-CodeRunner": "1"};
if (!window.coderunnerShell || !/Mac/i.test(navigator.platform || "")) document.documentElement.classList.add("opaque");
const CLIP = "M20.5 11.5l-8.2 8.2a5 5 0 0 1-7.1-7.1l8.5-8.5a3.4 3.4 0 0 1 4.8 4.8l-8.4 8.4a1.8 1.8 0 0 1-2.6-2.6l7.6-7.6";
const MAX_ATTACHED = 8;

const ARROW = "M12 19V5 M5.5 11.5L12 5l6.5 6.5";
function icon(d) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), line = document.createElementNS(svg.namespaceURI, "path");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  line.setAttribute("d", d);
  svg.appendChild(line);
  return svg;
}
$("attach").appendChild(icon(CLIP));
$("send").appendChild(icon(ARROW));

// The button is there to press only when there is something to send and every file is in.
function ready() {
  $("send").disabled = (!$("askInput").value.trim() && !files.length) || files.some((file) => !file.id || file.failed);
}
$("askInput").addEventListener("input", ready);

let noteTimer = null;
function note(text) {
  $("askNote").textContent = text;
  $("askNote").title = text;
  $("askNote").hidden = !text;
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => { $("askNote").hidden = true; }, 6000);
}

// Files on their way: {name, id, failed, request}. Shown as one small pill: its name, or how many.
const files = [];
function drawFiles() {
  const pill = $("files");
  pill.hidden = !files.length;
  ready();
  if (!files.length) { pill.replaceChildren(); return; }
  const failed = files.find((file) => file.failed);
  pill.className = "quick-files" + (failed ? " failed" : files.some((file) => !file.id) ? " busy" : "");
  pill.title = failed ? `${failed.name}: ${failed.failed}` : files.map((file) => file.name).join("\n");
  const clear = el("button", "", "×");
  clear.type = "button"; clear.title = "Remove"; clear.setAttribute("aria-label", "Remove the files");
  clear.onclick = () => { for (const file of files.splice(0)) file.request?.abort(); drawFiles(); $("askInput").focus(); };
  pill.replaceChildren(el("span", "", files.length === 1 ? files[0].name : `${files.length} files`), clear);
}

function attach(chosen) {
  for (const given of chosen) {
    if (files.length >= MAX_ATTACHED) { note(`At most ${MAX_ATTACHED} files to a message.`); break; }
    const file = {name: given.name || "pasted file", id: "", failed: ""};
    const request = file.request = new XMLHttpRequest();
    request.open("POST", "/api/uploads");
    request.setRequestHeader("X-CodeRunner", "1");
    request.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    request.setRequestHeader("Content-Type", given.type || "application/octet-stream");
    request.onload = () => {
      let data = {};
      try { data = JSON.parse(request.responseText); } catch { /* not ours */ }
      if (request.status >= 300 || !data.id) file.failed = data.error || "It was not kept.";
      else file.id = data.id;
      drawFiles();
    };
    request.onerror = () => { file.failed = "The upload did not go through."; drawFiles(); };
    request.send(given);
    files.push(file);
  }
  drawFiles();
  $("askInput").focus();
}

$("attach").onclick = () => $("filePick").click();
$("filePick").onchange = () => { attach([...$("filePick").files]); $("filePick").value = ""; };
$("askInput").addEventListener("paste", (event) => {
  const pasted = [...(event.clipboardData?.files || [])];
  if (pasted.length) { event.preventDefault(); attach(pasted); }
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer?.files?.length) attach([...event.dataTransfer.files]);
});

$("ask").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("askInput"), message = input.value.trim();
  const failed = files.find((file) => file.failed);
  if (failed) { note(`${failed.name} was not attached. Remove it to go on.`); return; }
  if ((!message && !files.length) || files.some((file) => !file.id) || input.disabled) return;
  input.disabled = true;
  try {
    await api("/api/tasks", {method: "POST",
      body: JSON.stringify({message, conversation: "", attachments: files.map((file) => file.id)})});
    input.value = "";
    files.splice(0);
    drawFiles();
    note("");
    shell.taskStarted?.();
    shell.hidePanel?.();
  } catch (err) { note(err.message); }
  finally { input.disabled = false; input.focus(); }
};

async function api(path, options = {}) {
  const response = await fetch(path, {credentials: "same-origin", headers: HEADERS, ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "That did not work. Try again.");
  return data;
}

document.addEventListener("keydown", (event) => { if (event.key === "Escape") shell.hidePanel?.(); });
shell.onShown?.(() => { $("askInput").focus(); $("askInput").select(); });
