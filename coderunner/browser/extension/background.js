// CodeRunner: lets the local assistant use the tabs in its "CodeRunner" tab group, and no other tab.
//
// The assistant speaks the Chrome DevTools Protocol, relayed by a native messaging host
// (com.coderunner.browser, installed by `coderunner browser connect`). Commands for a tab go to
// chrome.debugger, but only for tabs in the CodeRunner group (and popups those tabs open in windows
// of their own, which cannot join a group). Browser-level Target methods are answered from the tabs
// and tabGroups APIs with those tabs alone, so the owner's other tabs are never listed, read or
// controlled. Dragging a tab out of the group takes it away from the assistant; dragging one in
// hands it over.
"use strict";

const HOST = "com.coderunner.browser";
const GROUP_TITLE = "CodeRunner";
const NONE = -1;   // chrome.tabGroups.TAB_GROUP_ID_NONE
// Protocol domains a tab session may use: what the assistant's broker needs. Nothing that reaches
// other tabs, the profile's cookies and storage, or the browser itself.
const DOMAINS = new Set(["Accessibility", "DOM", "Emulation", "Fetch", "Input", "Page", "Runtime"]);
const BLOCKED = new Set(["Page.setDownloadBehavior"]);   // the extension decides where downloads go
// Frames from other sites run in processes of their own: attach to them too, paused until their
// document requests pass the same check as the tab's.
const AUTO_ATTACH = {autoAttach: true, waitForDebuggerOnStart: true, flatten: true, filter: [{type: "iframe"}]};
// Let go of tabs the assistant has not used for a while, which also clears Chrome's "debugging" bar;
// the assistant attaches again on its next step.
const IDLE_MS = 5 * 60 * 1000;

let port = null;
let daemon = false;
let discover = false;
let fetchPatterns = null;
let denyDownloads = false;
let groupId = NONE;
const sessions = new Map();   // tabId -> "tab:<tabId>" while the debugger is attached
const children = new Map();   // child session id (out-of-process frame) -> tabId
const targets = new Map();    // tabId -> protocol target id, for the assistant's tabs
const openers = new Map();    // tabId -> target id of the tab that opened it
const adopted = new Set();    // popups of the assistant's tabs in windows that cannot hold a group
const used = new Map();       // tabId -> when the assistant last sent it a command

// ---- the native host ---------------------------------------------------------

function connect() {
  if (port) return;
  port = chrome.runtime.connectNative(HOST);
  port.onMessage.addListener(onMessage);
  port.onDisconnect.addListener(() => {
    void chrome.runtime.lastError;   // host not installed, or it exited
    port = null;
    setDaemon(false);
  });
}

function post(message) {
  if (!port) return;
  try {
    port.postMessage(message);
  } catch (e) {
    // the port closed; onDisconnect cleans up
  }
}

function emit(method, params, sessionId) {
  if (!daemon) return;
  const message = {method, params};
  if (sessionId) message.sessionId = sessionId;
  post(message);
}

function onMessage(message) {
  if (message && message.type === "daemon") {
    setDaemon(Boolean(message.connected));
  } else if (message && message.id !== undefined && message.method) {
    run(message);
  }
}

async function setDaemon(connected) {
  daemon = connected;
  chrome.action.setBadgeText({text: connected ? "on" : ""});
  chrome.action.setTitle({title: connected ? "CodeRunner: connected" : "CodeRunner: not connected"});
  if (connected) {
    post({type: "extension", version: chrome.runtime.getManifest().version});
  } else {
    discover = false;
    fetchPatterns = null;
    denyDownloads = false;
    await releaseAll();
  }
}

async function run(message) {
  const {id, method, params = {}, sessionId} = message;
  try {
    const result = sessionId ? await tabCommand(sessionId, method, params) : await browserCommand(method, params);
    post({id, result: result || {}});
  } catch (e) {
    post({id, error: {message: String((e && e.message) || e)}});
  }
}

// ---- which tabs are the assistant's ----------------------------------------------

async function group() {
  if (groupId !== NONE) {
    try {
      await chrome.tabGroups.get(groupId);
      return groupId;
    } catch (e) {
      groupId = NONE;
    }
  }
  const found = await chrome.tabGroups.query({title: GROUP_TITLE});
  groupId = found.length ? found[0].id : NONE;
  return groupId;
}

async function mine(tab) {
  if (!tab) return false;
  if (adopted.has(tab.id)) return true;
  return tab.groupId !== NONE && tab.groupId === await group();
}

async function ownTab(tabId) {
  try {
    const tab = await chrome.tabs.get(tabId);
    return (await mine(tab)) ? tab : null;
  } catch (e) {
    return null;
  }
}

async function ownTabs() {
  const id = await group();
  const tabs = id === NONE ? [] : await chrome.tabs.query({groupId: id});
  for (const tabId of adopted) {
    try {
      tabs.push(await chrome.tabs.get(tabId));
    } catch (e) {
      adopted.delete(tabId);
    }
  }
  return tabs;
}

async function targetOf(tabId) {
  if (targets.has(tabId)) return targets.get(tabId);
  for (let attempt = 0; attempt < 20; attempt++) {   // a brand-new tab gets its target a moment later
    const match = (await chrome.debugger.getTargets()).find((t) => t.tabId === tabId && t.type === "page");
    if (match) {
      targets.set(tabId, match.id);
      return match.id;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("No target with given id found");
}

async function tabOf(targetId) {
  for (const [tabId, id] of targets) {
    if (id === targetId && await ownTab(tabId)) return tabId;
  }
  const match = (await chrome.debugger.getTargets()).find((t) => t.id === targetId && t.type === "page");
  if (match && match.tabId !== undefined && await ownTab(match.tabId)) {
    targets.set(match.tabId, match.id);
    return match.tabId;
  }
  throw new Error("No target with given id found");
}

async function info(tab) {
  const result = {targetId: await targetOf(tab.id), type: "page", title: tab.title || "",
                  url: tab.url || tab.pendingUrl || "about:blank", attached: sessions.has(tab.id),
                  canAccessOpener: false, browserContextId: "default"};
  if (openers.has(tab.id)) result.openerId = openers.get(tab.id);
  return result;
}

async function announce(tab, method = "Target.targetCreated") {
  if (discover && daemon) emit(method, {targetInfo: await info(tab)});
}

async function forget(tabId) {
  const targetId = targets.get(tabId);
  await release(tabId);
  targets.delete(tabId);
  openers.delete(tabId);
  adopted.delete(tabId);
  if (targetId && discover) emit("Target.targetDestroyed", {targetId});
}

// ---- debugger sessions ------------------------------------------------------------

async function attach(tabId) {
  if (sessions.has(tabId)) return sessions.get(tabId);
  try {
    await chrome.debugger.attach({tabId}, "1.3");
  } catch (e) {
    // When Chrome restarts this worker, the debugger it had attached stays attached while this copy has
    // forgotten it, and every later attach to the tab failed ("Another debugger is already attached").
    // Detaching succeeds only for this extension's own attachment: then attach afresh. A debugger that is
    // someone else's (DevTools, another extension) is left alone and the error stands.
    try {
      await chrome.debugger.detach({tabId});
    } catch (_) {
      throw e;
    }
    await chrome.debugger.attach({tabId}, "1.3");
  }
  const sessionId = `tab:${tabId}`;
  sessions.set(tabId, sessionId);
  used.set(tabId, Date.now());
  try {
    if (fetchPatterns) await chrome.debugger.sendCommand({tabId}, "Fetch.enable", {patterns: fetchPatterns});
    if (denyDownloads) await denyDownloadsIn({tabId});
    await chrome.debugger.sendCommand({tabId}, "Target.setAutoAttach", AUTO_ATTACH);
  } catch (e) {
    await release(tabId);
    throw e;
  }
  return sessionId;
}

async function denyDownloadsIn(debuggee) {
  try {
    await chrome.debugger.sendCommand(debuggee, "Page.setDownloadBehavior", {behavior: "deny"});
  } catch (e) {
    // not available in this version: downloads then go where the owner's own downloads go
  }
}

async function release(tabId) {
  if (!sessions.has(tabId)) return;
  const sessionId = sessions.get(tabId);
  sessions.delete(tabId);
  used.delete(tabId);
  for (const [child, owner] of children) {
    if (owner === tabId) children.delete(child);
  }
  try {
    await chrome.debugger.detach({tabId});
  } catch (e) {
    // already detached, or the tab is gone
  }
  emit("Target.detachedFromTarget", {sessionId, targetId: targets.get(tabId)});
}

async function releaseAll() {
  for (const tabId of [...sessions.keys()]) await release(tabId);
}

async function adopt(source, params) {
  const child = {tabId: source.tabId, sessionId: params.sessionId};
  children.set(params.sessionId, source.tabId);
  try {
    if (fetchPatterns) await chrome.debugger.sendCommand(child, "Fetch.enable", {patterns: fetchPatterns});
    await chrome.debugger.sendCommand(child, "Target.setAutoAttach", AUTO_ATTACH);
  } catch (e) {
    // the frame went away
  }
  try {
    await chrome.debugger.sendCommand(child, "Runtime.runIfWaitingForDebugger");
  } catch (e) {
    // the frame went away
  }
}

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (!sessions.has(source.tabId)) return;
  if (method === "Target.attachedToTarget") {
    adopt(source, params);
  } else if (method === "Target.detachedFromTarget") {
    children.delete(params.sessionId);
  } else {
    emit(method, params, source.sessionId || sessions.get(source.tabId));
  }
});

setInterval(() => {
  const now = Date.now();
  for (const [tabId, time] of used) {
    if (now - time > IDLE_MS) release(tabId);
  }
}, 30 * 1000);

chrome.debugger.onDetach.addListener((source) => {
  const tabId = source.tabId;
  if (!sessions.has(tabId)) return;
  const sessionId = sessions.get(tabId);
  sessions.delete(tabId);
  used.delete(tabId);
  for (const [child, owner] of children) {
    if (owner === tabId) children.delete(child);
  }
  emit("Target.detachedFromTarget", {sessionId, targetId: targets.get(tabId)});
});

// ---- commands -----------------------------------------------------------------------

async function tabCommand(sessionId, method, params) {
  if (!DOMAINS.has(method.split(".")[0]) || BLOCKED.has(method)) {
    throw new Error(`${method} is not available to the assistant`);
  }
  if (method === "Page.navigate" && !/^(https?:|about:blank$)/i.test(String(params.url || ""))) {
    throw new Error("Only web pages can be opened");
  }
  let debuggee;
  if (sessionId.startsWith("tab:")) {
    const tabId = Number(sessionId.slice(4));
    if (sessions.get(tabId) !== sessionId) throw new Error("Session with given id not found");
    if (!await ownTab(tabId)) {
      await forget(tabId);
      throw new Error("Session with given id not found");
    }
    debuggee = {tabId};
  } else {
    const tabId = children.get(sessionId);
    if (tabId === undefined || !sessions.has(tabId)) throw new Error("Session with given id not found");
    debuggee = {tabId, sessionId};
  }
  used.set(debuggee.tabId, Date.now());
  return chrome.debugger.sendCommand(debuggee, method, params);
}

async function browserCommand(method, params) {
  switch (method) {
    case "Browser.getVersion":
      return {protocolVersion: "1.3", product: `CodeRunner extension ${chrome.runtime.getManifest().version}`,
              revision: "", userAgent: navigator.userAgent, jsVersion: ""};
    case "Browser.close":
      // The owner's browser stays open: the assistant only lets go of its tabs.
      await releaseAll();
      return {};
    case "Browser.setDownloadBehavior":
      denyDownloads = params.behavior === "deny";
      if (denyDownloads) for (const tabId of sessions.keys()) await denyDownloadsIn({tabId});
      return {};
    case "Target.setDiscoverTargets":
      discover = Boolean(params.discover);
      if (discover) for (const tab of await ownTabs()) await announce(tab);
      return {};
    case "Fetch.enable":
      fetchPatterns = params.patterns || [{urlPattern: "*"}];
      for (const tabId of sessions.keys()) {
        await chrome.debugger.sendCommand({tabId}, "Fetch.enable", {patterns: fetchPatterns});
      }
      for (const [sessionId, tabId] of children) {
        try {
          await chrome.debugger.sendCommand({tabId, sessionId}, "Fetch.enable", {patterns: fetchPatterns});
        } catch (e) {
          // the frame went away
        }
      }
      return {};
    case "Target.getTargets":
      return {targetInfos: await Promise.all((await ownTabs()).map(info))};
    case "Target.getTargetInfo":
      return {targetInfo: await info(await chrome.tabs.get(await tabOf(params.targetId)))};
    case "Target.createTarget":
      return {targetId: await create(params.url || "about:blank")};
    case "Target.attachToTarget":
      return {sessionId: await attach(await tabOf(params.targetId))};
    case "Target.detachFromTarget": {
      const tabId = Number(String(params.sessionId || "").replace(/^tab:/, ""));
      if (sessions.has(tabId)) await release(tabId);
      return {};
    }
    case "Target.closeTarget":
      await chrome.tabs.remove(await tabOf(params.targetId));
      return {success: true};
    default:
      throw new Error(`${method} is not available to the assistant`);
  }
}

async function create(url) {
  let id = await group();
  let tab;
  if (id === NONE) {
    // The group starts in a window of its own, so the assistant's tabs never cover the owner's.
    const win = await chrome.windows.create({url, focused: false});
    tab = win.tabs[0];
    id = await chrome.tabs.group({tabIds: [tab.id], createProperties: {windowId: win.id}});
    await chrome.tabGroups.update(id, {title: GROUP_TITLE, color: "blue", collapsed: false});
    groupId = id;
  } else {
    const [first] = await chrome.tabs.query({groupId: id});
    const window = await chrome.tabs.query({windowId: first.windowId});
    // Only bring the new tab forward when the group has its window to itself.
    const alone = window.every((t) => t.groupId === id);
    tab = await chrome.tabs.create({url, windowId: first.windowId, active: alone});
    await chrome.tabs.group({groupId: id, tabIds: [tab.id]});
  }
  await announce(await chrome.tabs.get(tab.id));
  return targetOf(tab.id);
}

// ---- tab events --------------------------------------------------------------------

chrome.tabs.onCreated.addListener(async (tab) => {
  if (!daemon || tab.openerTabId === undefined) return;
  const opener = await ownTab(tab.openerTabId);
  if (!opener) return;
  openers.set(tab.id, await targetOf(opener.id));
  if (tab.groupId !== NONE && tab.groupId === await group()) {
    await announce(tab);
    return;
  }
  try {
    await chrome.tabs.group({groupId: await group(), tabIds: [tab.id]});   // announced by onUpdated
  } catch (e) {
    // A popup window cannot hold a tab group; it still belongs to the assistant's tab that opened it.
    adopted.add(tab.id);
    await announce(tab);
  }
});

chrome.tabs.onUpdated.addListener(async (tabId, change, tab) => {
  if (!daemon) return;
  const own = await mine(tab);
  if ("groupId" in change) {
    if (own) {
      await announce(tab);
    } else if (targets.has(tabId)) {
      await forget(tabId);   // dragged out of the group: no longer the assistant's
    }
  } else if (own && targets.has(tabId) && (change.url || change.title)) {
    await announce(tab, "Target.targetInfoChanged");
  }
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (targets.has(tabId) || sessions.has(tabId)) await forget(tabId);
});

chrome.tabs.onReplaced.addListener(async (added, removed) => {
  if (targets.has(removed)) await forget(removed);
  const tab = await ownTab(added);
  if (tab) await announce(tab);
});

chrome.tabGroups.onRemoved.addListener((removed) => {
  if (removed.id === groupId) groupId = NONE;
});

// ---- start ------------------------------------------------------------------------------

chrome.alarms.create("coderunner-connect", {periodInMinutes: 1});   // retry if the host is not installed yet
chrome.alarms.onAlarm.addListener(connect);
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
connect();
