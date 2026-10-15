// Runs the extension's background script against a fake chrome API and checks how attach() handles a tab that
// already has a debugger: its own attachment left from a restarted worker is taken over, someone else's is not.
// Usage: node attach_check.js BACKGROUND_JS
const fs = require("fs");
const vm = require("vm");

const listener = () => ({addListener() {}});
function fakeChrome(attachedBy) {
  // attachedBy: tabId -> "us" | "devtools"; the debugger refuses a second attach to a tab.
  return {
    action: {setBadgeText() {}, setTitle() {}},
    alarms: {create() {}, onAlarm: listener()},
    runtime: {connectNative: () => ({onMessage: listener(), onDisconnect: listener(), postMessage() {}}),
              getManifest: () => ({version: "0"}), onInstalled: listener(), onStartup: listener()},
    tabs: {onCreated: listener(), onRemoved: listener(), onReplaced: listener(), onUpdated: listener()},
    tabGroups: {onRemoved: listener(), TAB_GROUP_ID_NONE: -1},
    debugger: {
      onEvent: listener(), onDetach: listener(),
      async attach({tabId}) {
        if (attachedBy[tabId]) throw new Error(`Another debugger is already attached to the tab with id: ${tabId}.`);
        attachedBy[tabId] = "us";
      },
      async detach({tabId}) {
        if (attachedBy[tabId] !== "us") throw new Error(`Debugger is not attached to the tab with id: ${tabId}.`);
        delete attachedBy[tabId];
      },
      async sendCommand() { return {}; },
      async getTargets() { return []; },
    },
  };
}

async function main() {
  const source = fs.readFileSync(process.argv[2], "utf8");
  const results = [];
  for (const [owner, expect] of [["us", "attached"], ["devtools", "refused"]]) {
    const attachedBy = {7: owner};
    const context = vm.createContext({chrome: fakeChrome(attachedBy), console, setInterval() {}, setTimeout,
                                      Date, Map, Set, Error, Promise, JSON, String, Number});
    vm.runInContext(source, context);
    let outcome;
    try {
      outcome = (await context.attach(7)) === "tab:7" && attachedBy[7] === "us" ? "attached" : "wrong session";
    } catch (e) {
      outcome = /Another debugger/.test(e.message) && attachedBy[7] === owner ? "refused" : `error: ${e.message}`;
    }
    results.push(`${owner}: ${outcome}`);
    if (outcome !== expect) {
      console.error(`tab attached by ${owner}: expected ${expect}, got ${outcome}`);
      process.exit(1);
    }
  }
  console.log(results.join("; "));
}
main();
