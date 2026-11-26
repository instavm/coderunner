// Electron main for tests/test_app_browser_e2e.py: the app's browser (app/browser.js) in a window that is never
// shown, connected to the test's bridge. Nothing appears on the screen, the Dock or the menu bar, and nothing takes
// the focus. Lines on stdin: "shot <path>" saves the active tab as a PNG (capturePage), "hide" / "show" say whether
// the Browser tab is shown, "state" prints the tabs, "act {json}" presses one of the owner's buttons or menu items,
// "outside" lists what would have gone to the default browser or the clipboard (stand-ins here: the owner's are never
// touched), "fresh-quit" starts the browser again as the app does with Keep cookies "Until quit"; each answers one
// JSON line on stdout.
"use strict";
const {app, BrowserWindow} = require("electron");
const fs = require("fs");
const path = require("path");

app.commandLine.appendSwitch("use-mock-keychain");         // a hidden Keychain prompt would hang every page
if (process.env.CODERUNNER_APP_DATA) app.setPath("userData", process.env.CODERUNNER_APP_DATA);
if (process.platform === "darwin") { app.setActivationPolicy("accessory"); app.dock?.hide(); }

const {AppBrowser} = require(path.join(process.env.CODERUNNER_APP_DIR, "browser.js"));
const SIZE = {x: 0, y: 0, width: 1280, height: 860};

app.whenReady().then(() => {
  const win = new BrowserWindow({show: false, width: SIZE.width, height: SIZE.height, paintWhenInitiallyHidden: true});
  let keep = "always";
  const outside = [];
  const browser = new AppBrowser({assistantDir: process.env.CODERUNNER_ASSISTANT, keep: () => keep, setKeep: (mode) => { keep = mode; },
                                  openExternal: (url) => { outside.push(["open", url]); }, writeClipboard: (text) => { outside.push(["copy", text]); }});
  browser.setHost(win);
  browser.place(SIZE, true);
  browser.onState = (state) => { browser.last = state; };
  browser.connect();
  const say = (message) => process.stdout.write(JSON.stringify(message) + "\n");
  let buffer = "";
  process.stdin.on("data", async (chunk) => {
    buffer += chunk;
    for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
      const [command, ...rest] = buffer.slice(0, end).trim().split(" ");
      buffer = buffer.slice(end + 1);
      try {
        if (command === "shot") {
          const tab = browser.tabs.get(browser.active);
          const image = await tab.view.webContents.capturePage();
          fs.writeFileSync(rest.join(" "), image.toPNG());
          say({ok: true, size: image.getSize()});
        } else if (command === "hide" || command === "show") {
          browser.place(SIZE, command === "show");
          say({ok: true});
        } else if (command === "act") {
          const result = await browser.act(JSON.parse(rest.join(" ")));
          say({ok: true, result: typeof result === "string" && result.length > 200 ? `${result.slice(0, 23)}…` : result ?? null});
        } else if (command === "outside") {
          say({ok: true, calls: outside});
        } else if (command === "fresh-quit") {
          new AppBrowser({assistantDir: "/nonexistent", keep: () => "quit"});
          await new Promise((resolve) => setTimeout(resolve, 500));
          say({ok: true});
        } else if (command === "state") {
          say({ok: true, state: browser.state()});
        } else if (command === "quit") {
          say({ok: true});
          setTimeout(() => process.exit(0), 100);
        }
      } catch (error) {
        say({ok: false, error: String(error?.message || error)});
      }
    }
  });
  say({ready: true});
});
app.on("window-all-closed", () => {});
