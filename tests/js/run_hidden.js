// Runs one of the hidden-window checks (tests/js/*_check.js) so that nothing it does can reach the screen when it
// fails: Electron shows an "Uncaught Exception" dialog for any error in its main script, even one that only hides
// windows, and a script with a syntax error never gets as far as hiding anything. Here the handlers come first and
// the check is loaded inside them, so every error goes to stderr and the process ends instead.
// Usage: Electron tests/js/run_hidden.js tests/js/<name>_check.js [the check's own arguments...]
"use strict";
const path = require("path");
const {dialog} = require("electron");

const quit = (what, error) => {
  process.stderr.write(`${what}: ${error?.stack || error}\n`);
  process.exit(1);
};
dialog.showErrorBox = (title, content) => process.stderr.write(`error box suppressed: ${title}\n${content}\n`);
process.on("uncaughtException", (error) => quit("uncaught", error));
process.on("unhandledRejection", (error) => quit("unhandled rejection", error));
process.env.CODERUNNER_APP_HIDDEN = "1";      // set before any check loads app/main.js

const check = process.argv.slice(2).find((arg) => arg.endsWith("_check.js"));
if (!check) quit("usage", "Electron tests/js/run_hidden.js tests/js/<name>_check.js [args...]");
try {
  require(path.resolve(check));
} catch (error) {
  quit("could not load the check", error);
}
