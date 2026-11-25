// What the page may ask of the desktop shell, and nothing else: a folder chosen by the owner in the system's own
// dialog (where to keep the models), and the assistant started again when it has stopped. The page is the
// daemon's; it has no other reach into the computer.
const {contextBridge, ipcRenderer} = require("electron");

const text = (value) => (typeof value === "string" ? value : "");

contextBridge.exposeInMainWorld("coderunnerShell", {
  pickFolder: (start) => ipcRenderer.invoke("pick-folder", typeof start === "string" ? start : ""),
  startAssistant: () => ipcRenderer.invoke("start-assistant"),
  // The menu bar's popover and the quick ask (also the daemon's pages): a thread, the app or the computer's
  // screen opened, the panel put away, word that a task was started. The main page says which thread it shows
  // and is told what to open. The shell shows the notifications (notifies), so the page does not show them twice.
  notifies: true,
  openThread: (conversation) => ipcRenderer.send("open-thread", text(conversation)),
  openApp: () => ipcRenderer.send("open-app"),
  openComputer: () => ipcRenderer.send("open-computer"),
  hidePanel: () => ipcRenderer.send("hide-panel"),
  taskStarted: () => ipcRenderer.send("task-started"),
  threadOpen: (conversation) => ipcRenderer.send("thread-open", text(conversation)),
  onShowThread: (callback) => ipcRenderer.on("show-thread", (_, conversation) => callback(text(conversation))),
  onShowComputer: (callback) => ipcRenderer.on("show-computer", () => callback()),
  onShown: (callback) => ipcRenderer.on("panel-shown", () => callback()),
});
