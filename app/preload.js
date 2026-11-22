// What the page may ask of the desktop shell, and nothing else: a folder chosen by the owner in the system's own
// dialog (where to keep the models), and the assistant started again when it has stopped. The page is the
// daemon's; it has no other reach into the computer.
const {contextBridge, ipcRenderer} = require("electron");

contextBridge.exposeInMainWorld("coderunnerShell", {
  pickFolder: (start) => ipcRenderer.invoke("pick-folder", typeof start === "string" ? start : ""),
  startAssistant: () => ipcRenderer.invoke("start-assistant"),
});
