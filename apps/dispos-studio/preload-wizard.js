// Preload script for the first-run setup wizard window.
//
// NOTE: the corresponding `ipcMain.on('wizard:start-setup', ...)`,
// `ipcMain.on('wizard:retry-step', ...)`, `ipcMain.on('wizard:launch-main-app', ...)`,
// and the `wizard:progress` / `wizard:complete` senders still need to be added
// to electron-main.js separately — this file only sets up the renderer-side
// bridge and assumes those handlers will exist.
//
// The main window currently runs with `contextIsolation: false`, so this repo
// doesn't use contextBridge anywhere — matching that pattern here and just
// attaching directly to `window` instead of introducing contextBridge.

const { ipcRenderer } = require('electron');

window.disposWizard = {
  // Hardcoded size estimates shown in Step 1 — not worth an IPC round trip.
  getComponentSizes() {
    return {
      image: { label: 'Image generation', bytes: 5 * 1024 * 1024 * 1024 },
      video: { label: 'Video generation', bytes: 6 * 1024 * 1024 * 1024 },
      threed: { label: '3D generation', bytes: 6 * 1024 * 1024 * 1024 },
      voice: { label: 'Voice / TTS', bytes: 3 * 1024 * 1024 * 1024 },
      chat: { label: 'Chat', bytes: 200 * 1024 * 1024 },
    };
  },

  startSetup(selectedComponents) {
    ipcRenderer.send('wizard:start-setup', selectedComponents);
  },

  retryStep(stepId) {
    ipcRenderer.send('wizard:retry-step', stepId);
  },

  onProgress(callback) {
    ipcRenderer.on('wizard:progress', (event, data) => callback(data));
  },

  onComplete(callback) {
    ipcRenderer.on('wizard:complete', (event, data) => callback(data));
  },

  launchMainApp() {
    ipcRenderer.send('wizard:launch-main-app');
  },
};
