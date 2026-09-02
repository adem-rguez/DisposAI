const { app, BrowserWindow, dialog, ipcMain, Menu, session } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');

Menu.setApplicationMenu(null);

// Without this, Electron's default app name (package.json's "name":
// "dispos-studio") puts userData — including the renderer's localStorage,
// where chat history and the model catalog live — at
// %APPDATA%\dispos-studio, a folder the installer/uninstaller don't know
// about and that silently survives reinstalls. Naming it to match the
// product keeps it under %APPDATA%\DisposAI, consistent with dataRoot.
app.setName('DisposAI');

// Without an explicit AppUserModelID, Windows groups/caches the taskbar icon
// under the generic electron.exe identity, so icon-file changes can appear
// to have no effect between dev runs. Giving this app its own ID avoids that.
if (process.platform === 'win32') {
  app.setAppUserModelId('com.disposai.studio');
}

process.on('uncaughtException', (err) => {
  console.error('[Main Exception]', err);
});

// Chromium batches localStorage writes to disk; app.quit() below doesn't wait
// for that flush. Without this, a quit shortly after the renderer writes to
// localStorage (e.g. saving model-card settings right after a "Load
// configured model" click) can lose that write, making recently-set fields
// look "cleared" on next launch even though older/stable ones persisted fine.
app.on('before-quit', () => {
  console.log('[Electron Lifecycle] before-quit');
  try { session.defaultSession.flushStorageData(); } catch {}
  shutdownDaemon();
});
app.on('will-quit', () => console.log('[Electron Lifecycle] will-quit'));
app.on('quit', () => console.log('[Electron Lifecycle] quit'));
app.on('child-process-gone', (e, details) => console.log('[Electron Lifecycle] child-process-gone:', details));
app.on('render-process-gone', (e, webContents, details) => console.log('[Electron Lifecycle] render-process-gone:', details));

const distPath = process.env.DISPOS_DIST_PATH || path.join(__dirname, 'dist', 'index.html');

let win = null;
let daemon = null;
let wizardWin = null;

// --- Setup wizard: download-binaries.mjs bridge + component/step orchestration ---

let downloadBinariesModule = null;
async function getDownloadBinaries() {
  if (!downloadBinariesModule) downloadBinariesModule = await import('./scripts/download-binaries.mjs');
  return downloadBinariesModule;
}

// Set by the 'wizard:start-setup' handler so runSetupStep('binaries', ...) knows
// whether to also fetch sd.exe.
let lastRequestedComponents = [];

function resolveScriptsDir() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'scripts')
    : path.join(__dirname, '..', '..', 'scripts');
}

function runPythonSetupScript(scriptName, stepId, sendProgress) {
  return new Promise((resolve) => {
    const scriptPath = path.join(resolveScriptsDir(), scriptName);
    const env = {
      ...process.env,
      DISPOS_VENV_ROOT: runtimeDir,
      PATH: runtimeBinDir + path.delimiter + process.env.PATH,
    };

    let child;
    try {
      child = spawn('python', [scriptPath], { env, windowsHide: true });
    } catch (err) {
      sendProgress({ phase: stepId, status: 'error', message: err.message });
      resolve(false);
      return;
    }
    sendProgress({ phase: stepId, status: 'running', message: 'Starting setup...' });

    let buffer = '';
    let sawTerminal = false;
    let failed = false;

    const handleData = (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('PROGRESS: ')) continue;
        try {
          const parsed = JSON.parse(trimmed.slice('PROGRESS: '.length));
          if (parsed.status === 'done' || parsed.status === 'error') sawTerminal = true;
          if (parsed.status === 'error') failed = true;
          sendProgress({
            phase: stepId,
            status: parsed.status,
            message: `${parsed.phase}: ${parsed.message || parsed.status}`,
          });
        } catch {}
      }
    };

    child.stdout.on('data', handleData);
    child.stderr.on('data', () => {});

    child.on('error', (err) => {
      sendProgress({ phase: stepId, status: 'error', message: err.message });
      resolve(false);
    });

    child.on('exit', (code) => {
      if (code !== 0 && !sawTerminal) {
        sendProgress({ phase: stepId, status: 'error', message: 'Setup script exited with code ' + code });
        failed = true;
      }
      resolve(!failed);
    });
  });
}

async function runSetupStep(stepId, sendProgress) {
  try {
    if (stepId === 'binaries') {
      const { downloadLlamaServer, downloadSdBinary } = await getDownloadBinaries();
      await downloadLlamaServer(runtimeBinDir, sendProgress);
      if (lastRequestedComponents.includes('image')) {
        await downloadSdBinary(runtimeBinDir, sendProgress);
      } else {
        sendProgress({ phase: 'binaries', status: 'done' });
      }
    } else if (stepId === 'uv') {
      const { ensureUv } = await getDownloadBinaries();
      await ensureUv(runtimeBinDir, sendProgress);
    } else if (stepId === 'image') {
      return await runPythonSetupScript('setup_sd_env.py', 'image', sendProgress);
    } else if (stepId === 'video') {
      return await runPythonSetupScript('setup_video_env.py', 'video', sendProgress);
    } else if (stepId === 'voice') {
      return await runPythonSetupScript('setup_tts_env.py', 'voice', sendProgress);
    } else if (stepId === 'threed') {
      return await runPythonSetupScript('setup_3d_env.py', 'threed', sendProgress);
    }
    return true;
  } catch (err) {
    sendProgress({ phase: stepId, status: 'error', message: err.message });
    return false;
  }
}

// --- Daemon binary + runtime data layout ---

function resolveDaemonPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'daemon-core.exe');
  }
  return process.env.DISPOS_DAEMON_PATH || path.join(__dirname, '..', '..', 'target', 'release', 'daemon-core.exe');
}

// In production, all downloadable runtime bits (daemon-managed binaries,
// python venvs, model weights, install state) live under %LOCALAPPDATA%\DisposAI
// so they survive app updates/reinstalls and don't require admin rights to write.
// In dev we keep using the existing repo-relative "models" dir so local workflows
// (start.mjs, cargo, etc.) don't change.
// app.getPath('localAppData') is only safe to call once Electron has finished
// its own startup (calling it earlier, e.g. at module load time, throws
// "Failed to get 'localAppData' path" in packaged builds), so these are
// populated by initRuntimePaths() from within app.whenReady() instead of here.
let dataRoot;
let runtimeDir;
let runtimeBinDir;
let runtimeVenvsDir;
let modelsDir;
let installStatePath;

function initRuntimePaths() {
  // app.getPath('localAppData') throws "Failed to get 'localAppData' path" on
  // some Electron builds even after whenReady(), while other paths (appData,
  // temp, home) resolve fine. Read the env var directly instead.
  const localAppData = process.env.LOCALAPPDATA || path.join(app.getPath('home'), 'AppData', 'Local');
  dataRoot = app.isPackaged
    ? path.join(localAppData, 'DisposAI')
    : path.join(__dirname, '..', '..');

  runtimeDir = path.join(dataRoot, 'runtime');
  runtimeBinDir = path.join(runtimeDir, 'bin');
  runtimeVenvsDir = path.join(runtimeDir, 'venvs');
  modelsDir = app.isPackaged
    ? path.join(dataRoot, 'models')
    : path.join(__dirname, '..', '..', 'models');
  installStatePath = path.join(dataRoot, 'install-state.json');
}

function ensureRuntimeDirs() {
  for (const dir of [dataRoot, runtimeDir, runtimeBinDir, runtimeVenvsDir, modelsDir]) {
    try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  }
}

function needsFirstRunSetup() {
  try {
    const raw = fs.readFileSync(installStatePath, 'utf-8');
    const state = JSON.parse(raw);
    return state.setupComplete !== true;
  } catch {
    return true;
  }
}

// --- Daemon lifecycle (ported from scripts/start.mjs) ---

function daemonIsReady() {
  return new Promise(resolve => {
    const request = http.get('http://127.0.0.1:8080/health', response => {
      response.resume();
      resolve(true);
    });
    request.on('error', () => resolve(false));
    request.setTimeout(750, () => { request.destroy(); resolve(false); });
  });
}

async function ensureDaemon() {
  if (await daemonIsReady()) {
    console.log('[Daemon] Reusing local daemon at http://127.0.0.1:8080.');
    return;
  }
  const daemonBin = resolveDaemonPath();
  if (!fs.existsSync(daemonBin)) {
    console.error(`[Daemon] daemon-core not found at ${daemonBin}.`);
    return;
  }
  const env = { ...process.env, DISPOS_MODELS_DIR: modelsDir };
  if (app.isPackaged) {
    env.DISPOS_LLAMA_SERVER_BINARY = path.join(runtimeBinDir, 'llama-server.exe');
    env.DISPOS_SD_BINARY = path.join(runtimeBinDir, 'sd.exe');
    env.DISPOS_SD_PYTHON = path.join(runtimeVenvsDir, '.venv-sd', 'Scripts', 'python.exe');
    env.DISPOS_VIDEO_PYTHON = path.join(runtimeVenvsDir, '.venv-video', 'Scripts', 'python.exe');
    env.DISPOS_TTS_PYTHON = path.join(runtimeVenvsDir, '.venv-tts', 'Scripts', 'python.exe');
    env.DISPOS_3D_PYTHON = path.join(runtimeVenvsDir, '.venv-3d', 'Scripts', 'python.exe');
  }
  console.log('[Daemon] Starting local daemon...');
  daemon = spawn(daemonBin, [], {
    cwd: path.join(__dirname, '..', '..'),
    stdio: 'inherit',
    env,
    windowsHide: true,
  });
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (await daemonIsReady()) {
      console.log('[Daemon] Local daemon is ready.');
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  console.error('[Daemon] Timed out waiting for daemon-core on http://127.0.0.1:8080.');
}

async function shutdownDaemon() {
  if (await daemonIsReady()) {
    console.log('[Daemon] Sending graceful shutdown signal to daemon');
    await new Promise(resolve => {
      http.get('http://127.0.0.1:8080/shutdown', () => {}).on('error', () => {});
      setTimeout(resolve, 750);
    });
  }
  try { daemon?.kill(); } catch {}
  daemon = null;
}

function createWindow() {
  console.log('[Electron] Creating BrowserWindow...');
  win = new BrowserWindow({
    width: 1300,
    height: 850,
    minWidth: 1000,
    minHeight: 650,
    title: 'Dispos Studio',
    icon: path.join(__dirname, 'src', 'assets', 'dispos_logo.ico'),
    backgroundColor: '#07090e',
    show: true,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#0f172a',
      symbolColor: '#f8fafc',
      height: 52,
    },
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
    }
  });

  win.loadFile(distPath);

  win.webContents.on('did-finish-load', () => {
    console.log('[Electron] Page finish load success');
  });

  win.webContents.on('did-fail-load', (event, code, desc) => {
    console.error(`[Electron Load Error] Code ${code}: ${desc}`);
  });

  win.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`[Renderer Console] ${message}`);
  });

  if (process.env.DISPOS_DEBUG) {
    win.webContents.openDevTools({ mode: 'detach' });
  }

  win.on('closed', () => {
    console.log('[Electron] Window closed event');
    win = null;
  });
}

function createWizardWindow() {
  console.log('[Electron] Creating setup wizard BrowserWindow...');
  wizardWin = new BrowserWindow({
    width: 640,
    height: 520,
    title: 'Dispos Studio Setup',
    icon: path.join(__dirname, 'src', 'assets', 'dispos_logo.ico'),
    backgroundColor: '#07090e',
    show: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
      preload: path.join(__dirname, 'preload-wizard.js'),
    }
  });

  wizardWin.loadFile(path.join(__dirname, 'dist', 'wizard.html'));

  wizardWin.on('closed', () => {
    console.log('[Electron] Wizard window closed event');
    wizardWin = null;
  });
}

ipcMain.handle('select-file', async (event, { defaultPath, filters, properties } = {}) => {
  const result = await dialog.showOpenDialog({ defaultPath, filters, properties: properties || ['openFile'] });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

ipcMain.on('wizard:start-setup', async (event, selectedComponents) => {
  lastRequestedComponents = selectedComponents || [];
  const sendProgress = (data) => event.sender.send('wizard:progress', data);

  const results = await Promise.allSettled([
    runSetupStep('binaries', sendProgress),
    runSetupStep('uv', sendProgress),
  ]);
  let allSucceeded = results.every((r) => r.status === 'fulfilled' && r.value !== false);

  for (const stepId of lastRequestedComponents) {
    const ok = await runSetupStep(stepId, sendProgress);
    if (ok === false) allSucceeded = false;
  }

  try {
    fs.writeFileSync(installStatePath, JSON.stringify({ setupComplete: allSucceeded, components: lastRequestedComponents }));
  } catch (err) {
    console.error('[Wizard] Failed to write install state:', err);
  }

  event.sender.send('wizard:complete');
});

ipcMain.on('wizard:retry-step', async (event, stepId) => {
  await runSetupStep(stepId, (data) => event.sender.send('wizard:progress', data));
  event.sender.send('wizard:complete');
});

ipcMain.on('wizard:launch-main-app', () => {
  // Only reached once the wizard UI shows no remaining step errors (including
  // after retries), so it's safe to mark setup complete here rather than only
  // at the end of the initial run, which a later successful retry wouldn't update.
  try {
    fs.writeFileSync(installStatePath, JSON.stringify({ setupComplete: true, components: lastRequestedComponents }));
  } catch (err) {
    console.error('[Wizard] Failed to write install state:', err);
  }
  if (wizardWin) wizardWin.close();
  createWindow();
});

app.whenReady().then(async () => {
  initRuntimePaths();
  ensureRuntimeDirs();
  await ensureDaemon();
  if (!(await daemonIsReady())) {
    dialog.showErrorBox(
      'Dispos Studio',
      'The local daemon failed to start. Some features may be unavailable until it is running.'
    );
  }
  if (app.isPackaged && needsFirstRunSetup()) {
    createWizardWindow();
  } else {
    createWindow();
  }
});

app.on('window-all-closed', () => {
  console.log('[Electron] window-all-closed event fired');
  try { session.defaultSession.flushStorageData(); } catch {}
  shutdownDaemon();
  if (process.platform !== 'darwin') app.quit();
});
