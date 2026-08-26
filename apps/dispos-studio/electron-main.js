const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const path = require('path');

Menu.setApplicationMenu(null);

// Without an explicit AppUserModelID, Windows groups/caches the taskbar icon
// under the generic electron.exe identity, so icon-file changes can appear
// to have no effect between dev runs. Giving this app its own ID avoids that.
if (process.platform === 'win32') {
  app.setAppUserModelId('com.disposai.studio');
}

process.on('uncaughtException', (err) => {
  console.error('[Main Exception]', err);
});

app.on('before-quit', () => console.log('[Electron Lifecycle] before-quit'));
app.on('will-quit', () => console.log('[Electron Lifecycle] will-quit'));
app.on('quit', () => console.log('[Electron Lifecycle] quit'));
app.on('child-process-gone', (e, details) => console.log('[Electron Lifecycle] child-process-gone:', details));
app.on('render-process-gone', (e, webContents, details) => console.log('[Electron Lifecycle] render-process-gone:', details));

const distPath = process.env.DISPOS_DIST_PATH || path.join(__dirname, 'dist', 'index.html');

let win = null;

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

ipcMain.handle('select-file', async (event, { defaultPath, filters, properties } = {}) => {
  const result = await dialog.showOpenDialog({ defaultPath, filters, properties: properties || ['openFile'] });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  console.log('[Electron] window-all-closed event fired');
  if (process.platform !== 'darwin') app.quit();
});
