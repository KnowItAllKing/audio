import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1100,
    height: 800,
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  // In dev, electron-vite provides a dev server URL in process.env.VITE_DEV_SERVER_URL
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    win.loadURL(devUrl);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    // Production: load built renderer html
    win.loadFile(join(__dirname, "../renderer/index.html"));
  }

  return win;
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle(
  "saveTranscript",
  async (
    _event,
    args: { suggestedName: string; jsonText: string }
  ): Promise<{ saved: boolean; path?: string; error?: string }> => {
    try {
      const win = BrowserWindow.getFocusedWindow();
      if (!win) return { saved: false, error: "no focused window" };

      const result = await dialog.showSaveDialog(win, {
        title: "Save transcript JSON",
        defaultPath: args.suggestedName,
        filters: [{ name: "JSON", extensions: ["json"] }]
      });

      if (result.canceled || !result.filePath) return { saved: false };
      await writeFile(result.filePath, args.jsonText, "utf8");
      return { saved: true, path: result.filePath };
    } catch (e) {
      return { saved: false, error: String(e) };
    }
  }
);

