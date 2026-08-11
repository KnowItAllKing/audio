import { app, BrowserWindow, clipboard, dialog, ipcMain } from "electron";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  getAiProviderStatus,
  resolveAiExecutable,
  runAiCliTurn,
  validateAiRunRequest
} from "./AiCliRunner";
import type { AiCliStatus, AiRunRequest, AiRunResult } from "../shared/AiTypes";

const MAX_ACTIVE_AI_REQUESTS = 3;
const activeAiRequests = new Map<
  string,
  { controller: AbortController; uiThreadId: string }
>();

async function aiWorkspacePath(): Promise<string> {
  const path = join(app.getPath("userData"), "ai-workspace");
  await mkdir(path, { recursive: true });
  return path;
}

function aiFailure(requestId: string, error: string, cancelled = false): AiRunResult {
  return { ok: false, requestId, error, cancelled, durationMs: 0 };
}

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    title: "Cadence",
    width: 1100,
    height: 800,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#f4f1ec",
    webPreferences: {
      // Build preload as CommonJS for best compatibility.
      preload: join(__dirname, "../preload/index.cjs"),
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

ipcMain.on("preload:log", (_event, msg: unknown) => {
  // Helpful for debugging when the preload fails to run/crashes.
  console.log("[preload]", msg);
});

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  for (const request of activeAiRequests.values()) request.controller.abort();
  activeAiRequests.clear();
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

ipcMain.handle(
  "copyText",
  (_event, args: { text: string }): { copied: boolean; error?: string } => {
    try {
      if (typeof args?.text !== "string") return { copied: false, error: "text is required" };
      clipboard.writeText(args.text);
      return { copied: true };
    } catch (e) {
      return { copied: false, error: String(e) };
    }
  }
);

ipcMain.handle("ai:getStatus", async (): Promise<AiCliStatus> => {
  const [codex, claude] = await Promise.all([
    getAiProviderStatus("codex"),
    getAiProviderStatus("claude")
  ]);
  return { codex, claude };
});

ipcMain.handle("ai:run", async (_event, request: AiRunRequest): Promise<AiRunResult> => {
  const requestId = typeof request?.requestId === "string" ? request.requestId : "invalid";
  try {
    validateAiRunRequest(request);
  } catch (error) {
    return aiFailure(requestId, error instanceof Error ? error.message : String(error));
  }

  if (activeAiRequests.has(request.requestId)) {
    return aiFailure(request.requestId, "This AI request is already running.");
  }
  if ([...activeAiRequests.values()].some((active) => active.uiThreadId === request.uiThreadId)) {
    return aiFailure(request.requestId, "This AI thread already has a request running.");
  }
  if (activeAiRequests.size >= MAX_ACTIVE_AI_REQUESTS) {
    return aiFailure(request.requestId, "Cadence can run at most three AI requests at once.");
  }

  const executable = await resolveAiExecutable(request.provider);
  if (!executable) {
    return aiFailure(
      request.requestId,
      `${request.provider === "codex" ? "Codex" : "Claude"} CLI was not found. Install it or set the matching CADENCE_*_PATH environment variable.`
    );
  }

  const controller = new AbortController();
  activeAiRequests.set(request.requestId, { controller, uiThreadId: request.uiThreadId });
  try {
    return await runAiCliTurn(request, {
      executable,
      cwd: await aiWorkspacePath(),
      signal: controller.signal
    });
  } finally {
    activeAiRequests.delete(request.requestId);
  }
});

ipcMain.handle("ai:cancel", (_event, requestId: unknown): boolean => {
  if (typeof requestId !== "string") return false;
  const active = activeAiRequests.get(requestId);
  if (!active) return false;
  active.controller.abort();
  return true;
});
