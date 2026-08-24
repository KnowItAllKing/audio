import { app, BrowserWindow, clipboard, dialog, ipcMain } from "electron";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  getAiProviderStatus,
  resolveAiExecutable,
  runAiCliTurn,
  validateAiRunRequest
} from "./AiCliRunner";
import {
  getArchiveCliStatus,
  resolveArchiveExecutable,
  runArchiveExtraction
} from "./ArchiveJotExtractor";
import {
  readSpeakerMemoryFile,
  SPEAKER_MEMORY_FILE_NAME,
  writeSpeakerMemoryFile
} from "./SpeakerMemoryFile";
import type { AiCliStatus, AiProvider, AiRunRequest, AiRunResult } from "../shared/AiTypes";
import type {
  ArchiveCliStatus,
  ArchiveExtractRequest,
  ArchiveExtractResult,
  ArchiveJotConfig
} from "../shared/ArchiveJot";

const MAX_ACTIVE_AI_REQUESTS = 3;
const activeAiRequests = new Map<
  string,
  { controller: AbortController; uiThreadId: string }
>();

const MAX_REMEMBERED_BATCHES = 8;
const archiveJottedByBatch = new Map<string, Set<string>>();
let activeArchiveExtraction: { requestId: string; controller: AbortController } | null = null;

async function aiWorkspacePath(): Promise<string> {
  const path = join(app.getPath("userData"), "ai-workspace");
  await mkdir(path, { recursive: true });
  return path;
}

function speakerMemoryPath(): string {
  return join(app.getPath("userData"), SPEAKER_MEMORY_FILE_NAME);
}

function aiFailure(requestId: string, error: string, cancelled = false): AiRunResult {
  return { ok: false, requestId, error, cancelled, durationMs: 0 };
}

function archiveFailure(requestId: string, error: string): ArchiveExtractResult {
  return { ok: false, requestId, jotted: [], skipped: 0, error, durationMs: 0 };
}

function clampInt(value: string | undefined, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function archiveJotConfigFromEnv(env: NodeJS.ProcessEnv = process.env): ArchiveJotConfig {
  const enabledByDefault = ["1", "true", "on", "yes"].includes(
    String(env.CADENCE_ARCHIVE_JOTS || "").toLowerCase()
  );
  const provider: AiProvider = env.CADENCE_ARCHIVE_EXTRACT_PROVIDER === "codex" ? "codex" : "claude";
  const model = env.CADENCE_ARCHIVE_EXTRACT_MODEL ?? (provider === "claude" ? "haiku" : "");
  return {
    enabledByDefault,
    provider,
    model,
    windowSec: clampInt(env.CADENCE_ARCHIVE_WINDOW_SEC, 150, 30, 900),
    tickSec: clampInt(env.CADENCE_ARCHIVE_TICK_SEC, 60, 10, 600)
  };
}

function jottedMemoryForBatch(batch: string): Set<string> {
  let memory = archiveJottedByBatch.get(batch);
  if (!memory) {
    memory = new Set();
    archiveJottedByBatch.set(batch, memory);
    if (archiveJottedByBatch.size > MAX_REMEMBERED_BATCHES) {
      const oldest = archiveJottedByBatch.keys().next();
      if (!oldest.done) archiveJottedByBatch.delete(oldest.value);
    }
  }
  return memory;
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
  activeArchiveExtraction?.controller.abort();
  activeArchiveExtraction = null;
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
  "speakerMemory:load",
  async (): Promise<{ loaded: boolean; found: boolean; profiles?: unknown; error?: string }> => {
    try {
      const result = await readSpeakerMemoryFile(speakerMemoryPath());
      return {
        loaded: true,
        found: result.found,
        profiles: result.value
      };
    } catch (error) {
      return { loaded: false, found: false, error: String(error) };
    }
  }
);

ipcMain.handle(
  "speakerMemory:save",
  async (
    _event,
    profiles: unknown
  ): Promise<{ saved: boolean; error?: string }> => {
    try {
      await writeSpeakerMemoryFile(speakerMemoryPath(), { version: 1, profiles });
      return { saved: true };
    } catch (error) {
      return { saved: false, error: String(error) };
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

ipcMain.handle("archive:getConfig", (): ArchiveJotConfig => archiveJotConfigFromEnv());

ipcMain.handle("archive:getStatus", async (): Promise<ArchiveCliStatus> => getArchiveCliStatus());

ipcMain.handle(
  "archive:extract",
  async (_event, request: ArchiveExtractRequest): Promise<ArchiveExtractResult> => {
    const requestId = typeof request?.requestId === "string" ? request.requestId : "invalid";
    if (activeArchiveExtraction) {
      return archiveFailure(requestId, "An archive extraction is already running.");
    }
    const provider: AiProvider = request?.provider === "codex" ? "codex" : "claude";
    const [aiExecutable, archiveExecutable] = await Promise.all([
      resolveAiExecutable(provider),
      resolveArchiveExecutable()
    ]);
    if (!archiveExecutable) {
      return archiveFailure(requestId, "The archive CLI was not found. Install it or set CADENCE_ARCHIVE_PATH.");
    }
    if (!aiExecutable) {
      return archiveFailure(
        requestId,
        `${provider === "codex" ? "Codex" : "Claude"} CLI was not found. Install it or set the matching CADENCE_*_PATH environment variable.`
      );
    }

    const controller = new AbortController();
    activeArchiveExtraction = { requestId, controller };
    try {
      return await runArchiveExtraction(request, {
        aiExecutable,
        archiveExecutable,
        cwd: await aiWorkspacePath(),
        signal: controller.signal,
        alreadyJotted: jottedMemoryForBatch(typeof request?.batch === "string" ? request.batch : "")
      });
    } finally {
      activeArchiveExtraction = null;
    }
  }
);

ipcMain.handle("archive:cancel", (_event, requestId: unknown): boolean => {
  if (typeof requestId !== "string") return false;
  if (activeArchiveExtraction?.requestId !== requestId) return false;
  activeArchiveExtraction.controller.abort();
  return true;
});

ipcMain.handle(
  "autoSaveTranscript",
  async (
    _event,
    args: { fileName: string; jsonText: string }
  ): Promise<{ saved: boolean; path?: string; error?: string }> => {
    try {
      if (typeof args?.jsonText !== "string") return { saved: false, error: "jsonText is required" };
      const fileName =
        String(args.fileName || "").replace(/[^a-zA-Z0-9._-]+/g, "-").slice(0, 120) || "transcript.json";
      const dir = join(app.getPath("userData"), "transcripts");
      await mkdir(dir, { recursive: true });
      const path = join(dir, fileName);
      await writeFile(path, args.jsonText, "utf8");
      return { saved: true, path };
    } catch (e) {
      return { saved: false, error: String(e) };
    }
  }
);
