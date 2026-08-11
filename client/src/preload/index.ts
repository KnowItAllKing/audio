import { contextBridge, ipcRenderer } from "electron";
import type { AiCliStatus, AiRunRequest, AiRunResult } from "../shared/AiTypes";

type SaveTranscriptArgs = {
  suggestedName: string;
  jsonText: string;
};

type SaveTranscriptResult = { saved: boolean; path?: string; error?: string };
type CopyTextResult = { copied: boolean; error?: string };

function preloadLog(msg: unknown): void {
  try {
    ipcRenderer.send("preload:log", msg);
  } catch {
    // ignore
  }
}

preloadLog("starting");

try {
  contextBridge.exposeInMainWorld("audioClient", {
    saveTranscript: (args: SaveTranscriptArgs): Promise<SaveTranscriptResult> =>
      ipcRenderer.invoke("saveTranscript", args),
    copyText: (text: string): Promise<CopyTextResult> => ipcRenderer.invoke("copyText", { text }),
    getAiCliStatus: (): Promise<AiCliStatus> => ipcRenderer.invoke("ai:getStatus"),
    runAiTurn: (request: AiRunRequest): Promise<AiRunResult> => ipcRenderer.invoke("ai:run", request),
    cancelAiTurn: (requestId: string): Promise<boolean> => ipcRenderer.invoke("ai:cancel", requestId)
  });
  preloadLog("exposed window.audioClient");
} catch (e) {
  preloadLog({ error: "failed to expose window.audioClient", detail: String(e) });
}

declare global {
  interface Window {
    audioClient: {
      saveTranscript: (args: SaveTranscriptArgs) => Promise<SaveTranscriptResult>;
      copyText: (text: string) => Promise<CopyTextResult>;
      getAiCliStatus: () => Promise<AiCliStatus>;
      runAiTurn: (request: AiRunRequest) => Promise<AiRunResult>;
      cancelAiTurn: (requestId: string) => Promise<boolean>;
    };
  }
}
