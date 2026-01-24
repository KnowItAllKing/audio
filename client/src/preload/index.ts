import { contextBridge, ipcRenderer } from "electron";

type SaveTranscriptArgs = {
  suggestedName: string;
  jsonText: string;
};

type SaveTranscriptResult = { saved: boolean; path?: string; error?: string };

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
      ipcRenderer.invoke("saveTranscript", args)
  });
  preloadLog("exposed window.audioClient");
} catch (e) {
  preloadLog({ error: "failed to expose window.audioClient", detail: String(e) });
}

declare global {
  interface Window {
    audioClient: {
      saveTranscript: (args: SaveTranscriptArgs) => Promise<SaveTranscriptResult>;
    };
  }
}

