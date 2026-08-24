import { contextBridge, ipcRenderer } from "electron";
import type { AiCliStatus, AiRunRequest, AiRunResult } from "../shared/AiTypes";
import type {
  ArchiveCliStatus,
  ArchiveExtractRequest,
  ArchiveExtractResult,
  ArchiveJotConfig
} from "../shared/ArchiveJot";

type SaveTranscriptArgs = {
  suggestedName: string;
  jsonText: string;
};

type AutoSaveTranscriptArgs = {
  fileName: string;
  jsonText: string;
};

type SaveTranscriptResult = { saved: boolean; path?: string; error?: string };
type CopyTextResult = { copied: boolean; error?: string };
type LoadSpeakerMemoryResult = {
  loaded: boolean;
  found: boolean;
  profiles?: unknown;
  error?: string;
};
type SaveSpeakerMemoryResult = { saved: boolean; error?: string };

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
    loadSpeakerMemoryProfiles: (): Promise<LoadSpeakerMemoryResult> =>
      ipcRenderer.invoke("speakerMemory:load"),
    saveSpeakerMemoryProfiles: (profiles: unknown): Promise<SaveSpeakerMemoryResult> =>
      ipcRenderer.invoke("speakerMemory:save", profiles),
    getAiCliStatus: (): Promise<AiCliStatus> => ipcRenderer.invoke("ai:getStatus"),
    runAiTurn: (request: AiRunRequest): Promise<AiRunResult> => ipcRenderer.invoke("ai:run", request),
    cancelAiTurn: (requestId: string): Promise<boolean> => ipcRenderer.invoke("ai:cancel", requestId),
    getArchiveJotConfig: (): Promise<ArchiveJotConfig> => ipcRenderer.invoke("archive:getConfig"),
    getArchiveCliStatus: (): Promise<ArchiveCliStatus> => ipcRenderer.invoke("archive:getStatus"),
    runArchiveExtraction: (request: ArchiveExtractRequest): Promise<ArchiveExtractResult> =>
      ipcRenderer.invoke("archive:extract", request),
    cancelArchiveExtraction: (requestId: string): Promise<boolean> =>
      ipcRenderer.invoke("archive:cancel", requestId),
    autoSaveTranscript: (args: AutoSaveTranscriptArgs): Promise<SaveTranscriptResult> =>
      ipcRenderer.invoke("autoSaveTranscript", args)
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
      loadSpeakerMemoryProfiles: () => Promise<LoadSpeakerMemoryResult>;
      saveSpeakerMemoryProfiles: (profiles: unknown) => Promise<SaveSpeakerMemoryResult>;
      getAiCliStatus: () => Promise<AiCliStatus>;
      runAiTurn: (request: AiRunRequest) => Promise<AiRunResult>;
      cancelAiTurn: (requestId: string) => Promise<boolean>;
      getArchiveJotConfig: () => Promise<ArchiveJotConfig>;
      getArchiveCliStatus: () => Promise<ArchiveCliStatus>;
      runArchiveExtraction: (request: ArchiveExtractRequest) => Promise<ArchiveExtractResult>;
      cancelArchiveExtraction: (requestId: string) => Promise<boolean>;
      autoSaveTranscript: (args: AutoSaveTranscriptArgs) => Promise<SaveTranscriptResult>;
    };
  }
}
