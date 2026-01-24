import { contextBridge, ipcRenderer } from "electron";

type SaveTranscriptArgs = {
  suggestedName: string;
  jsonText: string;
};

type SaveTranscriptResult = { saved: boolean; path?: string; error?: string };

contextBridge.exposeInMainWorld("audioClient", {
  saveTranscript: (args: SaveTranscriptArgs): Promise<SaveTranscriptResult> =>
    ipcRenderer.invoke("saveTranscript", args)
});

declare global {
  interface Window {
    audioClient: {
      saveTranscript: (args: SaveTranscriptArgs) => Promise<SaveTranscriptResult>;
    };
  }
}

