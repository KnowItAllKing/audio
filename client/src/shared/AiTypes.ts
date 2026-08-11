export type AiProvider = "codex" | "claude";

export type AiRunRequest = {
  requestId: string;
  uiThreadId: string;
  provider: AiProvider;
  model: string;
  remoteThreadId?: string;
  transcript: string;
  question: string;
  syncOnly: boolean;
};

export type AiRunResult = {
  ok: boolean;
  requestId: string;
  remoteThreadId?: string;
  response?: string;
  error?: string;
  cancelled?: boolean;
  durationMs: number;
};

export type AiProviderStatus = {
  available: boolean;
  path?: string;
  version?: string;
  error?: string;
};

export type AiCliStatus = Record<AiProvider, AiProviderStatus>;
