import type { AiProvider } from "../shared/AiTypes";

export const AI_THREAD_STORAGE_KEY = "cadence.aiThreads.v1";
const MAX_THREADS = 20;
const MAX_MESSAGES = 200;
const MAX_MESSAGE_CHARS = 100_000;

export type AiMessageRole = "user" | "assistant" | "status" | "error";

export type AiMessage = {
  id: string;
  role: AiMessageRole;
  text: string;
  createdAt: number;
};

export type AiThread = {
  id: string;
  provider: AiProvider;
  model: string;
  remoteThreadId?: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  lastTranscriptFingerprint?: string;
  lastTranscriptChars?: number;
  messages: AiMessage[];
};

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function cleanString(value: unknown, max: number): string {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function cleanTime(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function normalizeMessage(value: unknown, fallbackTime: number): AiMessage | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const role = source.role;
  if (role !== "user" && role !== "assistant" && role !== "status" && role !== "error") return null;
  const id = cleanString(source.id, 160);
  const text = cleanString(source.text, MAX_MESSAGE_CHARS).trim();
  if (!id || !text) return null;
  return { id, role, text, createdAt: cleanTime(source.createdAt, fallbackTime) };
}

export function normalizeAiThreads(value: unknown): AiThread[] {
  if (!Array.isArray(value)) return [];
  const now = Date.now();
  const ids = new Set<string>();
  const threads: AiThread[] = [];

  for (const item of value.slice(-MAX_THREADS)) {
    if (!item || typeof item !== "object") continue;
    const source = item as Record<string, unknown>;
    const id = cleanString(source.id, 160);
    const provider = source.provider;
    if (!id || ids.has(id) || (provider !== "codex" && provider !== "claude")) continue;
    ids.add(id);
    const createdAt = cleanTime(source.createdAt, now);
    const messages = Array.isArray(source.messages)
      ? source.messages
          .slice(-MAX_MESSAGES)
          .map((message) => normalizeMessage(message, createdAt))
          .filter((message): message is AiMessage => message !== null)
      : [];
    const remoteThreadId = cleanString(source.remoteThreadId, 160);
    const fingerprint = cleanString(source.lastTranscriptFingerprint, 100);
    const lastTranscriptChars =
      typeof source.lastTranscriptChars === "number" && Number.isFinite(source.lastTranscriptChars)
        ? Math.max(0, Math.floor(source.lastTranscriptChars))
        : undefined;
    threads.push({
      id,
      provider,
      model: cleanString(source.model, 100),
      remoteThreadId: remoteThreadId || undefined,
      title: cleanString(source.title, 100).trim() || `${provider === "codex" ? "Codex" : "Claude"} chat`,
      createdAt,
      updatedAt: cleanTime(source.updatedAt, createdAt),
      lastTranscriptFingerprint: fingerprint || undefined,
      lastTranscriptChars,
      messages
    });
  }
  return threads.sort((a, b) => a.createdAt - b.createdAt);
}

export function loadAiThreads(storage: StorageLike = localStorage): AiThread[] {
  try {
    const raw = storage.getItem(AI_THREAD_STORAGE_KEY);
    return raw ? normalizeAiThreads(JSON.parse(raw)) : [];
  } catch {
    return [];
  }
}

export function saveAiThreads(threads: AiThread[], storage: StorageLike = localStorage): AiThread[] {
  const normalized = normalizeAiThreads(threads);
  try {
    storage.setItem(AI_THREAD_STORAGE_KEY, JSON.stringify(normalized));
  } catch {
    // Q&A still works for this run when local storage is unavailable or full.
  }
  return normalized;
}

export function transcriptFingerprint(text: string): string {
  let hashA = 0x811c9dc5;
  let hashB = 0x9e3779b9;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    hashA = Math.imul(hashA ^ code, 0x01000193);
    hashB = Math.imul(hashB ^ (code + index), 0x85ebca6b);
  }
  return `${text.length.toString(36)}-${(hashA >>> 0).toString(36)}-${(hashB >>> 0).toString(36)}`;
}

export function titleFromQuestion(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length > 46 ? `${clean.slice(0, 45).trimEnd()}…` : clean || "New chat";
}
