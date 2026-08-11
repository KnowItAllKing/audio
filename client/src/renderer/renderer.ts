import { AudioStreamSender, type StreamId } from "./audio/AudioStreamSender";
import {
  chooseLikelyLoopbackDevice,
  chooseLikelyMicDevice
} from "./audio/LoopbackDevice";
import { MicInputFilter, getAudioStats, type MicInputFilterDecision } from "./audio/MicInputFilter";
import {
  loadSpeakerMemoryProfiles,
  saveSpeakerMemoryProfiles,
  type SpeakerMemoryProfile
} from "./SpeakerMemoryStore";
import { reconcileCrossStreamDuplicates } from "./TranscriptReconciler";
import { formatCleanTranscript } from "./CleanTranscript";
import {
  loadAiThreads,
  saveAiThreads,
  titleFromQuestion,
  transcriptFingerprint,
  type AiMessage,
  type AiThread
} from "./AiThreadStore";
import type { AiCliStatus, AiProvider, AiRunRequest } from "../shared/AiTypes";

type TranscriptSegment = {
  id: string;
  start_sec: number;
  end_sec: number;
  text: string;
  stream_tags: StreamId[];
  speaker_id: string;
  speaker_label: string;
  speaker_source: "stream" | "zoom" | "diarization" | "memory" | "manual" | "unknown";
  speaker_confidence: number;
  is_final: boolean;
  full_context_available: boolean;
  final_reason?: string;
};

type TranscriptLayer = "raw" | "processed";

type TranscriptUpdateMessage = {
  type: "transcript_update";
  session_id: string;
  layer?: TranscriptLayer;
  segments: TranscriptSegment[];
};

type ErrorMessage = {
  type: "error";
  session_id: string;
  code: string;
  message: string;
};

type SpeakerMemoryReviewSample = {
  sample_id: string;
  speaker_id: string;
  speaker_label: string;
  stream_id: StreamId;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  audio_wav_base64: string;
  matched_profile_id?: string | null;
  matched_name?: string | null;
  match_confidence?: number | null;
};

type SpeakerMemoryReviewMessage = {
  type: "speaker_memory_review";
  session_id: string;
  samples: SpeakerMemoryReviewSample[];
};

type SpeakerMemoryProfileMessage = {
  type: "speaker_memory_profile";
  session_id: string;
  profile: SpeakerMemoryProfile;
  profiles: SpeakerMemoryProfile[];
};

type SpeakerMemoryProfilesMessage = {
  type: "speaker_memory_profiles";
  session_id: string;
  profiles: SpeakerMemoryProfile[];
};

type ControlMessage = {
  type: "control";
  session_id: string;
  command:
    | "stop"
    | "speaker_memory_review"
    | "speaker_memory_enroll"
    | "speaker_memory_profiles";
  payload?: Record<string, unknown>;
};

type ServerControlMessage = {
  type: "control";
  session_id: string;
  command: "pong" | "stopped";
  payload?: Record<string, unknown>;
};

// AudioChunkMessage type lives in AudioStreamSender module.

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const wsUrlInput = $<HTMLInputElement>("wsUrl");
const sessionIdInput = $<HTMLInputElement>("sessionId");
const micSelect = $<HTMLSelectElement>("micSelect");
const sysSelect = $<HTMLSelectElement>("sysSelect");
const micMuteBtn = $<HTMLButtonElement>("micMuteBtn");
const micFilterEnabledInput = $<HTMLInputElement>("micFilterEnabled");
const micGateThresholdInput = $<HTMLInputElement>("micGateThreshold");
const micGateStatusEl = $<HTMLSpanElement>("micGateStatus");
const micLevelStatusEl = $<HTMLSpanElement>("micLevelStatus");
const sysLevelStatusEl = $<HTMLSpanElement>("sysLevelStatus");
const micMeterFillEl = $<HTMLDivElement>("micMeterFill");
const sysMeterFillEl = $<HTMLDivElement>("sysMeterFill");
const refreshBtn = $<HTMLButtonElement>("refreshBtn");
const startBtn = $<HTMLButtonElement>("startBtn");
const stopBtn = $<HTMLButtonElement>("stopBtn");
const memoryReviewBtn = $<HTMLButtonElement>("memoryReviewBtn");
const memoryStatusEl = $<HTMLSpanElement>("memoryStatus");
const speakerReviewListEl = $<HTMLDivElement>("speakerReviewList");
const transcriptEl = $<HTMLDivElement>("transcript");
const speakerLegendEl = $<HTMLDivElement>("speakerLegend");
const connStatusEl = $<HTMLSpanElement>("connStatus");
const lastUpdateEl = $<HTMLSpanElement>("lastUpdate");
const sentStatsEl = $<HTMLSpanElement>("sentStats");
const recvStatsEl = $<HTMLSpanElement>("recvStats");
const viewBubblesBtn = $<HTMLButtonElement>("viewBubblesBtn");
const viewBlocksBtn = $<HTMLButtonElement>("viewBlocksBtn");
const viewScriptBtn = $<HTMLButtonElement>("viewScriptBtn");
const layerRawBtn = $<HTMLButtonElement>("layerRawBtn");
const layerProcessedBtn = $<HTMLButtonElement>("layerProcessedBtn");
const rawLayerCountEl = $<HTMLSpanElement>("rawLayerCount");
const processedLayerCountEl = $<HTMLSpanElement>("processedLayerCount");
const layerDescriptionEl = $<HTMLSpanElement>("layerDescription");
const jumpLatestBtn = $<HTMLButtonElement>("jumpLatestBtn");
const copyTranscriptBtn = $<HTMLButtonElement>("copyTranscriptBtn");
const askAiBtn = $<HTMLButtonElement>("askAiBtn");
const aiPanelEl = $<HTMLElement>("aiPanel");
const aiPanelCloseBtn = $<HTMLButtonElement>("aiPanelCloseBtn");
const aiProviderSelect = $<HTMLSelectElement>("aiProviderSelect");
const aiModelSelect = $<HTMLSelectElement>("aiModelSelect");
const aiNewThreadBtn = $<HTMLButtonElement>("aiNewThreadBtn");
const aiCliStatusEl = $<HTMLSpanElement>("aiCliStatus");
const aiRefreshStatusBtn = $<HTMLButtonElement>("aiRefreshStatusBtn");
const aiTabsEl = $<HTMLDivElement>("aiTabs");
const aiEmptyEl = $<HTMLDivElement>("aiEmpty");
const aiConversationEl = $<HTMLDivElement>("aiConversation");
const aiTranscriptStateEl = $<HTMLSpanElement>("aiTranscriptState");
const aiSyncBtn = $<HTMLButtonElement>("aiSyncBtn");
const aiMessagesEl = $<HTMLDivElement>("aiMessages");
const aiQuestionEl = $<HTMLTextAreaElement>("aiQuestion");
const aiCancelBtn = $<HTMLButtonElement>("aiCancelBtn");
const aiSendBtn = $<HTMLButtonElement>("aiSendBtn");

function uuidv4(): string {
  // Browser-safe UUID (Chromium supports crypto.randomUUID)
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi.randomUUID === "function") return cryptoApi.randomUUID();
  const b = new Uint8Array(16);
  cryptoApi.getRandomValues(b);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const hex = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(
    12,
    16
  )}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function setConnStatus(text: string, kind: "good" | "bad" | "neutral" = "neutral") {
  connStatusEl.textContent = text;
  connStatusEl.classList.remove("good", "bad");
  if (kind === "good") connStatusEl.classList.add("good");
  if (kind === "bad") connStatusEl.classList.add("bad");
}

function setLastUpdate(ts: number | null) {
  lastUpdateEl.textContent = ts ? new Date(ts).toLocaleTimeString() : "—";
}

function formatTime(epochSec: number): string {
  // Timestamps are Unix epoch seconds from the client's clock.
  const date = new Date(epochSec * 1000);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 2
  } as Intl.DateTimeFormatOptions);
}

function floatToPcmS16leMono(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = Math.max(-1, Math.min(1, input[i]));
    out[i] = x < 0 ? Math.round(x * 32768) : Math.round(x * 32767);
  }
  return out;
}

function micGateThreshold(): number {
  return Number.parseFloat(micGateThresholdInput.value) || 0.012;
}

function updateMicMuteUi(): void {
  micMuteBtn.textContent = micMuted ? "Unmute mic" : "Mute mic";
  micMuteBtn.classList.toggle("danger", micMuted);
  micMuteBtn.classList.toggle("primary", !micMuted);
  micMuteBtn.setAttribute("aria-pressed", String(micMuted));
}

function updateMicGateStatus(decision: MicInputFilterDecision | null = micLastDecision): void {
  const threshold = micGateThreshold();
  if (!decision) {
    micGateStatusEl.textContent = `idle gate=${threshold.toFixed(3)}`;
    micGateStatusEl.classList.remove("good", "bad");
    return;
  }
  micGateStatusEl.textContent = `${decision.reason} rms=${decision.rms.toFixed(3)} gate=${threshold.toFixed(3)}`;
  micGateStatusEl.classList.toggle("good", decision.shouldSend && decision.reason !== "muted");
  micGateStatusEl.classList.toggle("bad", decision.reason === "muted");
}

function updateLevelStatus(
  streamId: StreamId,
  stats: { rms: number; peak: number } | null
): void {
  const statusEl = streamId === "mic" ? micLevelStatusEl : sysLevelStatusEl;
  const fillEl = streamId === "mic" ? micMeterFillEl : sysMeterFillEl;
  if (!stats) {
    statusEl.textContent = "idle";
    statusEl.classList.remove("good", "bad");
    fillEl.style.width = "0%";
    return;
  }

  const percent = Math.min(100, Math.round(stats.peak * 160));
  fillEl.style.width = `${percent}%`;

  if (stats.peak >= 0.98) {
    statusEl.textContent = `clip peak=${stats.peak.toFixed(2)}`;
    statusEl.classList.remove("good");
    statusEl.classList.add("bad");
    return;
  }

  statusEl.textContent = `rms=${stats.rms.toFixed(3)} peak=${stats.peak.toFixed(2)}`;
  statusEl.classList.toggle("good", stats.rms >= 0.006);
  statusEl.classList.remove("bad");
}

type CaptureHandle = {
  streamId: StreamId;
  mediaStream: MediaStream;
  audioContext: AudioContext;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  stop: () => Promise<void>;
};

let ws: WebSocket | null = null;
let startedAt: number | null = null;
let endedAt: number | null = null;
let resolveStopAck: ((received: boolean) => void) | null = null;

const rawSegmentsById = new Map<string, TranscriptSegment>();
const processedSegmentsById = new Map<string, TranscriptSegment>();
let recvCount = 0;
let transcriptViewMode: "bubbles" | "blocks" | "script" = "bubbles";
let transcriptLayer: TranscriptLayer = "raw";
let renderedTranscriptLayer: TranscriptLayer | null = null;

const AI_MODEL_OPTIONS: Record<AiProvider, Array<{ value: string; label: string }>> = {
  codex: [
    { value: "", label: "Automatic (CLI default)" },
    { value: "gpt-5.6-terra", label: "GPT-5.6 Terra" },
    { value: "gpt-5.6-sol", label: "GPT-5.6 Sol" }
  ],
  claude: [
    { value: "", label: "Automatic (CLI default)" },
    { value: "sonnet", label: "Sonnet" },
    { value: "opus", label: "Opus" },
    { value: "fable", label: "Fable" }
  ]
};

let aiThreads: AiThread[] = loadAiThreads();
let activeAiThreadId: string | null = aiThreads.at(-1)?.id ?? null;
let aiCliStatus: AiCliStatus | null = null;
const runningAiThreads = new Map<string, { requestId: string; cancelling: boolean }>();

type TranscriptScrollState = {
  scrollTop: number;
  followLatest: boolean;
};

const transcriptScrollState: Record<TranscriptLayer, TranscriptScrollState> = {
  raw: { scrollTop: 0, followLatest: true },
  processed: { scrollTop: 0, followLatest: true }
};

let micSender: AudioStreamSender | null = null;
let sysSender: AudioStreamSender | null = null;

let captures: CaptureHandle[] = [];
const captureSwitchSeq: Record<StreamId, number> = { mic: 0, system: 0 };
let micMuted = false;
let micLastDecision: MicInputFilterDecision | null = null;
const micInputFilter = new MicInputFilter();

function sendControl(command: ControlMessage["command"], payload?: Record<string, unknown>): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const msg: ControlMessage = {
    type: "control",
    session_id: sessionIdInput.value.trim(),
    command,
    payload
  };
  ws.send(JSON.stringify(msg));
}

function waitForStopAck(timeoutMs = 120_000): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (received: boolean) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      if (resolveStopAck === finish) resolveStopAck = null;
      resolve(received);
    };
    const timeout = window.setTimeout(() => finish(false), timeoutMs);
    resolveStopAck = finish;
  });
}

function updateSpeakerMemoryStatus(profiles: SpeakerMemoryProfile[]): void {
  if (profiles.length === 0) {
    memoryStatusEl.textContent = "no saved voices";
    memoryStatusEl.classList.remove("good", "bad");
    return;
  }
  memoryStatusEl.textContent = `${profiles.length} saved voice${profiles.length === 1 ? "" : "s"}`;
  memoryStatusEl.classList.add("good");
  memoryStatusEl.classList.remove("bad");
}

function syncSpeakerMemoryProfiles(): void {
  const profiles = loadSpeakerMemoryProfiles();
  updateSpeakerMemoryStatus(profiles);
  sendControl("speaker_memory_profiles", { profiles });
}

function storeSpeakerMemoryProfiles(profiles: SpeakerMemoryProfile[]): SpeakerMemoryProfile[] {
  const saved = saveSpeakerMemoryProfiles(profiles);
  updateSpeakerMemoryStatus(saved);
  return saved;
}

function updateSentStats(): void {
  sentStatsEl.textContent = `mic=${micSender?.getSentCount() ?? 0} sys=${sysSender?.getSentCount() ?? 0}`;
}

function isConnected(): boolean {
  return !!ws && ws.readyState === WebSocket.OPEN;
}

function selectedDeviceForStream(streamId: StreamId): string {
  return streamId === "mic" ? micSelect.value : sysSelect.value;
}

function setStreamSender(streamId: StreamId, sender: AudioStreamSender | null): void {
  if (streamId === "mic") {
    micSender = sender;
  } else {
    sysSender = sender;
  }
  updateSentStats();
}

function setStreamLevelStatus(
  streamId: StreamId,
  text: string,
  kind: "good" | "bad" | "neutral" = "neutral"
): void {
  const statusEl = streamId === "mic" ? micLevelStatusEl : sysLevelStatusEl;
  const fillEl = streamId === "mic" ? micMeterFillEl : sysMeterFillEl;
  statusEl.textContent = text;
  statusEl.classList.remove("good", "bad");
  if (kind === "good") statusEl.classList.add("good");
  if (kind === "bad") statusEl.classList.add("bad");
  fillEl.style.width = "0%";
}

async function stopCapturesForStream(streamId: StreamId): Promise<void> {
  const keep: CaptureHandle[] = [];
  const stopping: CaptureHandle[] = [];
  for (const capture of captures) {
    if (capture.streamId === streamId) {
      stopping.push(capture);
    } else {
      keep.push(capture);
    }
  }
  captures = keep;
  await Promise.all(stopping.map((capture) => capture.stop()));
  setStreamSender(streamId, null);
  setStreamLevelStatus(streamId, "idle");
  if (streamId === "mic") {
    micInputFilter.reset();
    micLastDecision = null;
    updateMicGateStatus(null);
  }
}

function applyMicMuteToTracks(): void {
  for (const capture of captures) {
    if (capture.streamId !== "mic") continue;
    for (const track of capture.mediaStream.getAudioTracks()) {
      track.enabled = !micMuted;
    }
  }
}

function pushMicInput(input: Float32Array): void {
  const decision = micInputFilter.decide(input, performance.now(), {
    muted: micMuted,
    enabled: micFilterEnabledInput.checked,
    thresholdRms: micGateThreshold()
  });
  micLastDecision = decision;
  updateMicGateStatus(decision);
  updateLevelStatus("mic", decision);

  if (!decision.shouldSend) return;

  const pcm = floatToPcmS16leMono(input);
  micSender?.pushPcmBytes(new Uint8Array(pcm.buffer));
  updateSentStats();
}

function pushSystemInput(input: Float32Array): void {
  updateLevelStatus("system", getAudioStats(input));
  const pcm = floatToPcmS16leMono(input);
  sysSender?.pushPcmBytes(new Uint8Array(pcm.buffer));
  updateSentStats();
}

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function toneClass(segment: TranscriptSegment): string {
  const label = (segment.speaker_label || "").toLowerCase();
  if (segment.stream_tags.includes("mic") || label === "you") return "tone-you";
  if (segment.speaker_source !== "diarization" && segment.speaker_source !== "memory") return "tone-system";
  const tones = ["tone-a", "tone-b", "tone-c"];
  return tones[hashString(segment.speaker_id || segment.speaker_label) % tones.length];
}

function sourceLabel(segment: TranscriptSegment): string {
  if (segment.stream_tags.includes("mic")) return "Microphone";
  if (segment.stream_tags.includes("system")) return "Meeting audio";
  return segment.stream_tags.join(", ");
}

function speakerLabel(segment: TranscriptSegment): string {
  return segment.speaker_label || segment.speaker_id || sourceLabel(segment);
}

function renderSpeakerLegend(segs: TranscriptSegment[]): void {
  const seen = new Set<string>();
  const chips: HTMLDivElement[] = [];

  for (const segment of segs) {
    const label = speakerLabel(segment);
    if (seen.has(label)) continue;
    seen.add(label);

    const chip = document.createElement("div");
    chip.className = `legendChip ${toneClass(segment)}`;

    const dot = document.createElement("span");
    dot.className = "legendDot";

    chip.append(dot, document.createTextNode(label));
    chips.push(chip);
  }

  speakerLegendEl.replaceChildren(...chips);
}

function setTranscriptViewMode(mode: "bubbles" | "blocks" | "script"): void {
  transcriptViewMode = mode;
  transcriptEl.classList.toggle("view-blocks", mode === "blocks");
  transcriptEl.classList.toggle("view-script", mode === "script");
  viewBubblesBtn.classList.toggle("active", mode === "bubbles");
  viewBlocksBtn.classList.toggle("active", mode === "blocks");
  viewScriptBtn.classList.toggle("active", mode === "script");
  renderTranscript();
}

function segmentsForLayer(layer: TranscriptLayer): Map<string, TranscriptSegment> {
  return layer === "raw" ? rawSegmentsById : processedSegmentsById;
}

function updateLayerCounts(): void {
  rawLayerCountEl.textContent = String(rawSegmentsById.size);
  processedLayerCountEl.textContent = String(processedSegmentsById.size);
  copyTranscriptBtn.disabled = processedSegmentsById.size === 0;
  refreshAiTranscriptState();
}

function reconcileTranscriptMap(segments: Map<string, TranscriptSegment>): void {
  const reconciled = reconcileCrossStreamDuplicates(segments.values());
  segments.clear();
  for (const segment of reconciled) segments.set(segment.id, segment);
}

let copyFeedbackTimer: number | null = null;

function setCopyButtonFeedback(state: "idle" | "copied" | "failed"): void {
  if (copyFeedbackTimer !== null) window.clearTimeout(copyFeedbackTimer);
  copyTranscriptBtn.classList.toggle("copied", state === "copied");
  copyTranscriptBtn.classList.toggle("failed", state === "failed");
  copyTranscriptBtn.innerHTML =
    state === "copied"
      ? "Copied"
      : state === "failed"
        ? "Copy failed"
        : 'Copy<span class="copyLongLabel"> clean</span>';
  if (state !== "idle") {
    copyFeedbackTimer = window.setTimeout(() => {
      copyFeedbackTimer = null;
      setCopyButtonFeedback("idle");
    }, 1800);
  }
}

async function copyCleanProcessedTranscript(): Promise<void> {
  const text = formatCleanTranscript(processedSegmentsById.values());
  if (!text) return;
  try {
    if (window.audioClient?.copyText) {
      const result = await window.audioClient.copyText(text);
      if (!result.copied) throw new Error(result.error || "clipboard write failed");
    } else {
      await navigator.clipboard.writeText(text);
    }
    setCopyButtonFeedback("copied");
  } catch {
    setCopyButtonFeedback("failed");
  }
}

function cleanProcessedTranscript(): string {
  return formatCleanTranscript(processedSegmentsById.values());
}

function currentAiThread(): AiThread | undefined {
  return activeAiThreadId ? aiThreads.find((thread) => thread.id === activeAiThreadId) : undefined;
}

function storeAiThreads(): void {
  saveAiThreads(aiThreads);
}

function aiProviderName(provider: AiProvider): string {
  return provider === "codex" ? "Codex" : "Claude";
}

function updateAiModelOptions(): void {
  const provider = aiProviderSelect.value as AiProvider;
  aiModelSelect.replaceChildren(
    ...AI_MODEL_OPTIONS[provider].map(({ value, label }) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    })
  );
}

function updateAiStatusText(): void {
  if (!aiCliStatus) {
    aiCliStatusEl.textContent = "Checking installed CLIs…";
    aiCliStatusEl.classList.remove("bad");
    return;
  }
  const details = (["codex", "claude"] as AiProvider[]).map((provider) => {
    const status = aiCliStatus?.[provider];
    const name = aiProviderName(provider);
    return status?.available ? `${name} ready${status.version ? ` · ${status.version}` : ""}` : `${name} unavailable`;
  });
  aiCliStatusEl.textContent = details.join("   |   ");
  const selected = aiCliStatus[aiProviderSelect.value as AiProvider];
  aiCliStatusEl.classList.toggle("bad", !selected.available);
}

async function refreshAiCliStatus(): Promise<void> {
  aiRefreshStatusBtn.disabled = true;
  aiCliStatus = null;
  updateAiStatusText();
  try {
    if (!window.audioClient?.getAiCliStatus) throw new Error("AI bridge is unavailable");
    aiCliStatus = await window.audioClient.getAiCliStatus();
  } catch (error) {
    aiCliStatusEl.textContent = `Could not check CLIs: ${error instanceof Error ? error.message : String(error)}`;
    aiCliStatusEl.classList.add("bad");
  } finally {
    aiRefreshStatusBtn.disabled = false;
    if (aiCliStatus) updateAiStatusText();
  }
}

function makeAiMessage(role: AiMessage["role"], text: string): AiMessage {
  return { id: uuidv4(), role, text, createdAt: Date.now() };
}

function addAiMessage(thread: AiThread, role: AiMessage["role"], text: string): void {
  thread.messages.push(makeAiMessage(role, text));
  if (thread.messages.length > 200) thread.messages.splice(0, thread.messages.length - 200);
  thread.updatedAt = Date.now();
}

function createAiThread(): void {
  const provider = aiProviderSelect.value as AiProvider;
  const now = Date.now();
  const thread: AiThread = {
    id: uuidv4(),
    provider,
    model: aiModelSelect.value,
    title: `${aiProviderName(provider)} chat`,
    createdAt: now,
    updatedAt: now,
    messages: []
  };
  aiThreads.push(thread);
  if (aiThreads.length > 20) {
    const removable = aiThreads.find((item) => !runningAiThreads.has(item.id));
    if (removable) aiThreads = aiThreads.filter((item) => item.id !== removable.id);
  }
  activeAiThreadId = thread.id;
  storeAiThreads();
  renderAiWorkspace();
  aiQuestionEl.focus();
}

async function closeAiThread(threadId: string): Promise<void> {
  const running = runningAiThreads.get(threadId);
  if (running) {
    running.cancelling = true;
    renderAiWorkspace();
    await window.audioClient?.cancelAiTurn?.(running.requestId);
  }
  const index = aiThreads.findIndex((thread) => thread.id === threadId);
  if (index < 0) return;
  aiThreads.splice(index, 1);
  if (activeAiThreadId === threadId) {
    activeAiThreadId = aiThreads[Math.min(index, aiThreads.length - 1)]?.id ?? null;
  }
  storeAiThreads();
  renderAiWorkspace();
}

function selectAiThread(threadId: string): void {
  if (!aiThreads.some((thread) => thread.id === threadId)) return;
  activeAiThreadId = threadId;
  renderAiWorkspace();
}

function renderAiTabs(): void {
  aiTabsEl.replaceChildren(
    ...aiThreads.map((thread) => {
      const tab = document.createElement("div");
      const isActive = thread.id === activeAiThreadId;
      tab.className = `aiTab${isActive ? " active" : ""}`;
      tab.setAttribute("role", "presentation");

      const select = document.createElement("button");
      select.type = "button";
      select.className = "aiTabSelect";
      select.textContent = thread.title;
      select.title = `${aiProviderName(thread.provider)} · ${thread.model || "CLI default"} · ${thread.title}`;
      select.setAttribute("role", "tab");
      select.setAttribute("aria-selected", String(isActive));
      select.addEventListener("click", () => selectAiThread(thread.id));

      const close = document.createElement("button");
      close.type = "button";
      close.className = "aiTabClose";
      close.textContent = "×";
      close.setAttribute("aria-label", `Close ${thread.title}`);
      close.addEventListener("click", () => void closeAiThread(thread.id));

      tab.append(select, close);
      return tab;
    })
  );
  aiTabsEl.hidden = aiThreads.length === 0;
}

function renderAiMessages(thread: AiThread): void {
  const wasNearBottom = aiMessagesEl.scrollHeight - aiMessagesEl.clientHeight - aiMessagesEl.scrollTop < 56;
  const messageEls = thread.messages.map((message) => {
    const element = document.createElement("div");
    element.className = `aiMessage ${message.role}`;
    element.textContent = message.text;
    return element;
  });
  const running = runningAiThreads.get(thread.id);
  if (running) {
    const thinking = document.createElement("div");
    thinking.className = "aiThinking";
    thinking.textContent = running.cancelling
      ? `Cancelling ${aiProviderName(thread.provider)}…`
      : `${aiProviderName(thread.provider)} is working…`;
    messageEls.push(thinking);
  }
  aiMessagesEl.replaceChildren(...messageEls);
  if (wasNearBottom || running) aiMessagesEl.scrollTop = aiMessagesEl.scrollHeight;
}

function refreshAiTranscriptState(): void {
  if (aiPanelEl.hidden) return;
  const thread = currentAiThread();
  if (!thread) return;
  const transcript = cleanProcessedTranscript();
  const fingerprint = transcript ? transcriptFingerprint(transcript) : "";
  const running = runningAiThreads.has(thread.id);

  if (!transcript) {
    aiTranscriptStateEl.textContent = "Processed transcript is empty";
  } else if (!thread.remoteThreadId) {
    aiTranscriptStateEl.textContent = `Not sent yet · ${transcript.length.toLocaleString()} characters`;
  } else if (thread.lastTranscriptFingerprint === fingerprint) {
    aiTranscriptStateEl.textContent = `In sync · ${transcript.length.toLocaleString()} characters`;
  } else {
    aiTranscriptStateEl.textContent = `Update ready · ${transcript.length.toLocaleString()} characters now`;
  }
  aiSyncBtn.disabled =
    running || !thread.remoteThreadId || !transcript || thread.lastTranscriptFingerprint === fingerprint;
  updateAiComposerState();
}

function updateAiComposerState(): void {
  const thread = currentAiThread();
  const running = thread ? runningAiThreads.get(thread.id) : undefined;
  const needsInitialTranscript = Boolean(thread && !thread.remoteThreadId);
  aiSendBtn.disabled =
    !thread || Boolean(running) || !aiQuestionEl.value.trim() || (needsInitialTranscript && !cleanProcessedTranscript());
  aiCancelBtn.hidden = !running;
  aiCancelBtn.disabled = Boolean(running?.cancelling);
  aiQuestionEl.disabled = !thread || Boolean(running);
}

function renderAiWorkspace(): void {
  renderAiTabs();
  const thread = currentAiThread();
  aiEmptyEl.hidden = Boolean(thread);
  aiConversationEl.hidden = !thread;
  if (!thread) {
    aiMessagesEl.replaceChildren();
    return;
  }
  renderAiMessages(thread);
  refreshAiTranscriptState();
}

function openAiPanel(): void {
  aiPanelEl.hidden = false;
  renderAiWorkspace();
  if (!aiCliStatus) void refreshAiCliStatus();
  if (currentAiThread()) aiQuestionEl.focus();
}

function closeAiPanel(): void {
  aiPanelEl.hidden = true;
  askAiBtn.focus();
}

async function runAiRequest(syncOnly: boolean): Promise<void> {
  const thread = currentAiThread();
  if (!thread || runningAiThreads.has(thread.id)) return;
  const transcript = cleanProcessedTranscript();
  const fingerprint = transcript ? transcriptFingerprint(transcript) : "";
  const shouldSendTranscript = !thread.remoteThreadId || thread.lastTranscriptFingerprint !== fingerprint;
  const question = syncOnly ? "" : aiQuestionEl.value.trim();
  if (!question && !syncOnly) return;
  if (shouldSendTranscript && !transcript) {
    addAiMessage(thread, "error", "The Processed transcript is empty. Wait for cleaned dialogue before sending.");
    storeAiThreads();
    renderAiWorkspace();
    return;
  }

  if (shouldSendTranscript) {
    addAiMessage(thread, "status", `Sending Processed transcript snapshot · ${transcript.length.toLocaleString()} characters`);
  }
  if (!syncOnly) {
    if (!thread.messages.some((message) => message.role === "user")) thread.title = titleFromQuestion(question);
    addAiMessage(thread, "user", question);
    aiQuestionEl.value = "";
  }

  const request: AiRunRequest = {
    requestId: uuidv4(),
    uiThreadId: thread.id,
    provider: thread.provider,
    model: thread.model,
    remoteThreadId: thread.remoteThreadId,
    transcript: shouldSendTranscript ? transcript : "",
    question,
    syncOnly
  };
  runningAiThreads.set(thread.id, { requestId: request.requestId, cancelling: false });
  storeAiThreads();
  renderAiWorkspace();

  try {
    if (!window.audioClient?.runAiTurn) throw new Error("AI bridge is unavailable");
    const result = await window.audioClient.runAiTurn(request);
    const liveThread = aiThreads.find((item) => item.id === thread.id);
    if (!liveThread) return;
    if (result.ok && result.response && result.remoteThreadId) {
      liveThread.remoteThreadId = result.remoteThreadId;
      if (shouldSendTranscript) {
        liveThread.lastTranscriptFingerprint = fingerprint;
        liveThread.lastTranscriptChars = transcript.length;
      }
      addAiMessage(liveThread, "assistant", result.response);
    } else if (result.cancelled) {
      addAiMessage(liveThread, "status", "Request cancelled.");
    } else {
      addAiMessage(liveThread, "error", result.error || "The AI CLI request failed.");
    }
  } catch (error) {
    const liveThread = aiThreads.find((item) => item.id === thread.id);
    if (liveThread) addAiMessage(liveThread, "error", error instanceof Error ? error.message : String(error));
  } finally {
    runningAiThreads.delete(thread.id);
    storeAiThreads();
    renderAiWorkspace();
  }
}

async function cancelCurrentAiRequest(): Promise<void> {
  const thread = currentAiThread();
  const running = thread ? runningAiThreads.get(thread.id) : undefined;
  if (!running || running.cancelling) return;
  running.cancelling = true;
  renderAiWorkspace();
  try {
    await window.audioClient?.cancelAiTurn?.(running.requestId);
  } catch {
    // The in-flight invocation reports the actionable error in the thread.
  }
}

function isTranscriptNearBottom(): boolean {
  return transcriptEl.scrollHeight - transcriptEl.clientHeight - transcriptEl.scrollTop <= 72;
}

function captureTranscriptScroll(): void {
  const state = transcriptScrollState[transcriptLayer];
  state.scrollTop = transcriptEl.scrollTop;
  state.followLatest = isTranscriptNearBottom();
}

function resetTranscriptScroll(): void {
  transcriptScrollState.raw = { scrollTop: 0, followLatest: true };
  transcriptScrollState.processed = { scrollTop: 0, followLatest: true };
  renderedTranscriptLayer = null;
  jumpLatestBtn.hidden = true;
}

function updateJumpLatestButton(): void {
  jumpLatestBtn.hidden = transcriptScrollState[transcriptLayer].followLatest;
}

function jumpToLatest(): void {
  const state = transcriptScrollState[transcriptLayer];
  state.followLatest = true;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  state.scrollTop = transcriptEl.scrollTop;
  updateJumpLatestButton();
}

function setTranscriptLayer(layer: TranscriptLayer): void {
  if (renderedTranscriptLayer === transcriptLayer) captureTranscriptScroll();
  transcriptLayer = layer;
  const isRaw = layer === "raw";
  layerRawBtn.classList.toggle("active", isRaw);
  layerProcessedBtn.classList.toggle("active", !isRaw);
  layerRawBtn.setAttribute("aria-pressed", String(isRaw));
  layerProcessedBtn.setAttribute("aria-pressed", String(!isRaw));
  transcriptEl.classList.toggle("layer-raw", isRaw);
  transcriptEl.classList.toggle("layer-processed", !isRaw);
  transcriptEl.dataset.emptyMessage = isRaw
    ? "Listening for speech…"
    : "The cleaned transcript follows a few seconds behind.";
  layerDescriptionEl.textContent = isRaw
    ? "Immediate ASR · source labels"
    : "Diarized · joined · canonical record";
  renderTranscript();
}

function appendTranscriptNotice(text: string, kind: "neutral" | "bad" = "neutral"): void {
  const state = transcriptScrollState[transcriptLayer];
  const line = document.createElement("div");
  line.className = `noticeLine ${kind === "bad" ? "bad" : ""}`.trim();
  line.textContent = text;
  transcriptEl.appendChild(line);
  if (state.followLatest) {
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  } else {
    transcriptEl.scrollTop = state.scrollTop;
  }
}

function renderTranscript(): void {
  if (renderedTranscriptLayer === transcriptLayer) captureTranscriptScroll();
  const scrollState = transcriptScrollState[transcriptLayer];
  const segs = [...segmentsForLayer(transcriptLayer).values()].sort((a, b) => {
    if (a.start_sec !== b.start_sec) return a.start_sec - b.start_sec;
    return a.id.localeCompare(b.id);
  });

  renderSpeakerLegend(segs);

  transcriptEl.replaceChildren(
    ...segs.map((s) => {
      const entry = document.createElement("article");
      entry.className = `transcriptEntry ${toneClass(s)} ${s.is_final ? "" : "isPartial"}`.trim();

      const meta = document.createElement("div");
      meta.className = "speakerMeta";

      const speaker = document.createElement("span");
      speaker.className = "speakerName";
      speaker.textContent = speakerLabel(s);

      const time = document.createElement("span");
      time.textContent = `${formatTime(s.start_sec)}-${formatTime(s.end_sec)}`;

      const source = document.createElement("span");
      source.className = "speakerSource";
      source.textContent = sourceLabel(s);

      meta.append(speaker, time, source);

      if (s.final_reason && transcriptViewMode !== "bubbles") {
        const reason = document.createElement("span");
        reason.className = "speakerSource";
        reason.textContent = s.final_reason;
        meta.append(reason);
      }

      const bubble = document.createElement("div");
      bubble.className = "transcriptBubble";
      bubble.textContent = s.text;

      entry.append(meta, bubble);
      return entry;
    })
  );
  renderedTranscriptLayer = transcriptLayer;
  if (scrollState.followLatest) {
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  } else {
    const maxScrollTop = Math.max(0, transcriptEl.scrollHeight - transcriptEl.clientHeight);
    transcriptEl.scrollTop = Math.min(scrollState.scrollTop, maxScrollTop);
  }
  scrollState.scrollTop = transcriptEl.scrollTop;
  updateJumpLatestButton();
}

function requestSpeakerMemoryReview(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    memoryStatusEl.textContent = "not connected";
    memoryStatusEl.classList.add("bad");
    memoryStatusEl.classList.remove("good");
    return;
  }
  memoryStatusEl.textContent = "loading";
  memoryStatusEl.classList.remove("good", "bad");
  sendControl("speaker_memory_review", {});
}

function enrollSpeakerMemorySample(sampleId: string, name: string): void {
  const cleanName = name.trim();
  if (!cleanName) {
    memoryStatusEl.textContent = "name required";
    memoryStatusEl.classList.add("bad");
    memoryStatusEl.classList.remove("good");
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    memoryStatusEl.textContent = "not connected";
    memoryStatusEl.classList.add("bad");
    memoryStatusEl.classList.remove("good");
    return;
  }
  memoryStatusEl.textContent = "saving";
  memoryStatusEl.classList.remove("good", "bad");
  sendControl("speaker_memory_enroll", { sample_id: sampleId, name: cleanName });
}

function renderSpeakerMemoryReview(samples: SpeakerMemoryReviewSample[]): void {
  if (!samples.length) {
    memoryStatusEl.textContent = "no clips";
    memoryStatusEl.classList.remove("good");
    speakerReviewListEl.replaceChildren();
    return;
  }

  memoryStatusEl.textContent = `${samples.length} clip${samples.length === 1 ? "" : "s"}`;
  memoryStatusEl.classList.add("good");
  memoryStatusEl.classList.remove("bad");

  speakerReviewListEl.replaceChildren(
    ...samples.map((sample) => {
      const card = document.createElement("div");
      card.className = "memorySample";

      const head = document.createElement("div");
      head.className = "memorySampleHead";

      const label = document.createElement("span");
      label.textContent = sample.matched_name || sample.speaker_label || sample.speaker_id;

      const timing = document.createElement("span");
      timing.textContent = `${formatTime(sample.start_sec)} · ${sample.duration_sec.toFixed(1)}s`;

      head.append(label, timing);

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = `data:audio/wav;base64,${sample.audio_wav_base64}`;

      const enroll = document.createElement("div");
      enroll.className = "memoryEnroll";

      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Speaker name";
      input.value = sample.matched_name || "";

      const button = document.createElement("button");
      button.className = "actionButton";
      button.type = "button";
      button.textContent = sample.matched_name ? "Update" : "Save";
      button.addEventListener("click", () => enrollSpeakerMemorySample(sample.sample_id, input.value));

      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") enrollSpeakerMemorySample(sample.sample_id, input.value);
      });

      enroll.append(input, button);
      card.append(head, audio, enroll);
      return card;
    })
  );
}

async function requestPermissionOnce(): Promise<void> {
  // On some platforms, labels are empty until permission is granted.
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  stream.getTracks().forEach((t) => t.stop());
}

async function refreshDevices(): Promise<void> {
  try {
    await requestPermissionOnce();
  } catch {
    // Permission denied -> still enumerate devices (labels may be empty)
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((d) => d.kind === "audioinput");

  const prevMic = micSelect.value;
  const prevSys = sysSelect.value;

  function fillSelect(sel: HTMLSelectElement) {
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const d of inputs) {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label ? `${d.label} (${d.deviceId})` : d.deviceId;
      sel.appendChild(opt);
    }
  }

  fillSelect(micSelect);
  fillSelect(sysSelect);

  const loopback = chooseLikelyLoopbackDevice(inputs);
  const mic = chooseLikelyMicDevice(inputs, loopback?.deviceId);

  if (prevMic) {
    micSelect.value = prevMic;
  } else if (mic) {
    micSelect.value = mic.deviceId;
  }

  if (prevSys) {
    sysSelect.value = prevSys;
  } else if (loopback) {
    sysSelect.value = loopback.deviceId;
  }

  if (isConnected()) {
    if (micSelect.value !== prevMic) void replaceStreamCapture("mic", micSelect.value, true);
    if (sysSelect.value !== prevSys) void replaceStreamCapture("system", sysSelect.value, true);
  }
}

async function startCaptureForDevice(
  streamId: StreamId,
  deviceId: string,
  sender: AudioStreamSender
): Promise<CaptureHandle> {
  const mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: deviceId ? { deviceId: { exact: deviceId } } : true
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(mediaStream);

  // ScriptProcessorNode is deprecated but simplest for MVP.
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (ev) => {
    // We take channel 0 (mono). If input is stereo, this is a simple downmix.
    const input = ev.inputBuffer.getChannelData(0);
    if (streamId === "mic") {
      pushMicInput(input);
    } else {
      const pcm = floatToPcmS16leMono(input);
      sender.pushPcmBytes(new Uint8Array(pcm.buffer));
      updateSentStats();
    }
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  async function stop() {
    try {
      processor.disconnect();
    } catch {}
    try {
      source.disconnect();
    } catch {}
    try {
      mediaStream.getTracks().forEach((t) => t.stop());
    } catch {}
    try {
      await audioContext.close();
    } catch {}
  }

  return { streamId, mediaStream, audioContext, source, processor, stop };
}

async function replaceStreamCapture(
  streamId: StreamId,
  deviceId: string,
  announce: boolean
): Promise<void> {
  if (!isConnected() || !ws) return;

  const seq = captureSwitchSeq[streamId] + 1;
  captureSwitchSeq[streamId] = seq;
  await stopCapturesForStream(streamId);

  if (!deviceId) {
    setStreamLevelStatus(streamId, "off");
    if (announce) appendTranscriptNotice(`${sourceLabelForStream(streamId)} source off`);
    return;
  }

  setStreamLevelStatus(streamId, "switching");
  const activeWs = ws;
  const sessionId = sessionIdInput.value.trim();
  const tmpSender = new AudioStreamSender({
    ws: activeWs,
    sessionId,
    streamId,
    sampleRateHz: 48000
  });
  setStreamSender(streamId, tmpSender);

  let handle: CaptureHandle;
  try {
    handle = await startCaptureForDevice(streamId, deviceId, tmpSender);
  } catch (e) {
    if (captureSwitchSeq[streamId] === seq) {
      setStreamSender(streamId, null);
      setStreamLevelStatus(streamId, "source error", "bad");
      appendTranscriptNotice(`${sourceLabelForStream(streamId)} source error: ${String(e)}`, "bad");
    }
    return;
  }

  if (
    captureSwitchSeq[streamId] !== seq ||
    ws !== activeWs ||
    !isConnected() ||
    selectedDeviceForStream(streamId) !== deviceId
  ) {
    await handle.stop();
    return;
  }

  const sender = new AudioStreamSender({
    ws: activeWs,
    sessionId,
    streamId,
    sampleRateHz: handle.audioContext.sampleRate
  });
  setStreamSender(streamId, sender);
  handle.processor.onaudioprocess = (ev) => {
    const input = ev.inputBuffer.getChannelData(0);
    if (streamId === "mic") {
      pushMicInput(input);
    } else {
      pushSystemInput(input);
    }
  };
  captures.push(handle);
  applyMicMuteToTracks();
  setStreamLevelStatus(streamId, "listening", "good");
  if (announce) appendTranscriptNotice(`${sourceLabelForStream(streamId)} source changed`);
}

function sourceLabelForStream(streamId: StreamId): string {
  return streamId === "mic" ? "Microphone" : "Meeting audio";
}

async function start(): Promise<void> {
  const url = wsUrlInput.value.trim() || "ws://localhost:8765";
  const sessionId = sessionIdInput.value.trim() || uuidv4();
  sessionIdInput.value = sessionId;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  rawSegmentsById.clear();
  processedSegmentsById.clear();
  resetTranscriptScroll();
  setTranscriptLayer("raw");
  updateLayerCounts();
  recvCount = 0;
  micSender = null;
  sysSender = null;
  micInputFilter.reset();
  micLastDecision = null;
  sentStatsEl.textContent = "mic=0 sys=0";
  updateMicGateStatus(null);
  updateLevelStatus("mic", null);
  updateLevelStatus("system", null);
  recvStatsEl.textContent = "0";
  speakerLegendEl.replaceChildren();
  speakerReviewListEl.replaceChildren();
  updateSpeakerMemoryStatus(loadSpeakerMemoryProfiles());
  setLastUpdate(null);

  startedAt = Date.now();
  endedAt = null;

  setConnStatus("Connecting…");

  ws = new WebSocket(url);
  ws.onopen = async () => {
    setConnStatus("Connected", "good");
    startBtn.disabled = true;
    stopBtn.disabled = false;
    syncSpeakerMemoryProfiles();

    await replaceStreamCapture("mic", micSelect.value, false);
    await replaceStreamCapture("system", sysSelect.value, false);
  };

  ws.onmessage = (ev) => {
    let msg: unknown;
    try {
      msg = JSON.parse(String(ev.data));
    } catch {
      return;
    }

    const type = (msg as any)?.type;
    if (type === "transcript_update") {
      const tu = msg as TranscriptUpdateMessage;
      const layer = tu.layer === "raw" ? "raw" : "processed";
      const layerSegments = segmentsForLayer(layer);
      for (const s of tu.segments ?? []) {
        layerSegments.set(s.id, s);
      }
      reconcileTranscriptMap(layerSegments);
      updateLayerCounts();
      recvCount += 1;
      recvStatsEl.textContent = String(recvCount);
      setLastUpdate(Date.now());
      if (layer === transcriptLayer) renderTranscript();
    } else if (type === "speaker_memory_review") {
      const review = msg as SpeakerMemoryReviewMessage;
      renderSpeakerMemoryReview(review.samples ?? []);
    } else if (type === "speaker_memory_profile") {
      const profile = msg as SpeakerMemoryProfileMessage;
      storeSpeakerMemoryProfiles(profile.profiles ?? [profile.profile]);
      memoryStatusEl.textContent = `saved ${profile.profile.name}`;
      memoryStatusEl.classList.add("good");
      memoryStatusEl.classList.remove("bad");
    } else if (type === "speaker_memory_profiles") {
      const profiles = msg as SpeakerMemoryProfilesMessage;
      storeSpeakerMemoryProfiles(profiles.profiles ?? []);
    } else if (type === "control") {
      const control = msg as ServerControlMessage;
      if (control.command === "stopped") resolveStopAck?.(true);
    } else if (type === "error") {
      const err = msg as ErrorMessage;
      setConnStatus(`Error: ${err.code}`, "bad");
      appendTranscriptNotice(`${err.code}: ${err.message}`, "bad");
    }
  };

  ws.onerror = () => {
    setConnStatus("Error", "bad");
  };

  ws.onclose = () => {
    resolveStopAck?.(false);
    setConnStatus("Disconnected");
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };
}

async function stop(): Promise<void> {
  endedAt = Date.now();
  captureSwitchSeq.mic += 1;
  captureSwitchSeq.system += 1;

  for (const c of captures) {
    await c.stop();
  }
  captures = [];

  let finalized = true;
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      setConnStatus("Finalizing…");
      const stopAck = waitForStopAck();
      sendControl("stop", {});
      finalized = await stopAck;
      setConnStatus(finalized ? "Finalized" : "Finalization timed out", finalized ? "good" : "bad");
    } catch {
      finalized = false;
    }
  }
  micSender = null;
  sysSender = null;
  updateSentStats();
  setStreamLevelStatus("mic", "idle");
  setStreamLevelStatus("system", "idle");
  micInputFilter.reset();
  micLastDecision = null;
  updateMicGateStatus(null);

  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (!finalized) {
    appendTranscriptNotice(
      "Server finalization did not complete before the transcript was saved; the last phrase may be incomplete.",
      "bad"
    );
  }

  // Save transcript JSON
  const session_id = sessionIdInput.value;
  const segments = [...processedSegmentsById.values()].sort((a, b) => {
    if (a.start_sec !== b.start_sec) return a.start_sec - b.start_sec;
    return a.id.localeCompare(b.id);
  });
  const rawSegments = [...rawSegmentsById.values()].sort((a, b) => {
    if (a.start_sec !== b.start_sec) return a.start_sec - b.start_sec;
    return a.id.localeCompare(b.id);
  });

  const payload = {
    session_id,
    started_at: startedAt,
    ended_at: endedAt,
    segments,
    raw_segments: rawSegments
  };

  const suggestedName = `transcript-${session_id}.json`;
  if (!window.audioClient?.saveTranscript) {
    appendTranscriptNotice(
      "Missing preload bridge (window.audioClient.saveTranscript). The Electron preload script may not be loading.",
      "bad"
    );
    return;
  }

  const res = await window.audioClient.saveTranscript({
    suggestedName,
    jsonText: JSON.stringify(payload, null, 2)
  });

  if (!res.saved && res.error) {
    appendTranscriptNotice(res.error, "bad");
  } else if (res.saved && res.path) {
    appendTranscriptNotice(`Saved ${res.path}`);
  }
}

refreshBtn.addEventListener("click", () => void refreshDevices());
startBtn.addEventListener("click", () => void start());
stopBtn.addEventListener("click", () => void stop());
micSelect.addEventListener("change", () => void replaceStreamCapture("mic", micSelect.value, true));
sysSelect.addEventListener("change", () => void replaceStreamCapture("system", sysSelect.value, true));
memoryReviewBtn.addEventListener("click", () => requestSpeakerMemoryReview());
micMuteBtn.addEventListener("click", () => {
  micMuted = !micMuted;
  micInputFilter.reset();
  applyMicMuteToTracks();
  updateMicMuteUi();
  updateMicGateStatus(null);
});
micFilterEnabledInput.addEventListener("change", () => {
  micInputFilter.reset();
  updateMicGateStatus(null);
});
micGateThresholdInput.addEventListener("input", () => {
  micInputFilter.reset();
  updateMicGateStatus(null);
});
viewBubblesBtn.addEventListener("click", () => setTranscriptViewMode("bubbles"));
viewBlocksBtn.addEventListener("click", () => setTranscriptViewMode("blocks"));
viewScriptBtn.addEventListener("click", () => setTranscriptViewMode("script"));
layerRawBtn.addEventListener("click", () => setTranscriptLayer("raw"));
layerProcessedBtn.addEventListener("click", () => setTranscriptLayer("processed"));
jumpLatestBtn.addEventListener("click", () => jumpToLatest());
copyTranscriptBtn.addEventListener("click", () => void copyCleanProcessedTranscript());
askAiBtn.addEventListener("click", () => openAiPanel());
aiPanelCloseBtn.addEventListener("click", () => closeAiPanel());
aiProviderSelect.addEventListener("change", () => {
  updateAiModelOptions();
  updateAiStatusText();
});
aiNewThreadBtn.addEventListener("click", () => createAiThread());
aiRefreshStatusBtn.addEventListener("click", () => void refreshAiCliStatus());
aiSyncBtn.addEventListener("click", () => void runAiRequest(true));
aiSendBtn.addEventListener("click", () => void runAiRequest(false));
aiCancelBtn.addEventListener("click", () => void cancelCurrentAiRequest());
aiQuestionEl.addEventListener("input", () => updateAiComposerState());
aiQuestionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!aiSendBtn.disabled) void runAiRequest(false);
  }
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !aiPanelEl.hidden) closeAiPanel();
});
transcriptEl.addEventListener("scroll", () => {
  if (renderedTranscriptLayer !== transcriptLayer) return;
  captureTranscriptScroll();
  updateJumpLatestButton();
});

// Initial state
sessionIdInput.value = uuidv4();
setTranscriptViewMode("bubbles");
setTranscriptLayer("raw");
setConnStatus("Disconnected");
updateMicMuteUi();
updateMicGateStatus(null);
updateLayerCounts();
updateSpeakerMemoryStatus(loadSpeakerMemoryProfiles());
updateAiModelOptions();
renderAiWorkspace();
void refreshDevices();
