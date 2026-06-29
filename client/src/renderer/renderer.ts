import { AudioStreamSender, type StreamId } from "./audio/AudioStreamSender";
import {
  chooseLikelyLoopbackDevice,
  chooseLikelyMicDevice,
  speakerIdFromLabel
} from "./audio/LoopbackDevice";
import { MicInputFilter, getAudioStats, type MicInputFilterDecision } from "./audio/MicInputFilter";
import {
  loadSpeakerMemoryProfiles,
  saveSpeakerMemoryProfiles,
  type SpeakerMemoryProfile
} from "./SpeakerMemoryStore";

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

type TranscriptUpdateMessage = {
  type: "transcript_update";
  session_id: string;
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
    | "metadata"
    | "speaker_activity"
    | "stop"
    | "ping"
    | "speaker_memory_review"
    | "speaker_memory_enroll"
    | "speaker_memory_profiles";
  payload?: Record<string, unknown>;
};

// AudioChunkMessage type lives in AudioStreamSender module.

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const wsUrlInput = $<HTMLInputElement>("wsUrl");
const sessionIdInput = $<HTMLInputElement>("sessionId");
const micSelect = $<HTMLSelectElement>("micSelect");
const sysSelect = $<HTMLSelectElement>("sysSelect");
const micLabelInput = $<HTMLInputElement>("micLabel");
const sysLabelInput = $<HTMLInputElement>("sysLabel");
const activeSpeakerLabelInput = $<HTMLInputElement>("activeSpeakerLabel");
const activeSpeakerBtn = $<HTMLButtonElement>("activeSpeakerBtn");
const activeSpeakerStatusEl = $<HTMLSpanElement>("activeSpeakerStatus");
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

const segmentsById = new Map<string, TranscriptSegment>();
let recvCount = 0;
let transcriptViewMode: "bubbles" | "blocks" | "script" = "bubbles";

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

function sendSpeakerMetadata(): void {
  const micLabel = micLabelInput.value.trim() || "You";
  const sysLabel = sysLabelInput.value.trim() || "System audio";
  sendControl("metadata", {
    stream_speakers: {
      mic: {
        speaker_id: "local:mic",
        speaker_label: micLabel,
        speaker_source: "manual",
        speaker_confidence: 0.9
      },
      system: {
        speaker_id: "system:unknown",
        speaker_label: sysLabel,
        speaker_source: "manual",
        speaker_confidence: 0.5
      }
    }
  });
}

function sendActiveSpeakerOverride(): void {
  const label = activeSpeakerLabelInput.value.trim();
  if (!label) {
    activeSpeakerStatusEl.textContent = "idle";
    activeSpeakerStatusEl.classList.remove("good", "bad");
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    activeSpeakerStatusEl.textContent = "not connected";
    activeSpeakerStatusEl.classList.add("bad");
    activeSpeakerStatusEl.classList.remove("good");
    return;
  }
  sendControl("speaker_activity", {
    stream_id: "system",
    speaker_id: speakerIdFromLabel(label),
    speaker_label: label,
    speaker_source: "manual",
    speaker_confidence: 0.8,
    start_timestamp_ms: Date.now()
  });
  activeSpeakerStatusEl.textContent = label;
  activeSpeakerStatusEl.classList.add("good");
  activeSpeakerStatusEl.classList.remove("bad");
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

function appendTranscriptNotice(text: string, kind: "neutral" | "bad" = "neutral"): void {
  const line = document.createElement("div");
  line.className = `noticeLine ${kind === "bad" ? "bad" : ""}`.trim();
  line.textContent = text;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderTranscript(): void {
  const segs = [...segmentsById.values()].sort((a, b) => {
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
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
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

  segmentsById.clear();
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
  transcriptEl.replaceChildren();
  speakerLegendEl.replaceChildren();
  speakerReviewListEl.replaceChildren();
  updateSpeakerMemoryStatus(loadSpeakerMemoryProfiles());
  activeSpeakerStatusEl.textContent = "idle";
  activeSpeakerStatusEl.classList.remove("good", "bad");
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
    sendSpeakerMetadata();

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
      for (const s of tu.segments ?? []) {
        segmentsById.set(s.id, s);
      }
      recvCount += 1;
      recvStatsEl.textContent = String(recvCount);
      setLastUpdate(Date.now());
      renderTranscript();
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

  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      sendControl("stop", {});
      sendControl("speaker_memory_review", {});
    } catch {}
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

  // Save transcript JSON
  const session_id = sessionIdInput.value;
  const segments = [...segmentsById.values()].sort((a, b) => {
    if (a.start_sec !== b.start_sec) return a.start_sec - b.start_sec;
    return a.id.localeCompare(b.id);
  });

  const payload = {
    session_id,
    started_at: startedAt,
    ended_at: endedAt,
    segments
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
activeSpeakerBtn.addEventListener("click", () => sendActiveSpeakerOverride());
activeSpeakerLabelInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") sendActiveSpeakerOverride();
});
viewBubblesBtn.addEventListener("click", () => setTranscriptViewMode("bubbles"));
viewBlocksBtn.addEventListener("click", () => setTranscriptViewMode("blocks"));
viewScriptBtn.addEventListener("click", () => setTranscriptViewMode("script"));

// Initial state
sessionIdInput.value = uuidv4();
setTranscriptViewMode("bubbles");
setConnStatus("Disconnected");
updateMicMuteUi();
updateMicGateStatus(null);
updateSpeakerMemoryStatus(loadSpeakerMemoryProfiles());
void refreshDevices();
