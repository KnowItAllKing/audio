import { AudioStreamSender, type StreamId } from "./audio/AudioStreamSender";

type TranscriptSegment = {
  id: string;
  start_sec: number;
  end_sec: number;
  text: string;
  stream_tags: StreamId[];
  is_final: boolean;
  full_context_available: boolean;
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

// AudioChunkMessage type lives in AudioStreamSender module.

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const wsUrlInput = $<HTMLInputElement>("wsUrl");
const sessionIdInput = $<HTMLInputElement>("sessionId");
const micSelect = $<HTMLSelectElement>("micSelect");
const sysSelect = $<HTMLSelectElement>("sysSelect");
const refreshBtn = $<HTMLButtonElement>("refreshBtn");
const startBtn = $<HTMLButtonElement>("startBtn");
const stopBtn = $<HTMLButtonElement>("stopBtn");
const transcriptEl = $<HTMLDivElement>("transcript");
const connStatusEl = $<HTMLSpanElement>("connStatus");
const lastUpdateEl = $<HTMLSpanElement>("lastUpdate");
const sentStatsEl = $<HTMLSpanElement>("sentStats");
const recvStatsEl = $<HTMLSpanElement>("recvStats");

function uuidv4(): string {
  // Browser-safe UUID (Chromium supports crypto.randomUUID)
  if ("randomUUID" in crypto) return crypto.randomUUID();
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
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

let micSender: AudioStreamSender | null = null;
let sysSender: AudioStreamSender | null = null;

let captures: CaptureHandle[] = [];

function renderTranscript(): void {
  const segs = [...segmentsById.values()].sort((a, b) => {
    if (a.start_sec !== b.start_sec) return a.start_sec - b.start_sec;
    return a.id.localeCompare(b.id);
  });

  transcriptEl.textContent = segs
    .map((s) => {
      const tags = s.stream_tags.join(",");
      return `[${formatTime(s.start_sec)}–${formatTime(s.end_sec)}] [${tags}] ${s.text}`;
    })
    .join("\n");
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

  if (prevMic) micSelect.value = prevMic;
  if (prevSys) sysSelect.value = prevSys;
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
    const pcm = floatToPcmS16leMono(input);
    sender.pushPcmBytes(new Uint8Array(pcm.buffer));
    sentStatsEl.textContent = `mic=${micSender?.getSentCount() ?? 0} sys=${sysSender?.getSentCount() ?? 0}`;
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

async function start(): Promise<void> {
  const url = wsUrlInput.value.trim() || "ws://localhost:8765";
  const sessionId = sessionIdInput.value.trim() || uuidv4();
  sessionIdInput.value = sessionId;

  segmentsById.clear();
  recvCount = 0;
  micSender = null;
  sysSender = null;
  sentStatsEl.textContent = "mic=0 sys=0";
  recvStatsEl.textContent = "0";
  transcriptEl.textContent = "";
  setLastUpdate(null);

  startedAt = Date.now();
  endedAt = null;

  setConnStatus("Connecting…");

  ws = new WebSocket(url);
  ws.onopen = async () => {
    setConnStatus("Connected", "good");
    startBtn.disabled = true;
    stopBtn.disabled = false;

    const micId = micSelect.value;
    const sysId = sysSelect.value;

    const handles: CaptureHandle[] = [];
    if (!ws) return;

    // Create senders (chunkDurationMs default is 250ms).
    // The sender wants a stable sample rate; we create the AudioContext first and pass its sample rate.
    // We create per-stream sender inside the capture start function after AudioContext is created.
    if (micId) {
      // temporary sender; will be recreated once we know the actual AudioContext sampleRate
      const tmp = new AudioStreamSender({
        ws,
        sessionId,
        streamId: "mic",
        sampleRateHz: 48000
      });
      micSender = tmp;
      const handle = await startCaptureForDevice("mic", micId, tmp);
      // Replace sender with accurate sampleRate from the created context.
      micSender = new AudioStreamSender({
        ws,
        sessionId,
        streamId: "mic",
        sampleRateHz: handle.audioContext.sampleRate
      });
      // Point processor to the new sender.
      handle.processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        const pcm = floatToPcmS16leMono(input);
        micSender?.pushPcmBytes(new Uint8Array(pcm.buffer));
        sentStatsEl.textContent = `mic=${micSender?.getSentCount() ?? 0} sys=${sysSender?.getSentCount() ?? 0}`;
      };
      handles.push(handle);
    }
    if (sysId) {
      const tmp = new AudioStreamSender({
        ws,
        sessionId,
        streamId: "system",
        sampleRateHz: 48000
      });
      sysSender = tmp;
      const handle = await startCaptureForDevice("system", sysId, tmp);
      sysSender = new AudioStreamSender({
        ws,
        sessionId,
        streamId: "system",
        sampleRateHz: handle.audioContext.sampleRate
      });
      handle.processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        const pcm = floatToPcmS16leMono(input);
        sysSender?.pushPcmBytes(new Uint8Array(pcm.buffer));
        sentStatsEl.textContent = `mic=${micSender?.getSentCount() ?? 0} sys=${sysSender?.getSentCount() ?? 0}`;
      };
      handles.push(handle);
    }

    captures = handles;
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
    } else if (type === "error") {
      const err = msg as ErrorMessage;
      setConnStatus(`Error: ${err.code}`, "bad");
      const line = `\n[error] ${err.code}: ${err.message}\n`;
      transcriptEl.textContent = (transcriptEl.textContent ?? "") + line;
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

  for (const c of captures) {
    await c.stop();
  }
  captures = [];

  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.close();
    } catch {}
  }
  ws = null;
  micSender = null;
  sysSender = null;

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
    transcriptEl.textContent =
      (transcriptEl.textContent ?? "") +
      "\n[save error] Missing preload bridge (window.audioClient.saveTranscript). " +
      "The Electron preload script may not be loading.\n";
    return;
  }

  const res = await window.audioClient.saveTranscript({
    suggestedName,
    jsonText: JSON.stringify(payload, null, 2)
  });

  if (!res.saved && res.error) {
    transcriptEl.textContent =
      (transcriptEl.textContent ?? "") + `\n[save error] ${res.error}\n`;
  } else if (res.saved && res.path) {
    transcriptEl.textContent =
      (transcriptEl.textContent ?? "") + `\n[saved] ${res.path}\n`;
  }
}

refreshBtn.addEventListener("click", () => void refreshDevices());
startBtn.addEventListener("click", () => void start());
stopBtn.addEventListener("click", () => void stop());

// Initial state
sessionIdInput.value = uuidv4();
setConnStatus("Disconnected");
void refreshDevices();

