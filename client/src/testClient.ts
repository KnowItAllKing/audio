import WebSocket, { type RawData } from "ws";
import { randomBytes, randomUUID } from "node:crypto";

type StreamId = "mic" | "system";

type AudioChunkMessage = {
  type: "audio_chunk";
  session_id: string;
  stream_id: StreamId;
  seq: number;
  timestamp_ms: number;
  audio_format: {
    encoding: "pcm_s16le";
    sample_rate_hz: number;
    num_channels: number;
  };
  audio_base64: string;
};

type TranscriptUpdateMessage = {
  type: "transcript_update";
  session_id: string;
  segments: Array<{
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
  }>;
};

type ErrorMessage = {
  type: "error";
  session_id: string;
  code: string;
  message: string;
};

function getArg(name: string): string | undefined {
  const prefix = `--${name}=`;
  const hit = process.argv.find((a: string) => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : undefined;
}

function getInt(value: string | undefined): number | undefined {
  if (value == null) return undefined;
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : undefined;
}

function msToHuman(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}m${String(r).padStart(2, "0")}s`;
}

const url = getArg("url") ?? process.env.WS_URL ?? "ws://localhost:8765";
const intervalMs =
  getInt(getArg("interval-ms")) ??
  getInt(process.env.CHUNK_INTERVAL_MS) ??
  100;
const durationSec =
  getInt(getArg("duration-sec")) ??
  getInt(process.env.DURATION_SEC) ??
  600; // 10 minutes

const sessionId = (() => {
  try {
    return randomUUID();
  } catch {
    // Older Node fallback (shouldn't happen on modern Node)
    return randomBytes(16).toString("hex");
  }
})();

const streamId: StreamId = "mic";

let seq = 0;
let sent = 0;
let received = 0;
const startedAt = Date.now();

console.log(
  `connecting: url=${url} session_id=${sessionId} interval_ms=${intervalMs} duration_sec=${durationSec}`
);

const ws = new WebSocket(url);

function rawDataToString(data: RawData): string {
  if (typeof data === "string") return data;
  if (Buffer.isBuffer(data)) return data.toString("utf8");
  if (data instanceof ArrayBuffer) return Buffer.from(data).toString("utf8");
  // Buffer[]
  return Buffer.concat(data).toString("utf8");
}

function sendChunk(): void {
  // Dummy audio payload: random bytes (pretend pcm_s16le).
  // 320 bytes ~= 160 samples @ 2 bytes/sample (10ms @ 16kHz mono) — just a plausible size.
  const dummy = randomBytes(320);
  const msg: AudioChunkMessage = {
    type: "audio_chunk",
    session_id: sessionId,
    stream_id: streamId,
    seq,
    timestamp_ms: Date.now(),
    audio_format: {
      encoding: "pcm_s16le",
      sample_rate_hz: 16000,
      num_channels: 1
    },
    audio_base64: dummy.toString("base64")
  };

  ws.send(JSON.stringify(msg));
  sent += 1;
  seq += 1;
}

function logStatus(): void {
  const uptimeMs = Date.now() - startedAt;
  console.log(
    `status: sent=${sent}, received=${received}, uptime=${msToHuman(uptimeMs)}`
  );
}

let sendTimer: ReturnType<typeof setInterval> | undefined;
let statusTimer: ReturnType<typeof setInterval> | undefined;
let stopTimer: ReturnType<typeof setTimeout> | undefined;

ws.on("open", () => {
  console.log("connected");
  ws.send(
    JSON.stringify({
      type: "control",
      session_id: sessionId,
      command: "metadata",
      payload: {
        stream_speakers: {
          mic: {
            speaker_id: "test:mic",
            speaker_label: "Test mic",
            speaker_source: "manual",
            speaker_confidence: 0.9
          }
        }
      }
    })
  );

  sendTimer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    try {
      sendChunk();
    } catch (e) {
      console.error("send failed:", e);
    }
  }, intervalMs);

  statusTimer = setInterval(logStatus, 30_000);

  stopTimer = setTimeout(() => {
    console.log("duration reached; closing");
    try {
      ws.close(1000, "done");
    } catch {}
  }, durationSec * 1000);
});

ws.on("message", (data: RawData) => {
  const raw = rawDataToString(data);
  let msg: unknown;
  try {
    msg = JSON.parse(raw);
  } catch (e) {
    console.error("received non-json message:", e);
    return;
  }

  if (typeof msg !== "object" || msg == null) {
    console.error("received non-object message:", msg);
    return;
  }

  const t = (msg as { type?: unknown }).type;
  if (t === "transcript_update") {
    const tu = msg as TranscriptUpdateMessage;
    received += 1;
    const seg0 = tu.segments?.[0];
    const text = seg0?.text ?? "<no text>";
    const speaker = seg0?.speaker_label ?? "<no speaker>";
    const final = seg0?.is_final ? "final" : "partial";
    console.log(`transcript_update: received=${received} speaker=${JSON.stringify(speaker)} ${final} text=${JSON.stringify(text)}`);
  } else if (t === "error") {
    const err = msg as ErrorMessage;
    console.error(`server_error: code=${err.code} message=${err.message}`);
  } else {
    console.log("server_message:", raw);
  }
});

ws.on("close", (code: number, reason: Buffer) => {
  console.log(`disconnected: code=${code} reason=${reason.toString()}`);
  if (sendTimer) clearInterval(sendTimer);
  if (statusTimer) clearInterval(statusTimer);
  if (stopTimer) clearTimeout(stopTimer);
  logStatus();
  process.exitCode = 0;
});

ws.on("error", (err: Error) => {
  console.error("ws error:", err);
});
