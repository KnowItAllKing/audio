export type StreamId = "mic" | "system";

export type AudioChunkMessage = {
  type: "audio_chunk";
  session_id: string;
  stream_id: StreamId;
  seq: number;
  timestamp_ms: number;
  audio_format: {
    encoding: "pcm_s16le";
    sample_rate_hz: number;
    num_channels: 1;
  };
  audio_base64: string;
};

export type AudioStreamSenderConfig = {
  ws: WebSocket;
  sessionId: string;
  streamId: StreamId;
  sampleRateHz: number;
  chunkDurationMs?: number; // default 250ms
  maxBufferedAmountBytes?: number; // default 5MB
  // For future: remember recent sent chunks for ack/retry (not implemented).
  maxUnacked?: number; // default 256
};

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export class AudioStreamSender {
  private readonly ws: WebSocket;
  private readonly sessionId: string;
  private readonly streamId: StreamId;
  private readonly sampleRateHz: number;
  private readonly chunkDurationMs: number;
  private readonly maxBufferedAmountBytes: number;
  private readonly maxUnacked: number;

  private seq = 0;
  private pending = new Uint8Array(0);

  // Retry seams (no logic yet)
  private readonly sentUnacked = new Map<number, Uint8Array>();

  private sentCount = 0;

  constructor(cfg: AudioStreamSenderConfig) {
    this.ws = cfg.ws;
    this.sessionId = cfg.sessionId;
    this.streamId = cfg.streamId;
    this.sampleRateHz = cfg.sampleRateHz;
    this.chunkDurationMs = cfg.chunkDurationMs ?? 250;
    this.maxBufferedAmountBytes = cfg.maxBufferedAmountBytes ?? 5_000_000;
    this.maxUnacked = cfg.maxUnacked ?? 256;
  }

  getStreamId(): StreamId {
    return this.streamId;
  }

  getSeq(): number {
    return this.seq;
  }

  getSentCount(): number {
    return this.sentCount;
  }

  // Placeholder for future ack logic.
  onAck(lastSeqReceived: number): void {
    for (const k of this.sentUnacked.keys()) {
      if (k <= lastSeqReceived) this.sentUnacked.delete(k);
    }
  }

  // Push raw PCM S16LE bytes for this stream (mono).
  pushPcmBytes(pcmBytes: Uint8Array): void {
    if (pcmBytes.length === 0) return;
    if (this.pending.length === 0) {
      this.pending = pcmBytes;
    } else {
      const merged = new Uint8Array(this.pending.length + pcmBytes.length);
      merged.set(this.pending, 0);
      merged.set(pcmBytes, this.pending.length);
      this.pending = merged;
    }

    // Attempt to flush multiple chunks if we have enough.
    for (let i = 0; i < 32; i++) {
      if (!this.tryFlushOne()) break;
    }
  }

  private tryFlushOne(): boolean {
    if (this.ws.readyState !== WebSocket.OPEN) return false;

    const bytesPerMs = (this.sampleRateHz * 2) / 1000; // s16le mono
    const targetBytes = Math.max(1, Math.floor(bytesPerMs * this.chunkDurationMs));
    if (this.pending.length < targetBytes) return false;

    // Backpressure (MVP): drop if socket buffer is too big.
    if (this.ws.bufferedAmount > this.maxBufferedAmountBytes) {
      // Drop one chunk worth to avoid unbounded growth.
      this.pending = this.pending.slice(targetBytes);
      return true;
    }

    const chunk = this.pending.slice(0, targetBytes);
    this.pending = this.pending.slice(targetBytes);

    const msg: AudioChunkMessage = {
      type: "audio_chunk",
      session_id: this.sessionId,
      stream_id: this.streamId,
      seq: this.seq,
      timestamp_ms: Date.now(),
      audio_format: {
        encoding: "pcm_s16le",
        sample_rate_hz: this.sampleRateHz,
        num_channels: 1
      },
      audio_base64: uint8ToBase64(chunk)
    };

    this.ws.send(JSON.stringify(msg));

    // Retry seam: store sent bytes by seq (bounded).
    this.sentUnacked.set(this.seq, chunk);
    if (this.sentUnacked.size > this.maxUnacked) {
      const oldest = this.sentUnacked.keys().next().value as number | undefined;
      if (oldest !== undefined) this.sentUnacked.delete(oldest);
    }

    this.seq += 1;
    this.sentCount += 1;
    return true;
  }
}

