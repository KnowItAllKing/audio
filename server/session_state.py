from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .protocol_types import StreamId


@dataclass()
class StreamBuffer:
    """Per-stream audio buffer with tracking metadata."""

    stream_id: StreamId

    # Raw PCM bytes retained in chronological order.
    audio_buffer: bytearray = field(default_factory=bytearray)

    # Absolute byte offset (cumulative) corresponding to audio_buffer[0].
    buffer_start_offset_bytes: int = 0

    # Last seq observed (best-effort; not strictly enforced).
    last_seq: int = -1

    # Absolute byte offset up to which low-latency ASR has run.
    last_transcribed_offset_bytes: int = 0

    # Absolute byte offset promoted into the processed transcript. This trails
    # raw ASR while future context, diarization, and joining are pending.
    last_processed_offset_bytes: int = 0

    # Client-provided epoch (ms since Unix epoch) for the first chunk.
    # Used to anchor transcript timestamps to the client's wall clock.
    first_timestamp_ms: Optional[int] = None

    # Server wall time when the newest chunk arrived. This lets pause flushing
    # advance a client-relative audio timeline even while no silence is sent.
    last_received_at: Optional[float] = None

    def buffer_end_offset_bytes(self) -> int:
        return self.buffer_start_offset_bytes + len(self.audio_buffer)

    def timeline_now_sec(self, *, bytes_per_sec: int, wall_now_sec: float) -> float:
        base_sec = float(self.first_timestamp_ms or 0) / 1000.0
        processed_end_sec = base_sec
        if bytes_per_sec > 0:
            processed_end_sec += float(self.last_processed_offset_bytes) / float(bytes_per_sec)
        caught_up = self.last_processed_offset_bytes >= self.buffer_end_offset_bytes()
        if caught_up and self.last_received_at is not None:
            processed_end_sec += max(0.0, wall_now_sec - self.last_received_at)
        return processed_end_sec

    def trim_to_max_bytes(self, max_bytes: int) -> None:
        """
        Keep only the newest `max_bytes` of audio. Adjust absolute offsets accordingly.
        """
        if max_bytes <= 0:
            dropped = len(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_start_offset_bytes += dropped
            self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)
            self.last_processed_offset_bytes = max(self.last_processed_offset_bytes, self.buffer_start_offset_bytes)
            return

        if len(self.audio_buffer) <= max_bytes:
            return

        drop = len(self.audio_buffer) - max_bytes
        del self.audio_buffer[:drop]
        self.buffer_start_offset_bytes += drop
        self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)
        self.last_processed_offset_bytes = max(self.last_processed_offset_bytes, self.buffer_start_offset_bytes)


@dataclass()
class SessionState:
    session_id: str

    # Per-stream buffers
    mic_buffer: StreamBuffer = field(default_factory=lambda: StreamBuffer(stream_id="mic"))
    system_buffer: StreamBuffer = field(default_factory=lambda: StreamBuffer(stream_id="system"))

    created_at: float = 0.0
    updated_at: float = 0.0

    # Audio format (locked to the first received chunk)
    sample_rate_hz: int = 16000
    num_channels: int = 1

    def get_buffer(self, stream_id: StreamId) -> StreamBuffer:
        if stream_id == "mic":
            return self.mic_buffer
        return self.system_buffer

    def bytes_per_second(self) -> int:
        # pcm_s16le = 2 bytes per sample per channel
        return int(self.sample_rate_hz) * int(self.num_channels) * 2
