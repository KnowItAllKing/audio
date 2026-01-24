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

    # Absolute byte offset up to which we've "committed" transcription.
    last_transcribed_offset_bytes: int = 0

    # Client-provided epoch (ms since Unix epoch) for the first chunk.
    # Used to anchor transcript timestamps to the client's wall clock.
    first_timestamp_ms: Optional[int] = None

    def buffer_end_offset_bytes(self) -> int:
        return self.buffer_start_offset_bytes + len(self.audio_buffer)

    def trim_to_max_bytes(self, max_bytes: int) -> None:
        """
        Keep only the newest `max_bytes` of audio. Adjust absolute offsets accordingly.
        """
        if max_bytes <= 0:
            dropped = len(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_start_offset_bytes += dropped
            self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)
            return

        if len(self.audio_buffer) <= max_bytes:
            return

        drop = len(self.audio_buffer) - max_bytes
        del self.audio_buffer[:drop]
        self.buffer_start_offset_bytes += drop
        self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)


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
