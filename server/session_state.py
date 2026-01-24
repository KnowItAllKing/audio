from __future__ import annotations

from dataclasses import dataclass, field

from .protocol_types import StreamId


@dataclass
class SessionState:
    session_id: str

    # Raw PCM bytes retained in chronological order.
    audio_buffer: bytearray = field(default_factory=bytearray)

    # Absolute byte offset (cumulative) corresponding to audio_buffer[0].
    buffer_start_offset_bytes: int = 0

    # Last seq observed (best-effort; not strictly enforced).
    last_seq: int = -1

    # Absolute byte offset up to which we've "committed" transcription.
    last_transcribed_offset_bytes: int = 0

    created_at: float = 0.0
    updated_at: float = 0.0

    # Audio format (locked to the first received chunk)
    sample_rate_hz: int = 16000
    num_channels: int = 1

    # Track which streams have been observed for tagging.
    streams_seen: set[StreamId] = field(default_factory=set)

    def bytes_per_second(self) -> int:
        # pcm_s16le = 2 bytes per sample per channel
        return int(self.sample_rate_hz) * int(self.num_channels) * 2

    def buffer_end_offset_bytes(self) -> int:
        return self.buffer_start_offset_bytes + len(self.audio_buffer)

    def trim_to_max_bytes(self, max_bytes: int) -> None:
        """
        Keep only the newest `max_bytes` of audio. Adjust absolute offsets accordingly.
        """
        if max_bytes <= 0:
            # Degenerate: keep nothing.
            dropped = len(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_start_offset_bytes += dropped
            # Ensure we don't point into dropped audio.
            self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)
            return

        if len(self.audio_buffer) <= max_bytes:
            return

        drop = len(self.audio_buffer) - max_bytes
        del self.audio_buffer[:drop]
        self.buffer_start_offset_bytes += drop
        self.last_transcribed_offset_bytes = max(self.last_transcribed_offset_bytes, self.buffer_start_offset_bytes)

