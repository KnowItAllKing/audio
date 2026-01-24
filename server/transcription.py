from __future__ import annotations

from typing import Tuple


def bytes_per_second(sample_rate_hz: int, num_channels: int) -> int:
    """
    PCM S16LE bytes per second.
    - 2 bytes per sample per channel
    """
    return int(sample_rate_hz) * int(num_channels) * 2


def eligible_for_transcription(
    buffer_end_offset_bytes: int,
    last_transcribed_offset_bytes: int,
    min_new_bytes: int,
) -> bool:
    return (buffer_end_offset_bytes - last_transcribed_offset_bytes) >= int(min_new_bytes)


def compute_window(
    last_transcribed_offset_bytes: int,
    buffer_end_offset_bytes: int,
    window_bytes: int,
) -> Tuple[int, int]:
    """
    Returns (window_start, window_end) in absolute byte offsets.
    """
    window_end = int(buffer_end_offset_bytes)
    window_start = max(int(last_transcribed_offset_bytes), window_end - int(window_bytes))
    return window_start, window_end

