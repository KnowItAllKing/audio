from __future__ import annotations

from server.session_state import SessionState
from server.transcription import bytes_per_second, compute_window, eligible_for_transcription


def test_bytes_per_second_pcm_s16le_mono() -> None:
    assert bytes_per_second(16000, 1) == 32000


def test_eligibility_threshold() -> None:
    assert eligible_for_transcription(1000, 0, 999) is True
    assert eligible_for_transcription(1000, 0, 1000) is True
    assert eligible_for_transcription(1000, 0, 1001) is False


def test_compute_window_basic() -> None:
    start, end = compute_window(0, 1000, 200)
    assert (start, end) == (800, 1000)


def test_compute_window_respects_last_transcribed() -> None:
    start, end = compute_window(900, 1000, 200)
    assert (start, end) == (900, 1000)


def test_multiple_sessions_independent_buffers() -> None:
    a = SessionState(session_id="a", sample_rate_hz=16000, num_channels=1)
    b = SessionState(session_id="b", sample_rate_hz=16000, num_channels=1)

    a.mic_buffer.audio_buffer.extend(b"\x00" * 100)
    b.mic_buffer.audio_buffer.extend(b"\x00" * 10)

    assert a.mic_buffer.buffer_end_offset_bytes() == 100
    assert b.mic_buffer.buffer_end_offset_bytes() == 10

