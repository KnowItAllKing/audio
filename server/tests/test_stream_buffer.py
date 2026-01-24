from __future__ import annotations

from server.session_state import StreamBuffer


def test_append_increases_buffer_size() -> None:
    buf = StreamBuffer(stream_id="mic")
    assert len(buf.audio_buffer) == 0
    buf.audio_buffer.extend(b"\x00\x01\x02")
    assert len(buf.audio_buffer) == 3
    assert buf.buffer_end_offset_bytes() == 3


def test_trim_drops_oldest_and_adjusts_offsets() -> None:
    buf = StreamBuffer(stream_id="mic")
    buf.audio_buffer.extend(b"abcdef")
    buf.last_transcribed_offset_bytes = 0

    buf.trim_to_max_bytes(3)

    assert bytes(buf.audio_buffer) == b"def"
    assert buf.buffer_start_offset_bytes == 3
    # last_transcribed_offset can't point into dropped region
    assert buf.last_transcribed_offset_bytes == 3


def test_trim_to_zero_keeps_offsets_consistent() -> None:
    buf = StreamBuffer(stream_id="mic")
    buf.audio_buffer.extend(b"abcdef")
    buf.last_transcribed_offset_bytes = 1

    buf.trim_to_max_bytes(0)

    assert bytes(buf.audio_buffer) == b""
    assert buf.buffer_start_offset_bytes == 6
    assert buf.last_transcribed_offset_bytes == 6

