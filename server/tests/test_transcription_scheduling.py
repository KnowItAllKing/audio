from __future__ import annotations

import asyncio
import json
import time

import numpy as np

from server.diarization import DiarizationTurn
from server.main import ServerState, _transcribe_stream
from server.session_state import SessionState
from server.transcription import bytes_per_second, compute_window, eligible_for_transcription
from server.whisper_backend import TranscriptSegment


class _RecordingSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


def test_bytes_per_second_pcm_s16le_mono() -> None:
    assert bytes_per_second(16000, 1) == 32000


def test_eligibility_threshold() -> None:
    assert eligible_for_transcription(1000, 0, 999) is True
    assert eligible_for_transcription(1000, 0, 1000) is True
    assert eligible_for_transcription(1000, 0, 1001) is False


def test_compute_window_basic() -> None:
    start, end = compute_window(0, 1000, 200)
    assert (start, end) == (0, 200)


def test_compute_window_respects_last_transcribed() -> None:
    start, end = compute_window(900, 1000, 200)
    assert (start, end) == (900, 1000)


def test_compute_window_does_not_skip_audio_when_more_than_one_window_is_pending() -> None:
    start, end = compute_window(200, 1000, 300)
    assert (start, end) == (200, 500)


def test_multiple_sessions_independent_buffers() -> None:
    a = SessionState(session_id="a", sample_rate_hz=16000, num_channels=1)
    b = SessionState(session_id="b", sample_rate_hz=16000, num_channels=1)

    a.mic_buffer.audio_buffer.extend(b"\x00" * 100)
    b.mic_buffer.audio_buffer.extend(b"\x00" * 10)

    assert a.mic_buffer.buffer_end_offset_bytes() == 100
    assert b.mic_buffer.buffer_end_offset_bytes() == 10


def test_diarization_uses_future_context_before_finalizing_current_window(monkeypatch) -> None:
    class RecordingDiarizer:
        def __init__(self) -> None:
            self.sample_counts: list[int] = []

        def diarize(self, samples: np.ndarray, sample_rate: int) -> list[DiarizationTurn]:
            self.sample_counts.append(samples.size)
            return [DiarizationTurn(0.0, samples.size / sample_rate, "local:a", "A")]

    class FakeBackend:
        def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
            return [TranscriptSegment(0.0, samples.size / sample_rate, "hello.", ["system"])]

    monkeypatch.setenv("WHISPER_DISABLE", "1")
    monkeypatch.setenv("DIARIZATION_STREAMS", "system")
    monkeypatch.setenv("DIARIZATION_LAG_SEC", "8")
    monkeypatch.setenv("DIARIZATION_WINDOW_SEC", "20")
    monkeypatch.setenv("VAD_ML_MIN_SPEECH_RATIO", "0")
    monkeypatch.setenv("VAD_RMS_THRESHOLD", "0")

    state = ServerState()
    diarizer = RecordingDiarizer()
    state.diarizer = diarizer
    state.backend = FakeBackend()
    session = SessionState(session_id="future-context", sample_rate_hz=16_000, num_channels=1)
    session.system_buffer.first_timestamp_ms = 0
    session.system_buffer.audio_buffer.extend(b"\x00\x00" * 16_000 * 16)

    asyncio.run(
        _transcribe_stream(
            state,
            session,
            "system",
            session.system_buffer,
            min_new_audio_sec=2.0,
            window_sec=8.0,
            max_buffer_sec=120.0,
            force=True,
        )
    )

    assert session.system_buffer.last_transcribed_offset_bytes == 16_000 * 2 * 8
    assert diarizer.sample_counts == [16_000 * 16]


def test_raw_layer_arrives_before_lagged_processed_layer(monkeypatch) -> None:
    class FakeBackend:
        def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
            return [TranscriptSegment(0.0, samples.size / sample_rate, "hello", ["system"])]

    monkeypatch.setenv("WHISPER_DISABLE", "1")
    monkeypatch.setenv("PROCESSED_TRANSCRIPT_LAG_SEC", "3")
    monkeypatch.setenv("PROCESSED_MIN_NEW_AUDIO_SEC", "2")
    monkeypatch.setenv("VAD_ML_MIN_SPEECH_RATIO", "0")
    monkeypatch.setenv("VAD_RMS_THRESHOLD", "0")

    state = ServerState()
    state.backend = FakeBackend()
    state.diarizer = None
    session = SessionState(session_id="dual-layer", sample_rate_hz=16_000, num_channels=1)
    session.system_buffer.first_timestamp_ms = 100_000
    session.system_buffer.last_received_at = time.time()
    session.system_buffer.audio_buffer.extend(b"\x00\x00" * 16_000 * 2)
    socket = _RecordingSocket()
    state.session_sockets[session.session_id] = {socket}  # type: ignore[assignment]

    asyncio.run(
        _transcribe_stream(
            state,
            session,
            "system",
            session.system_buffer,
            min_new_audio_sec=2.0,
            window_sec=8.0,
            max_buffer_sec=120.0,
        )
    )

    assert [message["layer"] for message in socket.messages] == ["raw"]
    assert socket.messages[0]["segments"][0]["speaker_label"] == "System audio"

    session.system_buffer.audio_buffer.extend(b"\x00\x00" * 16_000 * 4)
    session.system_buffer.last_received_at = time.time()
    asyncio.run(
        _transcribe_stream(
            state,
            session,
            "system",
            session.system_buffer,
            min_new_audio_sec=2.0,
            window_sec=8.0,
            max_buffer_sec=120.0,
        )
    )

    assert [message["layer"] for message in socket.messages] == ["raw", "raw", "processed"]
    assert socket.messages[-1]["segments"][0]["text"] == "hello"


def test_short_idle_phrase_reaches_both_layers_without_stop(monkeypatch) -> None:
    class FakeBackend:
        def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
            return [TranscriptSegment(0.0, samples.size / sample_rate, "short phrase.", ["mic"])]

    monkeypatch.setenv("WHISPER_DISABLE", "1")
    monkeypatch.setenv("RAW_TRANSCRIPT_IDLE_FLUSH_SEC", "0.8")
    monkeypatch.setenv("PROCESSED_TRANSCRIPT_LAG_SEC", "3")
    monkeypatch.setenv("PROCESSED_MIN_NEW_AUDIO_SEC", "2")
    monkeypatch.setenv("VAD_ML_MIN_SPEECH_RATIO", "0")
    monkeypatch.setenv("VAD_RMS_THRESHOLD", "0")

    state = ServerState()
    state.backend = FakeBackend()
    state.diarizer = None
    session = SessionState(session_id="short-idle", sample_rate_hz=16_000, num_channels=1)
    session.mic_buffer.first_timestamp_ms = 100_000
    session.mic_buffer.last_received_at = time.time() - 0.9
    session.mic_buffer.audio_buffer.extend(b"\x00\x00" * 16_000)
    socket = _RecordingSocket()
    state.session_sockets[session.session_id] = {socket}  # type: ignore[assignment]

    asyncio.run(
        _transcribe_stream(
            state,
            session,
            "mic",
            session.mic_buffer,
            min_new_audio_sec=2.0,
            window_sec=8.0,
            max_buffer_sec=120.0,
        )
    )
    assert [message["layer"] for message in socket.messages] == ["raw"]

    session.mic_buffer.last_received_at = time.time() - 3.1
    asyncio.run(
        _transcribe_stream(
            state,
            session,
            "mic",
            session.mic_buffer,
            min_new_audio_sec=2.0,
            window_sec=8.0,
            max_buffer_sec=120.0,
        )
    )
    assert [message["layer"] for message in socket.messages] == ["raw", "processed"]
