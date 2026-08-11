from __future__ import annotations

import asyncio
import json

import numpy as np

from server.main import ServerState, _assemble_transcript_segments, _handle_control_message
from server.session_state import SessionState, StreamBuffer
from server.speakers import SpeakerIdentity
from server.utterances import UtteranceAssembler, UtteranceAssemblerConfig
from server.whisper_backend import TranscriptSegment, TranscriptWord


SPEAKER_A = SpeakerIdentity("speaker:a", "Alice", "manual", 0.9)
SPEAKER_B = SpeakerIdentity("speaker:b", "Blair", "manual", 0.9)


def make_assembler() -> UtteranceAssembler:
    return UtteranceAssembler(
        session_id="session-1234",
        config=UtteranceAssemblerConfig(pause_sec=1.0, max_duration_sec=5.0, emit_partials=True),
    )


def test_emits_partial_until_sentence_is_complete() -> None:
    assembler = make_assembler()

    first = assembler.push(stream_id="mic", start_sec=0.0, end_sec=0.5, text="hello", speaker=SPEAKER_A)
    second = assembler.push(stream_id="mic", start_sec=0.5, end_sec=1.0, text="world.", speaker=SPEAKER_A)

    assert len(first) == 1
    assert first[0].is_final is False
    assert first[0].text == "hello"
    assert len(second) == 1
    assert second[0].is_final is True
    assert second[0].text == "hello world."
    assert second[0].final_reason == "punctuation"
    assert second[0].speaker_label == "Alice"


def test_splits_complete_prefix_and_keeps_remainder_live() -> None:
    assembler = make_assembler()

    out = assembler.push(stream_id="mic", start_sec=0.0, end_sec=2.0, text="First sentence. second", speaker=SPEAKER_A)

    assert [item.text for item in out] == ["First sentence.", "second"]
    assert out[0].is_final is True
    assert out[1].is_final is False
    assert out[0].id != out[1].id


def test_speaker_change_finalizes_open_phrase() -> None:
    assembler = make_assembler()

    assembler.push(stream_id="mic", start_sec=0.0, end_sec=0.5, text="unfinished thought", speaker=SPEAKER_A)
    out = assembler.push(stream_id="mic", start_sec=0.6, end_sec=1.0, text="reply", speaker=SPEAKER_B)

    assert [item.text for item in out] == ["unfinished thought", "reply"]
    assert out[0].is_final is True
    assert out[0].final_reason == "speaker_change"
    assert out[0].speaker_label == "Alice"
    assert out[1].is_final is False
    assert out[1].speaker_label == "Blair"


def test_pause_finalizes_open_phrase() -> None:
    assembler = make_assembler()

    assembler.push(stream_id="mic", start_sec=0.0, end_sec=0.5, text="one phrase", speaker=SPEAKER_A)
    out = assembler.push(stream_id="mic", start_sec=2.0, end_sec=2.5, text="next phrase", speaker=SPEAKER_A)

    assert [item.text for item in out] == ["one phrase", "next phrase"]
    assert out[0].is_final is True
    assert out[0].final_reason == "pause"
    assert out[1].is_final is False


def test_stale_flush_finalizes_after_pause_without_new_audio() -> None:
    assembler = make_assembler()

    assembler.push(stream_id="mic", start_sec=100.0, end_sec=101.0, text="one phrase", speaker=SPEAKER_A)
    early = assembler.flush_if_stale(101.5)
    late = assembler.flush_if_stale(102.1)

    assert early == []
    assert len(late) == 1
    assert late[0].is_final is True
    assert late[0].final_reason == "pause"
    assert late[0].text == "one phrase"


def test_max_duration_finalizes_phrase_without_punctuation() -> None:
    assembler = make_assembler()

    out = assembler.push(stream_id="mic", start_sec=0.0, end_sec=6.0, text="long phrase without punctuation", speaker=SPEAKER_A)

    assert len(out) == 1
    assert out[0].is_final is True
    assert out[0].final_reason == "max_duration"


def test_trailing_ellipsis_stays_open_for_the_next_fragment() -> None:
    assembler = make_assembler()

    first = assembler.push(
        stream_id="mic",
        start_sec=0.0,
        end_sec=0.5,
        text="I'm David and I'm...",
        speaker=SPEAKER_A,
    )
    second = assembler.push(
        stream_id="mic",
        start_sec=0.5,
        end_sec=1.0,
        text="supposed to be a designer.",
        speaker=SPEAKER_A,
    )

    assert len(first) == 1
    assert first[0].is_final is False
    assert len(second) == 1
    assert second[0].is_final is True
    assert second[0].text == "I'm David and I'm... supposed to be a designer."


def test_server_keeps_open_fragments_independent_between_audio_streams(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_DISABLE", "1")
    state = ServerState()
    mic = state.get_utterance_assembler("session", "mic")
    system = state.get_utterance_assembler("session", "system")

    mic.push(stream_id="mic", start_sec=0.0, end_sec=0.4, text="hello", speaker=SPEAKER_A)
    system.push(stream_id="system", start_sec=0.1, end_sec=0.5, text="reply", speaker=SPEAKER_B)
    out = mic.push(stream_id="mic", start_sec=0.4, end_sec=0.8, text="world.", speaker=SPEAKER_A)

    assert len(out) == 1
    assert out[0].text == "hello world."
    assert out[0].final_reason == "punctuation"


def test_word_timestamps_split_one_asr_segment_at_diarized_speaker_change(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_DISABLE", "1")
    state = ServerState()
    session = SessionState(session_id="word-speakers")
    buf = StreamBuffer(stream_id="system", first_timestamp_ms=100_000)
    tracker = state.get_speaker_tracker(session.session_id)
    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "diarization:first",
            "speaker_label": "Speaker 1",
            "speaker_source": "diarization",
            "start_sec": 100.0,
            "end_sec": 101.0,
        }
    )
    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "diarization:second",
            "speaker_label": "Speaker 2",
            "speaker_source": "diarization",
            "start_sec": 101.0,
            "end_sec": 102.0,
        }
    )
    raw = TranscriptSegment(
        start_sec=0.0,
        end_sec=2.0,
        text="Hello there. General Kenobi.",
        stream_tags=["system"],
        words=(
            TranscriptWord(0.0, 0.4, "Hello"),
            TranscriptWord(0.4, 0.9, "there."),
            TranscriptWord(1.1, 1.5, "General"),
            TranscriptWord(1.5, 1.9, "Kenobi."),
        ),
    )

    out = _assemble_transcript_segments(state, session, "system", [raw], 0.0, buf)

    assert [(item["speaker_label"], item["text"]) for item in out] == [
        ("Speaker 1", "Hello there."),
        ("Speaker 2", "General Kenobi."),
    ]


class _TailBackend:
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        duration = float(samples.size) / float(sample_rate)
        return [TranscriptSegment(0.0, duration, "last words.", ["system"])]


class _RecordingSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


def test_stop_drains_subthreshold_audio_before_acknowledging(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_DISABLE", "1")
    monkeypatch.setenv("MIN_NEW_AUDIO_SEC", "2")
    monkeypatch.setenv("WINDOW_SEC", "8")
    monkeypatch.setenv("VAD_ML_MIN_SPEECH_RATIO", "0")
    monkeypatch.setenv("VAD_RMS_THRESHOLD", "0")

    async def run() -> list[dict]:
        state = ServerState()
        state.backend = _TailBackend()
        session = SessionState(session_id="stop-tail", sample_rate_hz=16_000, num_channels=1)
        session.system_buffer.first_timestamp_ms = 100_000
        session.system_buffer.audio_buffer.extend(b"\x00\x00" * 3_200)  # 0.2 seconds, below normal threshold.
        state.sessions[session.session_id] = session
        socket = _RecordingSocket()
        await _handle_control_message(
            socket,  # type: ignore[arg-type]
            state,
            {"type": "control", "session_id": session.session_id, "command": "stop", "payload": {}},
        )
        return socket.messages

    messages = asyncio.run(run())
    transcript_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("type") == "transcript_update"
        and message.get("layer") == "processed"
        and any(segment.get("text") == "last words." for segment in message.get("segments", []))
    )
    raw_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("type") == "transcript_update"
        and message.get("layer") == "raw"
        and any(segment.get("text") == "last words." for segment in message.get("segments", []))
    )
    stopped_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("type") == "control" and message.get("command") == "stopped"
    )
    assert raw_index < transcript_index < stopped_index
