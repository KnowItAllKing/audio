from __future__ import annotations

from server.speakers import SpeakerIdentity
from server.utterances import UtteranceAssembler, UtteranceAssemblerConfig


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
