from __future__ import annotations

from dataclasses import dataclass

from server.diarization import DiarizationTurn, _turns_from_pyannote_annotation
from server.main import _apply_diarization_turns
from server.session_state import SessionState, StreamBuffer
from server.speakers import SpeakerTracker


@dataclass
class _FakeDataFrame:
    rows: list[dict[str, object]]

    def iterrows(self):
        for index, row in enumerate(self.rows):
            yield index, row


class _FakeState:
    def __init__(self) -> None:
        self.trackers: dict[str, SpeakerTracker] = {}

    def get_speaker_tracker(self, session_id: str) -> SpeakerTracker:
        tracker = self.trackers.get(session_id)
        if tracker is None:
            tracker = SpeakerTracker()
            self.trackers[session_id] = tracker
        return tracker


@dataclass(frozen=True)
class _FakeSegment:
    start: float
    end: float


class _FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        for index, (start, end, label) in enumerate(self.tracks):
            yield _FakeSegment(start, end), index, label


def test_turns_from_pyannote_annotation_maps_stable_labels() -> None:
    turns = _turns_from_pyannote_annotation(
        _FakeAnnotation(
            [
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 1.1, "SPEAKER_01"),
                (1.2, 2.0, "SPEAKER_01"),
                (2.2, 3.0, "SPEAKER_00"),
            ]
        ),
        min_turn_sec=0.2,
    )

    assert [(turn.speaker_id, turn.speaker_label) for turn in turns] == [
        ("diarization:SPEAKER_00", "Speaker 1"),
        ("diarization:SPEAKER_01", "Speaker 2"),
        ("diarization:SPEAKER_00", "Speaker 1"),
    ]


def test_apply_diarization_turns_anchors_to_client_timestamps() -> None:
    state = _FakeState()
    session = SessionState(session_id="diarization-session")
    buf = StreamBuffer(stream_id="system", first_timestamp_ms=100_000)

    _apply_diarization_turns(
        state,
        session,
        "system",
        [
            DiarizationTurn(
                start_sec=0.5,
                end_sec=2.0,
                speaker_id="diarization:SPEAKER_00",
                speaker_label="Speaker 1",
                speaker_confidence=0.7,
            )
        ],
        base_offset_sec=10.0,
        buf=buf,
    )

    speaker = state.get_speaker_tracker("diarization-session").resolve(
        stream_id="system",
        start_sec=111.0,
        end_sec=111.5,
    )
    assert speaker.speaker_id == "diarization:SPEAKER_00"
    assert speaker.speaker_label == "Speaker 1"
    assert speaker.speaker_source == "diarization"
    assert speaker.speaker_confidence == 0.7
