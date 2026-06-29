from __future__ import annotations

import os
from dataclasses import dataclass

from server import diarization as diarization_module
from server.diarization import DiarizationTurn, _turns_from_pyannote_annotation, create_diarization_backend
from server.main import _load_dotenv
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


def test_load_dotenv_sets_missing_values_only(tmp_path, monkeypatch) -> None:
    loaded_keys = ("HF_TOKEN", "DIARIZATION_DEVICE", "WHISPER_MODEL")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# local secrets",
                "HF_TOKEN=from-file",
                "DIARIZATION_DEVICE='cpu'",
                'WHISPER_MODEL="tiny"',
                "EXISTING=from-file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("DIARIZATION_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.setenv("EXISTING", "from-env")

    try:
        _load_dotenv(dotenv)

        assert os.environ["HF_TOKEN"] == "from-file"
        assert os.environ["DIARIZATION_DEVICE"] == "cpu"
        assert os.environ["WHISPER_MODEL"] == "tiny"
        assert os.environ["EXISTING"] == "from-env"
    finally:
        for key in loaded_keys:
            os.environ.pop(key, None)


def test_auto_diarization_is_off_without_token(monkeypatch) -> None:
    monkeypatch.delenv("DIARIZATION_BACKEND", raising=False)
    monkeypatch.delenv("DIARIZATION_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    assert create_diarization_backend() is None


def test_auto_diarization_uses_pyannote_when_token_exists(monkeypatch) -> None:
    class FakeBackend:
        model_name = "fake-model"
        device = "cpu"

    fake_backend = FakeBackend()
    monkeypatch.delenv("DIARIZATION_BACKEND", raising=False)
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(diarization_module, "PyannoteDiarizationBackend", lambda: fake_backend)

    assert create_diarization_backend() is fake_backend


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
