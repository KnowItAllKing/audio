from __future__ import annotations

from server.speakers import SpeakerTracker


def test_mic_uses_local_label_by_default() -> None:
    tracker = SpeakerTracker(local_label="Alex")
    speaker = tracker.resolve(stream_id="mic", start_sec=10.0, end_sec=11.0)

    assert speaker.speaker_id == "local:mic"
    assert speaker.speaker_label == "Alex"
    assert speaker.speaker_source == "stream"


def test_metadata_overrides_stream_labels() -> None:
    tracker = SpeakerTracker()
    tracker.apply_metadata(
        {
            "stream_speakers": {
                "mic": {"speaker_id": "me", "speaker_label": "Me"},
                "system": "Room audio",
            }
        }
    )

    mic = tracker.resolve(stream_id="mic", start_sec=0.0, end_sec=1.0)
    system = tracker.resolve(stream_id="system", start_sec=0.0, end_sec=1.0)

    assert mic.speaker_id == "me"
    assert mic.speaker_label == "Me"
    assert mic.speaker_source == "manual"
    assert system.speaker_label == "Room audio"


def test_zoom_activity_labels_system_audio() -> None:
    tracker = SpeakerTracker(active_speaker_ttl_sec=30.0)
    tracker.apply_metadata({"participants": [{"id": "p1", "name": "Taylor"}]})
    tracker.apply_activity(
        {
            "participant_id": "p1",
            "stream_id": "system",
            "start_sec": 100.0,
            "end_sec": 105.0,
        }
    )

    speaker = tracker.resolve(stream_id="system", start_sec=101.0, end_sec=102.0)

    assert speaker.speaker_id == "zoom:p1"
    assert speaker.speaker_label == "Taylor"
    assert speaker.speaker_source == "zoom"
    assert speaker.speaker_confidence == 0.95


def test_manual_activity_labels_system_audio_without_zoom_app() -> None:
    tracker = SpeakerTracker(active_speaker_ttl_sec=30.0)

    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "manual:taylor",
            "speaker_label": "Taylor",
            "speaker_source": "manual",
            "speaker_confidence": 0.8,
            "start_timestamp_ms": 100_000,
        }
    )

    speaker = tracker.resolve(stream_id="system", start_sec=101.0, end_sec=102.0)
    assert speaker.speaker_id == "manual:taylor"
    assert speaker.speaker_label == "Taylor"
    assert speaker.speaker_source == "manual"
    assert speaker.speaker_confidence == 0.8


def test_manual_activity_persists_beyond_rtms_ttl() -> None:
    tracker = SpeakerTracker(active_speaker_ttl_sec=5.0)

    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "manual:taylor",
            "speaker_label": "Taylor",
            "speaker_source": "manual",
            "start_sec": 100.0,
        }
    )

    speaker = tracker.resolve(stream_id="system", start_sec=140.0, end_sec=141.0)
    assert speaker.speaker_label == "Taylor"
    assert speaker.speaker_source == "manual"


def test_manual_activity_takes_priority_over_diarization() -> None:
    tracker = SpeakerTracker(active_speaker_ttl_sec=5.0)
    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "manual:taylor",
            "speaker_label": "Taylor",
            "speaker_source": "manual",
            "start_sec": 100.0,
        }
    )
    tracker.apply_activity(
        {
            "stream_id": "system",
            "speaker_id": "diarization:SPEAKER_00",
            "speaker_label": "Speaker 1",
            "speaker_source": "diarization",
            "start_sec": 120.0,
            "end_sec": 125.0,
        }
    )

    speaker = tracker.resolve(stream_id="system", start_sec=121.0, end_sec=122.0)
    assert speaker.speaker_id == "manual:taylor"
    assert speaker.speaker_label == "Taylor"
    assert speaker.speaker_source == "manual"


def test_stale_open_zoom_activity_falls_back_to_system_label() -> None:
    tracker = SpeakerTracker(system_label="Computer audio", active_speaker_ttl_sec=5.0)
    tracker.apply_activity({"participant_id": "p1", "participant_name": "Taylor", "start_sec": 10.0})

    speaker = tracker.resolve(stream_id="system", start_sec=30.0, end_sec=31.0)

    assert speaker.speaker_id == "system:unknown"
    assert speaker.speaker_label == "Computer audio"


def test_activity_boundary_tolerance_absorbs_small_timestamp_drift() -> None:
    tracker = SpeakerTracker(activity_boundary_tolerance_sec=0.35)
    tracker.apply_activity(
        {
            "participant_id": "p1",
            "participant_name": "Taylor",
            "stream_id": "system",
            "speaker_source": "diarization",
            "start_sec": 10.0,
            "end_sec": 12.0,
        }
    )

    before = tracker.resolve(stream_id="system", start_sec=9.5, end_sec=9.9)
    after = tracker.resolve(stream_id="system", start_sec=12.1, end_sec=12.4)

    assert before.speaker_id == "diarization:p1"
    assert after.speaker_id == "diarization:p1"
