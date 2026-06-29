from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np

from server.diarization import DiarizationTurn
from server.main import _handle_control_message
from server.speaker_memory import (
    SpeakerMemory,
    SpeakerReviewStore,
    samples_to_wav_base64,
)


SAMPLE_RATE = 16_000


def test_speaker_memory_exports_imports_and_matches(monkeypatch) -> None:
    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "1")
    memory = SpeakerMemory(threshold=0.35)

    alice_profile = memory.enroll(
        name="Alice",
        samples=voice_like_audio(fundamental_hz=180.0, formant_hz=950.0),
        sample_rate_hz=SAMPLE_RATE,
    )
    bob_profile = memory.enroll(
        name="Bob",
        samples=voice_like_audio(fundamental_hz=125.0, formant_hz=560.0),
        sample_rate_hz=SAMPLE_RATE,
    )

    imported = SpeakerMemory(threshold=0.35)
    imported.replace_profiles([profile.to_protocol() for profile in memory.profiles()])
    alice_match = imported.match(
        samples=voice_like_audio(fundamental_hz=180.0, formant_hz=950.0, phase=0.25),
        sample_rate_hz=SAMPLE_RATE,
    )
    bob_match = imported.match(
        samples=voice_like_audio(fundamental_hz=125.0, formant_hz=560.0, phase=0.5),
        sample_rate_hz=SAMPLE_RATE,
    )

    assert alice_match is not None
    assert bob_match is not None
    assert alice_match.profile.profile_id == alice_profile.profile_id
    assert bob_match.profile.profile_id == bob_profile.profile_id
    assert isinstance(alice_profile.to_protocol()["fingerprint"], list)


def test_speaker_memory_merges_repeated_name(monkeypatch) -> None:
    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "1")
    memory = SpeakerMemory(threshold=0.35)

    first = memory.enroll(name="Alice", samples=voice_like_audio(fundamental_hz=180.0), sample_rate_hz=SAMPLE_RATE)
    second = memory.enroll(name=" alice ", samples=voice_like_audio(fundamental_hz=181.0), sample_rate_hz=SAMPLE_RATE)

    assert first.profile_id == second.profile_id
    assert second.sample_count == 2
    assert len(memory.profiles()) == 1


def test_speaker_memory_import_skips_bad_profiles(monkeypatch) -> None:
    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "1")
    memory = SpeakerMemory(threshold=0.35)

    memory.replace_profiles(
        [
            {"profile_id": "bad-empty", "name": "Bad", "fingerprint": []},
            {"profile_id": "bad-nan", "name": "Bad", "fingerprint": [float("nan")]},
            {"profile_id": "ok", "name": "Ok", "fingerprint": [0.0, 1.0, 0.0], "sample_count": 2},
        ]
    )

    profiles = memory.profiles()
    assert len(profiles) == 1
    assert profiles[0].profile_id == "ok"
    assert profiles[0].sample_count == 2


def test_review_store_keeps_best_clip_per_diarized_speaker() -> None:
    store = SpeakerReviewStore(min_duration_sec=0.2, max_duration_sec=2.0)
    quiet = np.full(SAMPLE_RATE, 0.02, dtype=np.float32)
    loud = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    turn = DiarizationTurn(
        start_sec=0.0,
        end_sec=1.0,
        speaker_id="diarization:SPEAKER_00",
        speaker_label="Speaker 1",
    )

    first = store.observe(
        session_id="session",
        stream_id="system",
        turn=turn,
        absolute_start_sec=100.0,
        absolute_end_sec=101.0,
        samples=quiet,
        sample_rate_hz=SAMPLE_RATE,
        match=None,
    )
    second = store.observe(
        session_id="session",
        stream_id="system",
        turn=turn,
        absolute_start_sec=102.0,
        absolute_end_sec=103.0,
        samples=loud,
        sample_rate_hz=SAMPLE_RATE,
        match=None,
    )

    samples = store.samples_for_session("session")
    assert first is not None
    assert second is not None
    assert len(samples) == 1
    assert samples[0].start_sec == 102.0
    assert samples[0].score > first.score


def test_review_sample_protocol_contains_playable_wav() -> None:
    encoded = samples_to_wav_base64(voice_like_audio(fundamental_hz=160.0, duration_sec=0.5), SAMPLE_RATE)
    raw = base64.b64decode(encoded)
    assert raw.startswith(b"RIFF")
    assert b"WAVE" in raw[:16]


def test_speaker_memory_review_and_enroll_controls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "1")
    asyncio.run(run_speaker_memory_review_and_enroll_controls(tmp_path))


async def run_speaker_memory_review_and_enroll_controls(tmp_path) -> None:
    memory = SpeakerMemory(threshold=0.35)
    review = SpeakerReviewStore(min_duration_sec=0.2, max_duration_sec=2.0)
    turn = DiarizationTurn(
        start_sec=0.0,
        end_sec=1.0,
        speaker_id="diarization:SPEAKER_00",
        speaker_label="Speaker 1",
    )
    sample = review.observe(
        session_id="session",
        stream_id="system",
        turn=turn,
        absolute_start_sec=100.0,
        absolute_end_sec=101.0,
        samples=voice_like_audio(fundamental_hz=180.0, duration_sec=1.0),
        sample_rate_hz=SAMPLE_RATE,
        match=None,
    )
    assert sample is not None

    state = SimpleNamespace(
        lock=asyncio.Lock(),
        session_sockets={},
        speaker_memory=memory,
        speaker_review=review,
    )
    ws = FakeWebSocket()

    await _handle_control_message(
        ws,
        state,
        {
            "type": "control",
            "session_id": "session",
            "command": "speaker_memory_review",
            "payload": {},
        },
    )
    assert ws.messages[-1]["type"] == "speaker_memory_review"
    assert ws.messages[-1]["samples"][0]["sample_id"] == sample.sample_id

    await _handle_control_message(
        ws,
        state,
        {
            "type": "control",
            "session_id": "session",
            "command": "speaker_memory_profiles",
            "payload": {
                "profiles": [
                    {
                        "profile_id": "imported",
                        "name": "Imported",
                        "fingerprint": [0.0, 1.0, 0.0],
                        "sample_count": 3,
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    }
                ]
            },
        },
    )
    assert ws.messages[-1]["type"] == "speaker_memory_profiles"
    assert ws.messages[-1]["profiles"][0]["name"] == "Imported"

    await _handle_control_message(
        ws,
        state,
        {
            "type": "control",
            "session_id": "session",
            "command": "speaker_memory_enroll",
            "payload": {"sample_id": sample.sample_id, "name": "Alice"},
        },
    )
    assert ws.messages[-2]["type"] == "speaker_memory_profile"
    assert ws.messages[-2]["profile"]["name"] == "Alice"
    assert memory.profiles()[0].name == "Alice"


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


def voice_like_audio(
    *,
    fundamental_hz: float,
    formant_hz: float = 800.0,
    duration_sec: float = 2.4,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_sec), dtype=np.float32) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, 18):
        freq = fundamental_hz * harmonic
        envelope = np.exp(-np.square((freq - formant_hz) / 520.0))
        signal += (envelope / harmonic) * np.sin((2.0 * np.pi * freq * t) + phase)
    signal *= 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    signal /= max(1e-6, float(np.max(np.abs(signal))))
    return (0.35 * signal).astype(np.float32)
