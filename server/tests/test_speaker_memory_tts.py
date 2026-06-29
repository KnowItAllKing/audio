from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from server.speaker_memory import SpeakerMemory


def test_speaker_memory_matches_macos_tts_voices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_SPEAKER_MEMORY_TTS") != "1":
        pytest.skip("set RUN_SPEAKER_MEMORY_TTS=1 to run macOS TTS speaker-memory harness")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("macOS say and afconvert are required")

    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "1")
    memory = SpeakerMemory(threshold=float(os.environ.get("SPEAKER_MEMORY_TTS_THRESHOLD", "0.82")))

    samantha_enroll, sample_rate = synthesize_speech(
        tmp_path,
        "samantha-enroll",
        "First speaker discusses roadmap planning and meeting notes.",
        voice="Samantha",
    )
    daniel_enroll, _ = synthesize_speech(
        tmp_path,
        "daniel-enroll",
        "Second speaker reviews budget approvals and vendor timing.",
        voice="Daniel",
    )
    samantha_check, _ = synthesize_speech(
        tmp_path,
        "samantha-check",
        "The same first speaker returns with product launch details.",
        voice="Samantha",
    )
    daniel_check, _ = synthesize_speech(
        tmp_path,
        "daniel-check",
        "The second speaker returns with refund and invoice details.",
        voice="Daniel",
    )

    memory.enroll(name="Samantha", samples=pcm_to_float32(samantha_enroll), sample_rate_hz=sample_rate)
    memory.enroll(name="Daniel", samples=pcm_to_float32(daniel_enroll), sample_rate_hz=sample_rate)

    samantha_match = memory.match(samples=pcm_to_float32(samantha_check), sample_rate_hz=sample_rate)
    daniel_match = memory.match(samples=pcm_to_float32(daniel_check), sample_rate_hz=sample_rate)

    assert samantha_match is not None
    assert daniel_match is not None
    assert samantha_match.profile.name == "Samantha"
    assert daniel_match.profile.name == "Daniel"


def synthesize_speech(tmp_path: Path, name: str, text: str, *, voice: str) -> tuple[bytes, int]:
    aiff_path = tmp_path / f"{name}.aiff"
    wav_path = tmp_path / f"{name}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)], check=True)
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        return wf.readframes(wf.getnframes()), wf.getframerate()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2 != 0:
        pcm = pcm[: len(pcm) - 1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
