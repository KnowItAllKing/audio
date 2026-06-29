from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from server.diarization import PyannoteDiarizationBackend


FIRST_PHRASE = "First speaker talks about payroll approvals."
SECOND_PHRASE = "Second speaker talks about vendor refunds."
THIRD_PHRASE = "First speaker returns to discuss database access."


def test_real_pyannote_diarization_two_speakers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_REAL_DIARIZATION") != "1":
        pytest.skip("set RUN_REAL_DIARIZATION=1 to run real diarization")
    if not _hf_token_present():
        pytest.skip("DIARIZATION_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN is required")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("macOS say and afconvert are required to synthesize local speech audio")

    pytest.importorskip("pyannote.audio")

    monkeypatch.setenv("DIARIZATION_MIN_SPEAKERS", "2")
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "2")
    monkeypatch.setenv("DIARIZATION_MIN_TURN_SEC", "0.1")

    first_pcm, sample_rate = synthesize_speech_pcm(tmp_path, "first", FIRST_PHRASE, voice="Samantha")
    second_pcm, _ = synthesize_speech_pcm(tmp_path, "second", SECOND_PHRASE, voice="Daniel")
    third_pcm, _ = synthesize_speech_pcm(tmp_path, "third", THIRD_PHRASE, voice="Samantha")
    audio, expected_ranges = build_turn_audio(
        sample_rate=sample_rate,
        turns=[
            ("samantha-1", first_pcm),
            ("daniel", second_pcm),
            ("samantha-2", third_pcm),
        ],
        gap_sec=0.8,
    )
    samples = pcm_to_float32(audio)

    backend = PyannoteDiarizationBackend(device=os.environ.get("DIARIZATION_DEVICE", "cpu"))
    turns = backend.diarize(samples, sample_rate)
    speaker_ids = {turn.speaker_id for turn in turns}
    speaker_by_expected_turn = {
        name: dominant_speaker_for_range(turns, start_sec=start_sec, end_sec=end_sec)
        for name, start_sec, end_sec in expected_ranges
    }

    assert len(turns) >= 2, f"expected at least 2 diarization turns, got {turns!r}"
    assert len(speaker_ids) >= 2, f"expected at least 2 speakers, got {turns!r}"
    assert speaker_by_expected_turn["samantha-1"], f"missing Samantha first turn in {turns!r}"
    assert speaker_by_expected_turn["daniel"], f"missing Daniel turn in {turns!r}"
    assert speaker_by_expected_turn["samantha-2"], f"missing Samantha return turn in {turns!r}"
    assert speaker_by_expected_turn["samantha-1"] == speaker_by_expected_turn["samantha-2"], (
        f"Samantha should map to one speaker across non-contiguous turns: {speaker_by_expected_turn!r}, turns={turns!r}"
    )
    assert speaker_by_expected_turn["samantha-1"] != speaker_by_expected_turn["daniel"], (
        f"Daniel should map to a different speaker: {speaker_by_expected_turn!r}, turns={turns!r}"
    )


def synthesize_speech_pcm(tmp_path: Path, name: str, text: str, *, voice: str) -> tuple[bytes, int]:
    aiff_path = tmp_path / f"{name}.aiff"
    wav_path = tmp_path / f"{name}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)], check=True)
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        return wf.readframes(wf.getnframes()), wf.getframerate()


def build_turn_audio(
    *,
    sample_rate: int,
    turns: list[tuple[str, bytes]],
    gap_sec: float,
) -> tuple[bytes, list[tuple[str, float, float]]]:
    out = bytearray()
    ranges: list[tuple[str, float, float]] = []
    gap = silence_pcm(sample_rate=sample_rate, duration_sec=gap_sec)
    bytes_per_sec = sample_rate * 2
    for index, (name, pcm) in enumerate(turns):
        if index > 0:
            out.extend(gap)
        start_sec = len(out) / bytes_per_sec
        out.extend(pcm)
        end_sec = len(out) / bytes_per_sec
        ranges.append((name, start_sec, end_sec))
    return bytes(out), ranges


def dominant_speaker_for_range(turns, *, start_sec: float, end_sec: float) -> str | None:
    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(turn.end_sec, end_sec) - max(turn.start_sec, start_sec))
        if overlap <= 0:
            continue
        overlaps[turn.speaker_id] = overlaps.get(turn.speaker_id, 0.0) + overlap
    if not overlaps:
        return None
    return max(overlaps, key=lambda speaker_id: overlaps[speaker_id])


def silence_pcm(*, sample_rate: int, duration_sec: float) -> bytes:
    return b"\x00" * int(sample_rate * duration_sec) * 2


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2 != 0:
        pcm = pcm[: len(pcm) - 1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _hf_token_present() -> bool:
    return any(
        os.environ.get(key)
        for key in ("DIARIZATION_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
    )
