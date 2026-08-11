from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets

from server.main import ServerState, _handle_client, _transcription_loop


PHRASE = "the quick brown fox jumps over the lazy dog"
EXPECTED_WORDS = set(PHRASE.split())


@dataclass(frozen=True)
class HardSpeechCase:
    name: str
    pcm: bytes
    sample_rate: int
    expected_terms: tuple[tuple[str, ...], ...]
    min_final_segments: int = 1


@dataclass(frozen=True)
class HardSpeechCaseResult:
    name: str
    transcript: str
    hit_count: int
    term_count: int
    final_segment_count: int
    required_final_segment_count: int

    @property
    def score(self) -> float:
        return self.hit_count / float(max(1, self.term_count))


@pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets")
def test_end_to_end_real_whisper_local_speech(tmp_path: Path) -> None:
    if os.environ.get("RUN_REAL_WHISPER_E2E") != "1":
        pytest.skip("set RUN_REAL_WHISPER_E2E=1 to run the real Whisper end-to-end test")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("macOS say and afconvert are required to synthesize local speech audio")

    asyncio.run(run_end_to_end_real_whisper_local_speech(tmp_path))


@pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets")
def test_hard_end_to_end_real_whisper_cases(tmp_path: Path) -> None:
    if os.environ.get("RUN_REAL_WHISPER_HARD_E2E") != "1":
        pytest.skip("set RUN_REAL_WHISPER_HARD_E2E=1 to run hard real Whisper end-to-end tests")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("macOS say and afconvert are required to synthesize local speech audio")

    asyncio.run(run_hard_end_to_end_real_whisper_cases(tmp_path))


async def run_end_to_end_real_whisper_local_speech(tmp_path: Path) -> None:
    previous_env = _set_test_env(model=_e2e_model())
    wav_path = synthesize_speech_wav(tmp_path, PHRASE)
    pcm, sample_rate = read_wav_pcm_s16le(wav_path)

    state = ServerState()
    transcriber = asyncio.create_task(_transcription_loop(state))

    try:
        async with websockets.serve(lambda ws: _handle_client(ws, state), "127.0.0.1", 0) as server:
            assert server.sockets
            port = int(server.sockets[0].getsockname()[1])

            final_segments = await transcribe_pcm_via_websocket(
                port=port,
                session_id="real-whisper-e2e",
                pcm=pcm,
                sample_rate=sample_rate,
            )

        assert final_segments, "real Whisper E2E produced no final transcript segments"
        transcript = normalize_text(" ".join(segment["text"] for segment in final_segments))
        missing = EXPECTED_WORDS - set(transcript.split())
        assert not missing, f"missing expected words {sorted(missing)} from transcript {transcript!r}"

        assert any(segment["speaker_label"] == "Real Whisper Speaker" for segment in final_segments)
        assert any(segment["stream_tags"] == ["mic"] for segment in final_segments)
    finally:
        transcriber.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await transcriber
        _restore_env(previous_env)


async def run_hard_end_to_end_real_whisper_cases(tmp_path: Path) -> None:
    model = _e2e_model()
    previous_env = _set_test_env(model=model)
    cases = build_hard_speech_cases(tmp_path)

    state = ServerState()
    transcriber = asyncio.create_task(_transcription_loop(state))

    try:
        async with websockets.serve(lambda ws: _handle_client(ws, state), "127.0.0.1", 0) as server:
            assert server.sockets
            port = int(server.sockets[0].getsockname()[1])
            results: list[HardSpeechCaseResult] = []

            for index, case in enumerate(cases):
                final_segments = await transcribe_pcm_via_websocket(
                    port=port,
                    session_id=f"real-whisper-hard-e2e-{index}",
                    pcm=case.pcm,
                    sample_rate=case.sample_rate,
                    idle_grace_sec=2.5,
                )
                transcript = normalize_text(" ".join(segment["text"] for segment in final_segments))
                hit_count = sum(1 for term in case.expected_terms if transcript_contains_any(transcript, term))
                results.append(
                    HardSpeechCaseResult(
                        name=case.name,
                        transcript=transcript,
                        hit_count=hit_count,
                        term_count=len(case.expected_terms),
                        final_segment_count=len(final_segments),
                        required_final_segment_count=case.min_final_segments,
                    )
                )

        assert_hard_case_results(model=model, results=results)
    finally:
        transcriber.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await transcriber
        _restore_env(previous_env)


def synthesize_speech_wav(tmp_path: Path, text: str) -> Path:
    aiff_path = tmp_path / "speech.aiff"
    wav_path = tmp_path / "speech.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff_path), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)], check=True)
    return wav_path


def synthesize_speech_pcm(tmp_path: Path, name: str, text: str) -> tuple[bytes, int]:
    aiff_path = tmp_path / f"{name}.aiff"
    wav_path = tmp_path / f"{name}.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff_path), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)], check=True)
    return read_wav_pcm_s16le(wav_path)


def build_hard_speech_cases(tmp_path: Path) -> list[HardSpeechCase]:
    proper_pcm, sample_rate = synthesize_speech_pcm(
        tmp_path,
        "proper-nouns",
        "Kubernetes, PostgreSQL, WebRTC, RTMS, and the Zoom SDK are on the agenda.",
    )
    numbers_pcm, _ = synthesize_speech_pcm(
        tmp_path,
        "numbers-acronyms",
        "Ship PR 1042 after QA says LGTM, then file the SOC 2 note.",
    )
    first_pcm, _ = synthesize_speech_pcm(tmp_path, "pause-first", "First item is payroll approvals.")
    second_pcm, _ = synthesize_speech_pcm(tmp_path, "pause-second", "Second item is vendor refunds.")
    pause_pcm = first_pcm + silence_pcm(sample_rate=sample_rate, duration_sec=1.2) + second_pcm

    primary_pcm, _ = synthesize_speech_pcm(
        tmp_path,
        "noisy-primary",
        "The action item is to rotate the database password today.",
    )
    background_pcm, _ = synthesize_speech_pcm(
        tmp_path,
        "noisy-background",
        "Budget calendar status sync update next week.",
    )
    noisy_pcm = add_white_noise_pcm(
        mix_pcm(primary_pcm, background_pcm, primary_gain=1.0, secondary_gain=0.16),
        amplitude=0.006,
        seed=17,
    )

    return [
        HardSpeechCase(
            name="proper-nouns-and-acronyms",
            pcm=proper_pcm,
            sample_rate=sample_rate,
            expected_terms=(
                ("kubernetes", "cube er net ease"),
                ("postgresql", "postgres", "postgre sql", "post gres q l"),
                ("webrtc", "web rtc", "web r t c"),
                ("rtms", "r t m s"),
                ("zoom",),
                ("sdk", "s d k"),
            ),
        ),
        HardSpeechCase(
            name="numbers-and-meeting-acronyms",
            pcm=numbers_pcm,
            sample_rate=sample_rate,
            expected_terms=(
                ("pr", "p r"),
                ("1042", "one thousand forty two", "ten forty two"),
                ("qa", "q a"),
                ("lgtm", "l g t m"),
                ("soc 2", "sock two", "soc two", "s o c 2"),
            ),
        ),
        HardSpeechCase(
            name="pause-creates-phrase-boundary",
            pcm=pause_pcm,
            sample_rate=sample_rate,
            expected_terms=(
                ("first item",),
                ("payroll approvals",),
                ("second item",),
                ("vendor refunds",),
            ),
            min_final_segments=2,
        ),
        HardSpeechCase(
            name="speech-with-background-chatter-and-noise",
            pcm=noisy_pcm,
            sample_rate=sample_rate,
            expected_terms=(
                ("action item",),
                ("rotate",),
                ("database",),
                ("password",),
                ("today",),
            ),
        ),
    ]


def read_wav_pcm_s16le(wav_path: Path) -> tuple[bytes, int]:
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        return wf.readframes(wf.getnframes()), wf.getframerate()


def silence_pcm(*, sample_rate: int, duration_sec: float) -> bytes:
    sample_count = int(sample_rate * duration_sec)
    return b"\x00" * sample_count * 2


def mix_pcm(primary_pcm: bytes, secondary_pcm: bytes, *, primary_gain: float, secondary_gain: float) -> bytes:
    primary = pcm_to_float32(primary_pcm)
    secondary = pcm_to_float32(secondary_pcm)
    if secondary.size < primary.size:
        secondary = np.pad(secondary, (0, primary.size - secondary.size))
    else:
        secondary = secondary[: primary.size]
    return float32_to_pcm(primary * primary_gain + secondary * secondary_gain)


def add_white_noise_pcm(pcm: bytes, *, amplitude: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    samples = pcm_to_float32(pcm)
    noise = rng.normal(0.0, amplitude, size=samples.size).astype(np.float32)
    return float32_to_pcm(samples + noise)


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2 != 0:
        pcm = pcm[: len(pcm) - 1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


async def transcribe_pcm_via_websocket(
    *,
    port: int,
    session_id: str,
    pcm: bytes,
    sample_rate: int,
    idle_grace_sec: float = 2.0,
) -> list[dict[str, Any]]:
    async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=16 * 1024 * 1024) as ws:
        await send_metadata(ws, session_id=session_id)
        await stream_pcm(ws, session_id=session_id, pcm=pcm, sample_rate=sample_rate)
        return await collect_final_segments(ws, min_count=1, idle_grace_sec=idle_grace_sec)


async def send_metadata(ws: Any, *, session_id: str) -> None:
    await ws.send(
        json.dumps(
            {
                "type": "control",
                "session_id": session_id,
                "command": "metadata",
                "payload": {
                    "stream_speakers": {
                        "mic": {
                            "speaker_id": "e2e:mic",
                            "speaker_label": "Real Whisper Speaker",
                            "speaker_source": "manual",
                            "speaker_confidence": 0.9,
                        }
                    }
                },
            }
        )
    )


async def stream_pcm(ws: Any, *, session_id: str, pcm: bytes, sample_rate: int) -> None:
    bytes_per_ms = sample_rate * 2 / 1000.0
    chunk_ms = 200
    chunk_bytes = int(bytes_per_ms * chunk_ms)
    timestamp_ms = 1_737_384_349_000
    seq = 0

    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        await ws.send(
            json.dumps(
                {
                    "type": "audio_chunk",
                    "session_id": session_id,
                    "stream_id": "mic",
                    "seq": seq,
                    "timestamp_ms": timestamp_ms + int(offset / bytes_per_ms),
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": sample_rate,
                        "num_channels": 1,
                    },
                    "audio_base64": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        seq += 1
        await asyncio.sleep(0.01)

    await ws.send(
        json.dumps(
            {
                "type": "control",
                "session_id": session_id,
                "command": "stop",
                "payload": {},
            }
        )
    )


async def collect_final_segments(ws: Any, *, min_count: int, idle_grace_sec: float) -> list[dict[str, Any]]:
    segments: dict[str, dict[str, Any]] = {}
    deadline = asyncio.get_running_loop().time() + 30.0
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        timeout = min(remaining, idle_grace_sec) if len(segments) >= min_count else remaining
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            if len(segments) >= min_count:
                break
            raise
        message = json.loads(raw)
        if message.get("type") != "transcript_update":
            continue
        if message.get("layer", "processed") != "processed":
            continue
        for segment in message.get("segments", []):
            if segment.get("is_final"):
                segments[segment["id"]] = segment
    return list(segments.values())


def normalize_text(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text]
    return " ".join("".join(chars).split())


def transcript_contains_any(transcript: str, alternatives: tuple[str, ...]) -> bool:
    compact_transcript = transcript.replace(" ", "")
    for alternative in alternatives:
        normalized = normalize_text(alternative)
        if normalized and normalized in transcript:
            return True
        compact = normalized.replace(" ", "")
        if compact and compact in compact_transcript:
            return True
    return False


def assert_hard_case_results(*, model: str, results: list[HardSpeechCaseResult]) -> None:
    assert results, "hard Whisper E2E produced no case results"

    missing_boundaries = [
        result
        for result in results
        if result.final_segment_count < result.required_final_segment_count
    ]
    total_hits = sum(result.hit_count for result in results)
    total_terms = sum(result.term_count for result in results)
    overall_score = total_hits / float(max(1, total_terms))
    min_score = hard_min_score(model)
    min_case_score = hard_min_case_score()
    weak_cases = [result for result in results if result.score < min_case_score]

    failure_lines = [
        f"model={model}",
        f"overall_score={overall_score:.3f}",
        f"required_score={min_score:.3f}",
        f"required_case_score={min_case_score:.3f}",
    ]
    for result in results:
        failure_lines.append(
            "case={name} score={score:.3f} finals={finals}/{required} transcript={transcript!r}".format(
                name=result.name,
                score=result.score,
                finals=result.final_segment_count,
                required=result.required_final_segment_count,
                transcript=result.transcript,
            )
        )

    if missing_boundaries or weak_cases or overall_score < min_score:
        report = "\n".join(failure_lines)
        if expected_hard_failure_allowed(model):
            pytest.xfail(report)
        assert not missing_boundaries, report
        assert not weak_cases, report
        assert overall_score >= min_score, report

    assert not missing_boundaries, "\n".join(failure_lines)
    assert not weak_cases, "\n".join(failure_lines)
    assert overall_score >= min_score, "\n".join(failure_lines)


def hard_min_score(model: str) -> float:
    override = os.environ.get("WHISPER_E2E_HARD_MIN_SCORE")
    if override:
        return float(override)

    normalized = model.lower()
    if normalized in ("tiny", "tiny.en"):
        return 0.80
    if normalized in ("base", "base.en"):
        return 0.84
    if normalized in ("small", "small.en"):
        return 0.88
    if normalized in ("medium", "medium.en"):
        return 0.90
    return 0.92


def hard_min_case_score() -> float:
    override = os.environ.get("WHISPER_E2E_HARD_CASE_MIN_SCORE")
    if override:
        return float(override)
    return 0.70


def expected_hard_failure_allowed(model: str) -> bool:
    if os.environ.get("WHISPER_E2E_HARD_STRICT") == "1":
        return False
    return model.lower() in ("tiny", "tiny.en")


def _e2e_model() -> str:
    return os.environ.get("WHISPER_E2E_MODEL") or os.environ.get("WHISPER_MODEL") or "tiny"


def _set_test_env(*, model: str) -> dict[str, str | None]:
    updates = {
        "WHISPER_MODEL": model,
        "WHISPER_DEVICE": "cpu",
        "WHISPER_LANGUAGE": "en",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "0",
        "TRANSCRIBE_INTERVAL_SEC": "0.1",
        "MIN_NEW_AUDIO_SEC": "0.4",
        "WINDOW_SEC": "8",
        "MAX_BUFFER_SEC": "20",
        "UTTERANCE_PAUSE_SEC": "0.4",
        "UTTERANCE_MAX_SEC": "10",
        "VAD_ML_MIN_SPEECH_RATIO": "0.05",
        "VAD_RMS_THRESHOLD": "0.002",
    }
    previous = {key: os.environ.get(key) for key in updates}
    previous["WHISPER_DISABLE"] = os.environ.get("WHISPER_DISABLE")
    os.environ.update(updates)
    os.environ.pop("WHISPER_DISABLE", None)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
