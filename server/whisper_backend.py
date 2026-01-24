from __future__ import annotations

import io
import json
import os
import wave
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    stream_tags: list[str]


class WhisperBackend(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        """
        samples:
          - shape: (num_samples,) float32
          - range: [-1.0, 1.0]
        """


class LocalWhisperBackend:
    """
    Stub backend for wiring the pipeline end-to-end.
    Replace with faster-whisper in Phase 3.4 if desired.
    """

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        dur = float(samples.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
        return [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=max(0.0, dur),
                text="Whisper not wired yet",
                stream_tags=["mic"],
            )
        ]


def _float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode mono float32 samples in [-1, 1] to WAV (PCM S16LE).
    """
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class HttpWhisperBackend:
    """
    Sends audio to an HTTP endpoint (e.g. LM Studio or custom service).

    Contract (placeholder):
      POST {WHISPER_ENDPOINT} with body:
        - content-type: audio/wav
        - WAV bytes (mono, 16-bit PCM)

      Response JSON (example):
        {
          "segments": [
            {"start_sec": 0.0, "end_sec": 1.2, "text": "hello"}
          ]
        }
    """

    def __init__(self, endpoint: str | None = None, timeout_sec: float = 30.0) -> None:
        self.endpoint = endpoint or os.environ.get("WHISPER_ENDPOINT")
        if not self.endpoint:
            raise ValueError("WHISPER_ENDPOINT is not set")
        self.timeout_sec = timeout_sec

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        wav_bytes = _float32_to_wav_bytes(samples, sample_rate)

        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(
                self.endpoint,
                content=wav_bytes,
                headers={"content-type": "audio/wav"},
            )
            resp.raise_for_status()
            data = resp.json()

        segments: list[TranscriptSegment] = []
        for seg in data.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start_sec=float(seg.get("start_sec", 0.0)),
                    end_sec=float(seg.get("end_sec", 0.0)),
                    text=str(seg.get("text", "")),
                    stream_tags=["mic"],
                )
            )
        return segments


def create_backend() -> WhisperBackend:
    """
    Chooses backend based on environment:
      - if WHISPER_ENDPOINT is set: HttpWhisperBackend
      - else: LocalWhisperBackend (stub)
    """
    if os.environ.get("WHISPER_ENDPOINT"):
        return HttpWhisperBackend()
    return LocalWhisperBackend()

