from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx
import numpy as np

logger = logging.getLogger("server.whisper")


@dataclass(frozen=True)
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


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Minimal resampler (linear interpolation) to avoid extra deps.
    Good enough for MVP; can be replaced with a higher-quality resampler later.
    """
    if src_rate == dst_rate:
        return samples
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)

    ratio = float(dst_rate) / float(src_rate)
    out_len = max(1, int(round(samples.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32, copy=False)


class OpenAIWhisperBackend:
    """
    Real local Whisper backend using OpenAI's reference whisper library (openai-whisper).

    Notes:
    - The server targets Python 3.11 so Torch/numba can stay on patched wheels.
    - Expects float32 mono samples in [-1, 1]. We resample to 16kHz if needed.
    """

    def __init__(self) -> None:
        import whisper  # type: ignore

        self._whisper = whisper
        self.model_name = os.environ.get("WHISPER_MODEL", "turbo")
        # whisper uses torch under the hood; device is typically "cpu" or "cuda".
        self.device = os.environ.get("WHISPER_DEVICE", "cpu")
        self.language = os.environ.get("WHISPER_LANGUAGE", "en")

        self._model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        target_rate = 16000
        if sample_rate != target_rate:
            samples = _resample_linear(samples, sample_rate, target_rate)

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        result = self._model.transcribe(
            samples,
            language=self.language,
            fp16=False if self.device == "cpu" else None,
        )

        out: list[TranscriptSegment] = []
        for seg in result.get("segments", []) or []:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            out.append(
                TranscriptSegment(
                    start_sec=float(seg.get("start", 0.0)),
                    end_sec=float(seg.get("end", 0.0)),
                    text=text,
                    stream_tags=["mic"],
                )
            )
        return out


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

    def __init__(self, endpoint: Optional[str] = None, timeout_sec: float = 30.0) -> None:
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
      - else: OpenAIWhisperBackend (local)

    To force stub mode:
      - set WHISPER_DISABLE=1, or
      - set WHISPER_MODEL=stub
    """
    if os.environ.get("WHISPER_ENDPOINT"):
        return HttpWhisperBackend()

    if os.environ.get("WHISPER_DISABLE") == "1" or os.environ.get("WHISPER_MODEL") == "stub":
        logger.info("whisper backend: stub (disabled)")
        return LocalWhisperBackend()

    try:
        backend = OpenAIWhisperBackend()
        logger.info("whisper backend: openai-whisper model=%s device=%s", backend.model_name, backend.device)
        return backend
    except Exception as e:
        logger.exception("Failed to init openai-whisper backend (falling back to stub): %s", e)
        return LocalWhisperBackend()
