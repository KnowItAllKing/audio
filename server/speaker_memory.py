from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import os
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .diarization import DiarizationTurn, default_sherpa_model_paths
from .protocol_types import StreamId


logger = logging.getLogger("server.speaker_memory")

MAX_PROFILE_EXEMPLARS = 8
EXEMPLAR_MERGE_SIMILARITY = 0.90


@dataclass(frozen=True)
class SpeakerProfile:
    profile_id: str
    name: str
    embedding: np.ndarray
    sample_count: int
    created_at: float
    updated_at: float
    # One voice can legitimately have several distinct fingerprints (different
    # mics, rooms, days). Matching uses the best exemplar, not an average.
    exemplars: tuple[np.ndarray, ...] = ()
    kind: str = "named"

    def bank(self) -> tuple[np.ndarray, ...]:
        return self.exemplars if self.exemplars else (self.embedding,)

    def to_protocol(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "fingerprint": [float(value) for value in self.embedding],
            "fingerprints": [[float(value) for value in exemplar] for exemplar in self.bank()],
            "kind": self.kind,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SpeakerMatch:
    profile: SpeakerProfile
    confidence: float


@dataclass
class ReviewSample:
    sample_id: str
    session_id: str
    stream_id: StreamId
    source_speaker_id: str
    source_speaker_label: str
    start_sec: float
    end_sec: float
    sample_rate_hz: int
    samples: np.ndarray
    score: float
    matched_profile_id: Optional[str] = None
    matched_name: Optional[str] = None
    match_confidence: Optional[float] = None

    def to_protocol(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "speaker_id": self.source_speaker_id,
            "speaker_label": self.source_speaker_label,
            "stream_id": self.stream_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": max(0.0, self.end_sec - self.start_sec),
            "sample_rate_hz": self.sample_rate_hz,
            "audio_wav_base64": samples_to_wav_base64(self.samples, self.sample_rate_hz),
            "matched_profile_id": self.matched_profile_id,
            "matched_name": self.matched_name,
            "match_confidence": self.match_confidence,
        }


class VoiceFingerprint:
    kind = "spectral"

    def __init__(self, *, target_rate_hz: int = 16_000) -> None:
        self.target_rate_hz = target_rate_hz

    def embed(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        audio = prepare_audio(samples=samples, sample_rate_hz=sample_rate_hz, target_rate_hz=self.target_rate_hz)
        if audio.size < int(self.target_rate_hz * 0.25):
            raise ValueError("speaker sample is too short")

        frames = frame_audio(audio, frame_size=int(self.target_rate_hz * 0.032), hop_size=int(self.target_rate_hz * 0.016))
        if frames.shape[0] == 0:
            raise ValueError("speaker sample has no usable frames")

        window = np.hanning(frames.shape[1]).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
        power = np.square(spectrum)
        freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / self.target_rate_hz).astype(np.float32)

        mel_filters = mel_filterbank(freqs=freqs, sample_rate_hz=self.target_rate_hz, band_count=30)
        mel_energy = np.maximum(power @ mel_filters.T, 1e-8)
        log_mel = np.log(mel_energy)

        frame_energy = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-9)
        centroid = np.sum(power * freqs[None, :], axis=1) / np.maximum(np.sum(power, axis=1), 1e-9)
        bandwidth = np.sqrt(
            np.sum(power * np.square(freqs[None, :] - centroid[:, None]), axis=1)
            / np.maximum(np.sum(power, axis=1), 1e-9)
        )
        zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        pitch = np.array([estimate_pitch_hz(frame, self.target_rate_hz) for frame in frames], dtype=np.float32)
        pitch = pitch[pitch > 0]

        pieces = [
            np.mean(log_mel, axis=0),
            np.std(log_mel, axis=0),
            quantiles(centroid / (self.target_rate_hz / 2.0)),
            quantiles(bandwidth / (self.target_rate_hz / 2.0)),
            quantiles(frame_energy),
            quantiles(zcr),
            quantiles(pitch / 500.0) if pitch.size else np.zeros(5, dtype=np.float32),
        ]
        embedding = np.concatenate([piece.astype(np.float32, copy=False) for piece in pieces])
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            raise ValueError("speaker sample has no voice fingerprint")
        return embedding / norm


class SherpaVoiceEmbedder:
    """Real speaker-verification embeddings (3D-Speaker ERes2Net) using the
    model already downloaded for sherpa-onnx diarization. Far more
    discriminative than the spectral VoiceFingerprint fallback: on synthetic
    multi-voice material, within-speaker cosine ~0.84 vs cross-speaker ~0.31,
    where the spectral fingerprint's ranges overlap (0.99 vs 0.97)."""

    kind = "sherpa-eres2net"

    def __init__(self, *, model_path: str | None = None, target_rate_hz: int = 16_000) -> None:
        import sherpa_onnx  # type: ignore

        path = model_path or str(default_sherpa_model_paths()[1])
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=path,
            num_threads=max(1, min(4, os.cpu_count() or 1)),
            provider="cpu",
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.target_rate_hz = target_rate_hz

    def embed(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        audio = prepare_audio(samples=samples, sample_rate_hz=sample_rate_hz, target_rate_hz=self.target_rate_hz)
        if audio.size < int(self.target_rate_hz * 0.4):
            raise ValueError("speaker sample is too short")
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.target_rate_hz, audio)
        stream.input_finished()
        embedding = np.array(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            raise ValueError("speaker sample has no voice fingerprint")
        return embedding / norm


def create_voice_embedder() -> VoiceFingerprint | SherpaVoiceEmbedder:
    """Best available embedder: sherpa ERes2Net when the diarization models are
    installed, else the dependency-free spectral fingerprint."""
    try:
        from .diarization import sherpa_models_present

        if sherpa_models_present():
            return SherpaVoiceEmbedder()
    except Exception:
        logger.info("sherpa voice embedder unavailable; using spectral fingerprint", exc_info=True)
    return VoiceFingerprint()


def default_match_threshold(embedder: Any) -> float:
    return 0.60 if getattr(embedder, "kind", "spectral") == "sherpa-eres2net" else 0.82


def bank_similarity(bank: tuple[np.ndarray, ...] | list[np.ndarray], embedding: np.ndarray) -> float:
    best = 0.0
    for exemplar in bank:
        if exemplar.shape != embedding.shape:
            continue
        best = max(best, cosine_similarity(embedding, exemplar))
    return best


def merge_into_bank(
    bank: list[np.ndarray],
    embedding: np.ndarray,
    *,
    merge_similarity: float = EXEMPLAR_MERGE_SIMILARITY,
    max_exemplars: int = MAX_PROFILE_EXEMPLARS,
) -> list[np.ndarray]:
    """Fold one embedding into an exemplar bank: average into the nearest
    exemplar when it is close enough, otherwise keep it as a new exemplar of
    the same voice. When the bank overflows, the two closest exemplars merge."""
    bank = [exemplar for exemplar in bank if exemplar.shape == embedding.shape]
    if not bank:
        return [embedding]

    similarities = [cosine_similarity(embedding, exemplar) for exemplar in bank]
    nearest = int(np.argmax(similarities))
    if similarities[nearest] >= merge_similarity:
        merged = _unit(bank[nearest] + embedding)
        bank[nearest] = merged
        return bank

    bank.append(embedding)
    while len(bank) > max_exemplars:
        best_pair, best_sim = (0, 1), -1.0
        for i in range(len(bank)):
            for j in range(i + 1, len(bank)):
                sim = cosine_similarity(bank[i], bank[j])
                if sim > best_sim:
                    best_pair, best_sim = (i, j), sim
        i, j = best_pair
        bank[i] = _unit(bank[i] + bank[j])
        del bank[j]
    return bank


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 1e-8 else vector.astype(np.float32)


class SpeakerMemory:
    def __init__(
        self,
        *,
        threshold: float = 0.82,
        fingerprinter: Optional[Any] = None,
    ) -> None:
        self.threshold = threshold
        self.fingerprinter = fingerprinter or VoiceFingerprint()
        self._profiles: dict[str, SpeakerProfile] = {}

    def profiles(self) -> list[SpeakerProfile]:
        return sorted(self._profiles.values(), key=lambda profile: profile.name.lower())

    def replace_profiles(self, items: Any) -> list[SpeakerProfile]:
        profiles: dict[str, SpeakerProfile] = {}
        if not isinstance(items, list):
            self._profiles = profiles
            return []

        for item in items:
            profile = speaker_profile_from_protocol(item)
            if profile is None:
                continue
            profiles[profile.profile_id] = profile

        self._profiles = profiles
        return self.profiles()

    def enroll(self, *, name: str, samples: np.ndarray, sample_rate_hz: int) -> SpeakerProfile:
        clean_name = " ".join(name.strip().split())
        if not clean_name:
            raise ValueError("speaker name is required")

        embedding = self.fingerprinter.embed(samples, sample_rate_hz)
        now = time.time()
        existing = self._profile_by_name(clean_name)
        if existing is not None:
            profile = self._with_exemplars(existing, [embedding], now=now)
        else:
            profile = SpeakerProfile(
                profile_id=profile_id_for_name(clean_name),
                name=clean_name,
                embedding=embedding.astype(np.float32),
                sample_count=1,
                created_at=now,
                updated_at=now,
                exemplars=(embedding.astype(np.float32),),
                kind="named",
            )

        self._profiles[profile.profile_id] = profile
        return profile

    def reinforce(self, profile_id: str, exemplars: list[np.ndarray]) -> Optional[SpeakerProfile]:
        """Fold a session's voice exemplars into a profile's bank so future
        sessions recognize that rendition of the voice too."""
        existing = self._profiles.get(profile_id)
        if existing is None or not exemplars:
            return None
        profile = self._with_exemplars(existing, exemplars, now=time.time())
        self._profiles[profile.profile_id] = profile
        return profile

    def add_profile(self, profile: SpeakerProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> Optional[SpeakerProfile]:
        return self._profiles.get(profile_id)

    def match(self, *, samples: np.ndarray, sample_rate_hz: int) -> Optional[SpeakerMatch]:
        if not self._profiles:
            return None
        try:
            embedding = self.fingerprinter.embed(samples, sample_rate_hz)
        except ValueError:
            return None
        return self.match_embedding(embedding)

    def match_embedding(self, embedding: np.ndarray) -> Optional[SpeakerMatch]:
        best: SpeakerMatch | None = None
        for profile in self._profiles.values():
            confidence = bank_similarity(profile.bank(), embedding)
            if best is None or confidence > best.confidence:
                best = SpeakerMatch(profile=profile, confidence=confidence)
        if best is None or best.confidence < self.threshold:
            return None
        return best

    @staticmethod
    def _with_exemplars(existing: SpeakerProfile, exemplars: list[np.ndarray], *, now: float) -> SpeakerProfile:
        bank = list(existing.bank())
        for exemplar in exemplars:
            bank = merge_into_bank(bank, exemplar.astype(np.float32))
        mean = _unit(np.sum(bank, axis=0)) if bank else existing.embedding
        return SpeakerProfile(
            profile_id=existing.profile_id,
            name=existing.name,
            embedding=mean,
            sample_count=max(1, existing.sample_count) + len(exemplars),
            created_at=existing.created_at,
            updated_at=now,
            exemplars=tuple(bank),
            kind=existing.kind,
        )

    def _profile_by_name(self, name: str) -> Optional[SpeakerProfile]:
        folded = name.casefold()
        for profile in self._profiles.values():
            if profile.name.casefold() == folded:
                return profile
        return None


class SpeakerReviewStore:
    def __init__(self, *, min_duration_sec: float = 0.5, max_duration_sec: float = 6.0) -> None:
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self._samples: dict[str, ReviewSample] = {}

    def observe(
        self,
        *,
        session_id: str,
        stream_id: StreamId,
        turn: DiarizationTurn,
        absolute_start_sec: float,
        absolute_end_sec: float,
        samples: np.ndarray,
        sample_rate_hz: int,
        match: Optional[SpeakerMatch],
    ) -> Optional[ReviewSample]:
        clip = clip_by_time(samples, sample_rate_hz, turn.start_sec, turn.end_sec, max_duration_sec=self.max_duration_sec)
        duration_sec = float(clip.size) / float(sample_rate_hz) if sample_rate_hz else 0.0
        if duration_sec < self.min_duration_sec:
            return None

        rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
        score = duration_sec * max(rms, 1e-4)
        clipped_end_sec = min(absolute_end_sec, absolute_start_sec + duration_sec)
        sample_id = review_sample_id(session_id=session_id, stream_id=stream_id, speaker_id=turn.speaker_id)
        existing = self._samples.get(sample_id)
        if existing is not None and existing.score >= score:
            return existing

        review = ReviewSample(
            sample_id=sample_id,
            session_id=session_id,
            stream_id=stream_id,
            source_speaker_id=turn.speaker_id,
            source_speaker_label=turn.speaker_label,
            start_sec=absolute_start_sec,
            end_sec=clipped_end_sec,
            sample_rate_hz=sample_rate_hz,
            samples=clip,
            score=score,
            matched_profile_id=match.profile.profile_id if match else None,
            matched_name=match.profile.name if match else None,
            match_confidence=match.confidence if match else None,
        )
        self._samples[sample_id] = review
        return review

    def samples_for_session(self, session_id: str) -> list[ReviewSample]:
        return sorted(
            [sample for sample in self._samples.values() if sample.session_id == session_id],
            key=lambda sample: (sample.start_sec, sample.source_speaker_id),
        )

    def get(self, sample_id: str) -> Optional[ReviewSample]:
        return self._samples.get(sample_id)

    def mark_enrolled(self, sample_id: str, profile: SpeakerProfile) -> None:
        sample = self._samples.get(sample_id)
        if sample is None:
            return
        sample.matched_profile_id = profile.profile_id
        sample.matched_name = profile.name
        sample.match_confidence = 1.0


def create_speaker_memory() -> Optional[SpeakerMemory]:
    if env_bool("SPEAKER_MEMORY_ENABLED", True) is False:
        return None
    embedder = create_voice_embedder()
    threshold = env_float("SPEAKER_MEMORY_THRESHOLD", default_match_threshold(embedder))
    return SpeakerMemory(threshold=threshold, fingerprinter=embedder)


def speaker_profile_from_protocol(item: Any) -> Optional[SpeakerProfile]:
    if not isinstance(item, dict):
        return None

    profile_id = item.get("profile_id")
    name = item.get("name")
    fingerprint = item.get("fingerprint") if "fingerprint" in item else item.get("embedding")
    if not isinstance(profile_id, str) or not profile_id.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(fingerprint, list) or not fingerprint:
        return None

    primary = _parse_fingerprint(fingerprint)
    if primary is None:
        return None

    bank: list[np.ndarray] = []
    raw_bank = item.get("fingerprints")
    if isinstance(raw_bank, list):
        for raw in raw_bank[:MAX_PROFILE_EXEMPLARS]:
            parsed = _parse_fingerprint(raw)
            if parsed is not None and parsed.shape == primary.shape:
                bank.append(parsed)
    if not bank:
        bank = [primary]

    kind = item.get("kind")
    return SpeakerProfile(
        profile_id=profile_id.strip(),
        name=" ".join(name.strip().split()),
        embedding=_unit(np.sum(bank, axis=0)),
        sample_count=max(1, coerce_int(item.get("sample_count"), default=1)),
        created_at=coerce_float(item.get("created_at"), default=0.0),
        updated_at=coerce_float(item.get("updated_at"), default=0.0),
        exemplars=tuple(bank),
        kind=kind if kind in ("named", "auto") else "named",
    )


def _parse_fingerprint(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, list) or not value:
        return None
    try:
        vector = np.array(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def prepare_audio(*, samples: np.ndarray, sample_rate_hz: int, target_rate_hz: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    audio = trim_silence(audio)
    if sample_rate_hz != target_rate_hz:
        audio = resample_linear(audio, sample_rate_hz, target_rate_hz)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def trim_silence(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    frame = 320
    hop = 160
    if samples.size < frame:
        return samples
    rms: list[float] = []
    starts: list[int] = []
    for start in range(0, samples.size - frame + 1, hop):
        chunk = samples[start:start + frame]
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
        starts.append(start)
    if not rms:
        return samples
    arr = np.array(rms, dtype=np.float32)
    threshold = max(0.01, float(np.max(arr)) * 0.12)
    active = np.where(arr >= threshold)[0]
    if active.size == 0:
        return samples
    start = max(0, starts[int(active[0])] - frame)
    end = min(samples.size, starts[int(active[-1])] + (2 * frame))
    return samples[start:end]


def frame_audio(samples: np.ndarray, *, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size < frame_size:
        return np.empty((0, frame_size), dtype=np.float32)
    count = 1 + ((samples.size - frame_size) // hop_size)
    frames = np.empty((count, frame_size), dtype=np.float32)
    for index in range(count):
        start = index * hop_size
        frames[index] = samples[start:start + frame_size]
    return frames


def mel_filterbank(*, freqs: np.ndarray, sample_rate_hz: int, band_count: int) -> np.ndarray:
    low_mel = hz_to_mel(80.0)
    high_mel = hz_to_mel(float(sample_rate_hz) / 2.0)
    mel_points = np.linspace(low_mel, high_mel, band_count + 2)
    hz_points = np.array([mel_to_hz(value) for value in mel_points], dtype=np.float32)
    filters = np.zeros((band_count, freqs.size), dtype=np.float32)
    for band in range(band_count):
        left, center, right = hz_points[band], hz_points[band + 1], hz_points[band + 2]
        rising = (freqs - left) / max(center - left, 1e-6)
        falling = (right - freqs) / max(right - center, 1e-6)
        filters[band] = np.maximum(0.0, np.minimum(rising, falling))
        total = float(np.sum(filters[band]))
        if total > 1e-8:
            filters[band] /= total
    return filters


def estimate_pitch_hz(frame: np.ndarray, sample_rate_hz: int) -> float:
    frame = frame.astype(np.float32, copy=False)
    frame = frame - float(np.mean(frame))
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
    min_lag = max(1, int(sample_rate_hz / 400.0))
    max_lag = min(corr.size - 1, int(sample_rate_hz / 70.0))
    if max_lag <= min_lag:
        return 0.0
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    if corr[lag] <= 0:
        return 0.0
    return float(sample_rate_hz) / float(lag)


def clip_by_time(samples: np.ndarray, sample_rate_hz: int, start_sec: float, end_sec: float, *, max_duration_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * sample_rate_hz))
    end = max(start, int(end_sec * sample_rate_hz))
    max_len = int(max_duration_sec * sample_rate_hz)
    if max_len > 0:
        end = min(end, start + max_len)
    return np.asarray(samples[start:end], dtype=np.float32)


def samples_to_wav_base64(samples: np.ndarray, sample_rate_hz: int) -> str:
    audio = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm)
    return base64.b64encode(out.getvalue()).decode("ascii")


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.size == 0 or src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    ratio = float(dst_rate) / float(src_rate)
    out_len = max(1, int(round(samples.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32, copy=False)


def quantiles(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(5, dtype=np.float32)
    return np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denom)))


def profile_id_for_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "speaker"
    digest = hashlib.sha256(name.strip().casefold().encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def review_sample_id(*, session_id: str, stream_id: str, speaker_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}|{stream_id}|{speaker_id}".encode("utf-8")).hexdigest()[:16]
    return f"review-{digest}"


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on", "enabled")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
