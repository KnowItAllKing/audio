from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

logger = logging.getLogger("server.diarization")


@dataclass(frozen=True)
class DiarizationTurn:
    start_sec: float
    end_sec: float
    speaker_id: str
    speaker_label: str
    speaker_confidence: float = 0.7


class DiarizationBackend(Protocol):
    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[DiarizationTurn]:
        """
        samples:
          - shape: (num_samples,) float32
          - range: [-1.0, 1.0]
        returns:
          - speaker turns relative to the provided samples
        """


class DiarizationSpeakerMapper:
    """Keep backend-local cluster labels stable across overlapping windows."""

    def __init__(self, *, min_overlap_sec: float = 0.2, history_sec: float = 120.0) -> None:
        self.min_overlap_sec = min_overlap_sec
        self.history_sec = history_sec
        self._turns: list[DiarizationTurn] = []
        self._next_speaker = 1

    def map_turns(
        self,
        turns: list[DiarizationTurn],
        *,
        base_offset_sec: float,
        window_duration_sec: float,
    ) -> list[DiarizationTurn]:
        if not turns:
            self._replace_history([], base_offset_sec=base_offset_sec, window_duration_sec=window_duration_sec)
            return []

        local_ids = list(dict.fromkeys(turn.speaker_id for turn in turns))
        global_ids = list(dict.fromkeys(turn.speaker_id for turn in self._turns))
        scores: list[tuple[float, str, str]] = []
        for local_id in local_ids:
            local_turns = [turn for turn in turns if turn.speaker_id == local_id]
            local_duration = sum(max(0.0, turn.end_sec - turn.start_sec) for turn in local_turns)
            for global_id in global_ids:
                overlap = 0.0
                for local_turn in local_turns:
                    absolute_start = base_offset_sec + local_turn.start_sec
                    absolute_end = base_offset_sec + local_turn.end_sec
                    for historical_turn in self._turns:
                        if historical_turn.speaker_id != global_id:
                            continue
                        overlap += max(
                            0.0,
                            min(absolute_end, historical_turn.end_sec)
                            - max(absolute_start, historical_turn.start_sec),
                        )
                if overlap >= self.min_overlap_sec and overlap >= min(local_duration, 1.0) * 0.25:
                    scores.append((overlap, local_id, global_id))

        assignments: dict[str, str] = {}
        assigned_globals: set[str] = set()
        for _, local_id, global_id in sorted(scores, reverse=True):
            if local_id in assignments or global_id in assigned_globals:
                continue
            assignments[local_id] = global_id
            assigned_globals.add(global_id)

        labels_by_id = {turn.speaker_id: turn.speaker_label for turn in self._turns}
        for local_id in local_ids:
            if local_id in assignments:
                continue
            global_id = f"diarization:session-speaker-{self._next_speaker:03d}"
            self._next_speaker += 1
            assignments[local_id] = global_id
            labels_by_id[global_id] = f"Speaker {self._next_speaker - 1}"

        mapped = [
            DiarizationTurn(
                start_sec=turn.start_sec,
                end_sec=turn.end_sec,
                speaker_id=assignments[turn.speaker_id],
                speaker_label=labels_by_id.get(
                    assignments[turn.speaker_id],
                    f"Speaker {self._speaker_number(assignments[turn.speaker_id])}",
                ),
                speaker_confidence=turn.speaker_confidence,
            )
            for turn in turns
        ]
        self._replace_history(mapped, base_offset_sec=base_offset_sec, window_duration_sec=window_duration_sec)
        return mapped

    def _replace_history(
        self,
        turns: list[DiarizationTurn],
        *,
        base_offset_sec: float,
        window_duration_sec: float,
    ) -> None:
        window_end_sec = base_offset_sec + max(0.0, window_duration_sec)
        keep_after_sec = window_end_sec - self.history_sec
        historical = [
            turn
            for turn in self._turns
            if turn.end_sec <= base_offset_sec and turn.end_sec >= keep_after_sec
        ]
        historical.extend(
            DiarizationTurn(
                start_sec=base_offset_sec + turn.start_sec,
                end_sec=base_offset_sec + turn.end_sec,
                speaker_id=turn.speaker_id,
                speaker_label=turn.speaker_label,
                speaker_confidence=turn.speaker_confidence,
            )
            for turn in turns
        )
        self._turns = historical

    @staticmethod
    def _speaker_number(speaker_id: str) -> int:
        try:
            return int(speaker_id.rsplit("-", 1)[-1])
        except ValueError:
            return 0


class PyannoteDiarizationBackend:
    """
    Current local diarization backend using pyannote.audio 4 Community-1.

    Requires:
      - optional `pyannote.audio>=4` install
      - Hugging Face token in DIARIZATION_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN
      - access accepted for the selected pyannote model
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import torch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
            from pyannote.audio import Pipeline  # type: ignore

        self._torch = torch
        self.device = device or os.environ.get("DIARIZATION_DEVICE") or os.environ.get("WHISPER_DEVICE") or "cpu"
        self.hf_token = hf_token or _required_hf_token()
        self.model_name = model_name or os.environ.get("DIARIZATION_MODEL") or "pyannote/speaker-diarization-community-1"
        self.min_speakers = _env_int_optional("DIARIZATION_MIN_SPEAKERS")
        self.max_speakers = _env_int_optional("DIARIZATION_MAX_SPEAKERS")
        self.min_turn_sec = _env_float("DIARIZATION_MIN_TURN_SEC", 0.2)

        try:
            self._pipeline = Pipeline.from_pretrained(self.model_name, token=self.hf_token)
            self._pipeline.to(torch.device(self.device))
        except TypeError:
            try:
                self._pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)
                self._pipeline.to(torch.device(self.device))
            except Exception:
                raise RuntimeError(_model_load_error_message(self.model_name)) from None
        except Exception:
            raise RuntimeError(_model_load_error_message(self.model_name)) from None

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[DiarizationTurn]:
        audio = _prepare_audio(samples=samples, sample_rate=sample_rate, target_rate=16_000)
        waveform = self._torch.from_numpy(audio).unsqueeze(0)
        kwargs: dict[str, int] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        result = self._pipeline({"waveform": waveform, "sample_rate": 16_000}, **kwargs)
        annotation = getattr(result, "exclusive_speaker_diarization", None) or getattr(result, "speaker_diarization", None) or result
        return _turns_from_pyannote_annotation(annotation, min_turn_sec=self.min_turn_sec)


class SherpaOnnxDiarizationBackend:
    """Token-free local diarization using sherpa-onnx models."""

    def __init__(
        self,
        *,
        segmentation_model: str | Path | None = None,
        embedding_model: str | Path | None = None,
    ) -> None:
        import sherpa_onnx  # type: ignore

        default_segmentation, default_embedding = default_sherpa_model_paths()
        self.segmentation_model = Path(
            segmentation_model
            or os.environ.get("DIARIZATION_SEGMENTATION_MODEL", "")
            or default_segmentation
        ).expanduser()
        self.embedding_model = Path(
            embedding_model
            or os.environ.get("DIARIZATION_EMBEDDING_MODEL", "")
            or default_embedding
        ).expanduser()
        self.device = "cpu"
        self.model_name = "sherpa-onnx/pyannote-segmentation-3.0+3dspeaker"
        self.min_turn_sec = _env_float("DIARIZATION_MIN_TURN_SEC", 0.2)
        num_speakers = _env_int_optional("DIARIZATION_NUM_SPEAKERS")
        min_speakers = _env_int_optional("DIARIZATION_MIN_SPEAKERS")
        max_speakers = _env_int_optional("DIARIZATION_MAX_SPEAKERS")
        if num_speakers is None and min_speakers is not None and min_speakers == max_speakers:
            num_speakers = min_speakers

        if not self.segmentation_model.is_file() or not self.embedding_model.is_file():
            raise RuntimeError(
                "sherpa-onnx diarization models are missing; run `make setup-diarization` "
                "or set DIARIZATION_SEGMENTATION_MODEL and DIARIZATION_EMBEDDING_MODEL"
            )

        num_threads = max(1, _env_int("DIARIZATION_NUM_THREADS", min(4, os.cpu_count() or 1)))
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model)
                ),
                num_threads=num_threads,
                provider="cpu",
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model),
                num_threads=num_threads,
                provider="cpu",
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers if num_speakers is not None else -1,
                threshold=_env_float("DIARIZATION_CLUSTER_THRESHOLD", 0.5),
            ),
            min_duration_on=self.min_turn_sec,
            min_duration_off=_env_float("DIARIZATION_MIN_SILENCE_SEC", 0.3),
        )
        if not config.validate():
            raise RuntimeError("invalid sherpa-onnx diarization configuration")
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[DiarizationTurn]:
        audio = _prepare_audio(samples=samples, sample_rate=sample_rate, target_rate=16_000)
        result = self._diarizer.process(audio).sort_by_start_time()
        return [
            DiarizationTurn(
                start_sec=float(turn.start),
                end_sec=float(turn.end),
                speaker_id=f"diarization:sherpa-{int(turn.speaker):02d}",
                speaker_label=f"Speaker {int(turn.speaker) + 1}",
                speaker_confidence=0.75,
            )
            for turn in result
            if float(turn.end) - float(turn.start) >= self.min_turn_sec
        ]


def create_diarization_backend() -> Optional[DiarizationBackend]:
    backend = os.environ.get("DIARIZATION_BACKEND", "auto").strip().lower() or "auto"
    if backend in ("0", "false", "off", "disabled", "none"):
        return None

    try:
        if backend == "auto":
            pyannote_error: Exception | None = None
            if _hf_token_present():
                try:
                    out = PyannoteDiarizationBackend()
                    logger.info("diarization backend: pyannote model=%s device=%s", out.model_name, out.device)
                    return out
                except Exception as exc:
                    pyannote_error = exc
                    logger.warning(
                        "pyannote diarization is unavailable; trying local sherpa-onnx models",
                        exc_info=True,
                    )
            if sherpa_models_present():
                out = SherpaOnnxDiarizationBackend()
                logger.info("diarization backend: sherpa-onnx model=%s device=%s", out.model_name, out.device)
                return out
            if pyannote_error is not None:
                raise pyannote_error
            logger.info("diarization backend: auto disabled; local models are not installed")
            return None
        if backend in ("1", "true", "on", "enabled", "pyannote", "pyannote-community", "community"):
            out = PyannoteDiarizationBackend()
            logger.info("diarization backend: pyannote model=%s device=%s", out.model_name, out.device)
            return out
        if backend in ("sherpa", "sherpa-onnx", "onnx"):
            out = SherpaOnnxDiarizationBackend()
            logger.info("diarization backend: sherpa-onnx model=%s device=%s", out.model_name, out.device)
            return out
        raise ValueError(f"unsupported DIARIZATION_BACKEND={backend!r}")
    except Exception:
        if _env_bool("DIARIZATION_STRICT", False):
            raise
        logger.exception("Failed to init diarization backend; continuing without diarization")
        return None


def should_diarize_stream(stream_id: str) -> bool:
    streams = os.environ.get("DIARIZATION_STREAMS", "system")
    allowed = {item.strip().lower() for item in streams.split(",") if item.strip()}
    return stream_id.lower() in allowed


def default_sherpa_model_paths() -> tuple[Path, Path]:
    cache_root = Path(
        os.environ.get("CADENCE_MODEL_CACHE", "")
        or (Path.home() / ".cache" / "cadence" / "diarization")
    ).expanduser()
    return (
        cache_root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx",
        cache_root / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    )


def sherpa_models_present() -> bool:
    segmentation_model, embedding_model = default_sherpa_model_paths()
    return segmentation_model.is_file() and embedding_model.is_file()


def _turns_from_pyannote_annotation(annotation: Any, *, min_turn_sec: float) -> list[DiarizationTurn]:
    speaker_labels: dict[str, str] = {}
    out: list[DiarizationTurn] = []
    for item in _iter_pyannote_tracks(annotation):
        start = _optional_float(item.get("start"))
        end = _optional_float(item.get("end"))
        raw_speaker = item.get("speaker")
        if start is None or end is None or end <= start:
            continue
        if (end - start) < min_turn_sec:
            continue
        if raw_speaker is None:
            continue

        speaker_key = str(raw_speaker)
        label = speaker_labels.get(speaker_key)
        if label is None:
            label = f"Speaker {len(speaker_labels) + 1}"
            speaker_labels[speaker_key] = label
        out.append(
            DiarizationTurn(
                start_sec=float(start),
                end_sec=float(end),
                speaker_id=f"diarization:{speaker_key}",
                speaker_label=label,
                speaker_confidence=0.75,
            )
        )
    return out


def _iter_rows(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "iterrows"):
        return [dict(row) for _, row in result.iterrows()]
    if isinstance(result, list):
        return [dict(item) for item in result if isinstance(item, dict)]
    if isinstance(result, tuple):
        return _iter_rows(list(result))
    return []


def _iter_pyannote_tracks(annotation: Any) -> list[dict[str, Any]]:
    if not hasattr(annotation, "itertracks"):
        return _iter_rows(annotation)

    rows: list[dict[str, Any]] = []
    for item in annotation.itertracks(yield_label=True):
        if len(item) == 3:
            segment, _, label = item
        elif len(item) == 2:
            segment, label = item
        else:
            continue
        rows.append(
            {
                "start": getattr(segment, "start", None),
                "end": getattr(segment, "end", None),
                "speaker": label,
            }
        )
    return rows


def _prepare_audio(*, samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    if sample_rate == target_rate:
        return samples
    return _resample_linear(samples, sample_rate, target_rate)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.size == 0 or src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    ratio = float(dst_rate) / float(src_rate)
    out_len = max(1, int(round(samples.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32, copy=False)


def _required_hf_token() -> str:
    token = (
        os.environ.get("DIARIZATION_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        raise ValueError("DIARIZATION_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN is required for diarization")
    return token


def _hf_token_present() -> bool:
    return bool(
        os.environ.get("DIARIZATION_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def _model_load_error_message(model_name: str = "pyannote/speaker-diarization-community-1") -> str:
    return (
        f"Failed to load diarization model {model_name!r}. Check the Hugging Face token "
        "and accept access terms for the selected pyannote model."
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_int_optional(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring", name, raw)
        return None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
