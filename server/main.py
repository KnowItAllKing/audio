from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import websockets
from websockets import ServerConnection

from .diarization import (
    DiarizationBackend,
    DiarizationSpeakerMapper,
    DiarizationTurn,
    create_diarization_backend,
    should_diarize_stream,
)
from .protocol_types import (
    AudioChunkMessage,
    ControlMessage,
    ErrorMessage,
    StreamId,
    TranscriptLayer,
    TranscriptSegment,
    TranscriptUpdateMessage,
)
from .session_state import SessionState, StreamBuffer
from .speaker_identity import SessionSpeakerRegistry, turn_embeddings
from .speaker_memory import (
    SpeakerMatch,
    SpeakerMemory,
    create_voice_embedder,
    SpeakerReviewStore,
    clip_by_time,
    create_speaker_memory,
)
from .speakers import SpeakerTracker
from .transcription import bytes_per_second, compute_window, eligible_for_transcription
from .utterances import UtteranceAssembler, UtteranceAssemblerConfig
from .whisper_backend import TranscriptSegment as WhisperTranscriptSegment
from .whisper_backend import WhisperBackend, create_backend


logger = logging.getLogger("server")


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def _new_webrtc_vad(aggressiveness: int) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
            module="webrtcvad",
        )
        import webrtcvad

    return webrtcvad.Vad(aggressiveness)


def _rms_energy(samples: np.ndarray) -> float:
    """
    Root-mean-square energy for float32 mono samples in [-1, 1].
    Used as a lightweight VAD gate to avoid transcribing near-silence.
    """
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.square(x))))


def _webrtcvad_window_has_speech(
    *,
    pcm_s16le: bytes,
    sample_rate_hz: int,
    aggressiveness: int,
    frame_ms: int,
    min_speech_ratio: float,
) -> bool:
    """
    Returns True if WebRTC VAD detects speech in enough frames within this window.
    - pcm_s16le: mono, 16-bit little-endian PCM bytes
    - sample_rate_hz: one of 8000/16000/32000/48000 for WebRTC VAD
    - frame_ms: 10/20/30 (WebRTC VAD supported frame sizes)
    - min_speech_ratio: fraction of frames that must be speech
    """
    if sample_rate_hz not in (8000, 16000, 32000, 48000):
        return True
    if frame_ms not in (10, 20, 30):
        return True
    if not pcm_s16le:
        return False

    # Ensure even byte length for int16 framing.
    if (len(pcm_s16le) % 2) != 0:
        pcm_s16le = pcm_s16le[: len(pcm_s16le) - 1]
    if not pcm_s16le:
        return False

    bytes_per_frame = int(sample_rate_hz * (frame_ms / 1000.0) * 2)  # s16le mono
    if bytes_per_frame <= 0:
        return True

    vad = _new_webrtc_vad(int(max(0, min(3, aggressiveness))))
    total = 0
    speech = 0
    for off in range(0, len(pcm_s16le) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm_s16le[off : off + bytes_per_frame]
        total += 1
        try:
            if vad.is_speech(frame, sample_rate_hz):
                speech += 1
        except Exception:
            # If VAD errors out on some frame, fail open for this window.
            return True

    if total == 0:
        return False
    return (speech / float(total)) >= float(min_speech_ratio)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _error(session_id: str, code: str, message: str) -> ErrorMessage:
    return {
        "type": "error",
        "session_id": session_id,
        "code": code,
        "message": message,
    }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _validate_audio_chunk(obj: Any) -> AudioChunkMessage:
    if not isinstance(obj, dict):
        raise ValueError("message must be a JSON object")

    # Required top-level fields
    required = ["type", "session_id", "stream_id", "seq", "timestamp_ms", "audio_format", "audio_base64"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    if obj.get("type") != "audio_chunk":
        raise ValueError('type must be "audio_chunk"')

    session_id = obj["session_id"]
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")

    stream_id = obj["stream_id"]
    if stream_id not in ("mic", "system"):
        raise ValueError('stream_id must be "mic" or "system"')

    seq = obj["seq"]
    if not isinstance(seq, int) or seq < 0:
        raise ValueError("seq must be a non-negative integer")

    timestamp_ms = obj["timestamp_ms"]
    if not isinstance(timestamp_ms, (int, float)):
        raise ValueError("timestamp_ms must be a number")

    audio_format = obj["audio_format"]
    if not isinstance(audio_format, dict):
        raise ValueError("audio_format must be an object")
    for k in ("encoding", "sample_rate_hz", "num_channels"):
        if k not in audio_format:
            raise ValueError(f"audio_format.{k} is required")

    if audio_format.get("encoding") != "pcm_s16le":
        raise ValueError('audio_format.encoding must be "pcm_s16le"')
    if not isinstance(audio_format.get("sample_rate_hz"), int):
        raise ValueError("audio_format.sample_rate_hz must be an integer")
    if not isinstance(audio_format.get("num_channels"), int):
        raise ValueError("audio_format.num_channels must be an integer")

    audio_base64 = obj["audio_base64"]
    if not isinstance(audio_base64, str):
        raise ValueError("audio_base64 must be a string")

    return cast(AudioChunkMessage, obj)


def _validate_control_message(obj: Any) -> ControlMessage:
    if not isinstance(obj, dict):
        raise ValueError("message must be a JSON object")
    if obj.get("type") != "control":
        raise ValueError('type must be "control"')

    session_id = obj.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")

    command = obj.get("command")
    if command not in (
        "start",
        "stop",
        "ping",
        "pong",
        "metadata",
        "speaker_activity",
        "speaker_memory_review",
        "speaker_memory_enroll",
        "speaker_memory_profiles",
    ):
        raise ValueError("unsupported control command")

    payload = obj.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("control payload must be an object")

    return cast(ControlMessage, obj)


def _decode_audio_base64(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        # Some base64 encoders omit padding; try permissive decode as fallback.
        try:
            return base64.b64decode(b64 + "==")
        except Exception:
            raise ValueError(f"invalid audio_base64: {e}") from e


def _pcm_s16le_bytes_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2 != 0:
        pcm = pcm[: len(pcm) - 1]
    i16 = np.frombuffer(pcm, dtype="<i2")
    return i16.astype(np.float32) / 32768.0


@dataclass(frozen=True)
class PendingTranscriptWindow:
    start_offset_bytes: int
    end_offset_bytes: int
    segments: tuple[WhisperTranscriptSegment, ...]


class ServerState:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.session_sockets: dict[str, set[ServerConnection]] = {}
        self.backend: WhisperBackend = create_backend()
        self.speaker_memory: SpeakerMemory | None = create_speaker_memory()
        # Voice identity across windows needs a discriminative embedder; the
        # spectral fallback would merge distinct voices, so identity resolution
        # only activates alongside the sherpa embedding model.
        self.voice_embedder = (
            self.speaker_memory.fingerprinter if self.speaker_memory is not None else create_voice_embedder()
        )
        self.identity_enabled = (
            _env_bool("DIARIZATION_IDENTITY_ENABLED", True)
            and getattr(self.voice_embedder, "kind", "spectral") == "sherpa-eres2net"
        )
        if self.identity_enabled:
            # With an identity layer on top, within-window clustering should
            # over-split rather than merge similar voices — the registry
            # re-merges clusters of the same voice, but it cannot split a
            # cluster that glued two speakers together (per-turn embeddings
            # recover most of it, still the cleaner input helps).
            os.environ.setdefault("DIARIZATION_CLUSTER_THRESHOLD", "0.35")
        self.diarizer: DiarizationBackend | None = create_diarization_backend()
        self.speaker_review = SpeakerReviewStore(
            min_duration_sec=_env_float("SPEAKER_MEMORY_MIN_CLIP_SEC", 0.5),
            max_duration_sec=_env_float("SPEAKER_MEMORY_MAX_CLIP_SEC", 6.0),
        )
        self.utterance_config = UtteranceAssemblerConfig(
            pause_sec=_env_float("UTTERANCE_PAUSE_SEC", 1.2),
            max_duration_sec=_env_float("UTTERANCE_MAX_SEC", 14.0),
            emit_partials=_env_bool("UTTERANCE_EMIT_PARTIALS", True),
        )
        self.utterance_assemblers: dict[tuple[str, StreamId], UtteranceAssembler] = {}
        self.diarization_mappers: dict[tuple[str, StreamId], DiarizationSpeakerMapper] = {}
        self.identity_registries: dict[tuple[str, StreamId], SessionSpeakerRegistry] = {}
        self.transcription_locks: dict[tuple[str, StreamId], asyncio.Lock] = {}
        self.pending_transcripts: dict[
            tuple[str, StreamId],
            list[PendingTranscriptWindow],
        ] = {}
        self.speaker_trackers: dict[str, SpeakerTracker] = {}
        self.lock = asyncio.Lock()

    def get_utterance_assembler(self, session_id: str, stream_id: StreamId) -> UtteranceAssembler:
        key = (session_id, stream_id)
        assembler = self.utterance_assemblers.get(key)
        if assembler is None:
            assembler = UtteranceAssembler(
                session_id=f"{stream_id}-{session_id}",
                config=self.utterance_config,
            )
            self.utterance_assemblers[key] = assembler
        return assembler

    def get_session_assemblers(self, session_id: str) -> list[UtteranceAssembler]:
        return [
            assembler
            for (candidate_session_id, _), assembler in self.utterance_assemblers.items()
            if candidate_session_id == session_id
        ]

    def get_diarization_mapper(self, session_id: str, stream_id: StreamId) -> DiarizationSpeakerMapper:
        key = (session_id, stream_id)
        mapper = self.diarization_mappers.get(key)
        if mapper is None:
            mapper = DiarizationSpeakerMapper(
                min_overlap_sec=_env_float("DIARIZATION_MAPPING_MIN_OVERLAP_SEC", 0.2),
                history_sec=_env_float("DIARIZATION_MAPPING_HISTORY_SEC", 120.0),
            )
            self.diarization_mappers[key] = mapper
        return mapper

    def get_identity_registry(self, session_id: str, stream_id: StreamId) -> SessionSpeakerRegistry:
        key = (session_id, stream_id)
        registry = self.identity_registries.get(key)
        if registry is None:
            registry = SessionSpeakerRegistry(
                match_threshold=_env_float("DIARIZATION_IDENTITY_THRESHOLD", 0.68),
                match_margin=_env_float("DIARIZATION_IDENTITY_MARGIN", 0.05),
            )
            self.identity_registries[key] = registry
        # Profiles can arrive at any point before or during the session;
        # seeding is idempotent per profile id.
        if self.speaker_memory is not None:
            registry.seed_profiles(self.speaker_memory.profiles())
        return registry

    def get_transcription_lock(self, session_id: str, stream_id: StreamId) -> asyncio.Lock:
        key = (session_id, stream_id)
        lock = self.transcription_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.transcription_locks[key] = lock
        return lock

    def get_pending_transcripts(
        self,
        session_id: str,
        stream_id: StreamId,
    ) -> list[PendingTranscriptWindow]:
        return self.pending_transcripts.setdefault((session_id, stream_id), [])

    def get_speaker_tracker(self, session_id: str) -> SpeakerTracker:
        tracker = self.speaker_trackers.get(session_id)
        if tracker is None:
            tracker = SpeakerTracker(
                local_label=os.environ.get("MIC_SPEAKER_LABEL", "You"),
                system_label=os.environ.get("SYSTEM_SPEAKER_LABEL", "System audio"),
                active_speaker_ttl_sec=_env_float("SPEAKER_ACTIVITY_TTL_SEC", 15.0),
                activity_boundary_tolerance_sec=_env_float(
                    "SPEAKER_ACTIVITY_BOUNDARY_TOLERANCE_SEC",
                    0.35,
                ),
            )
            self.speaker_trackers[session_id] = tracker
        return tracker


def _build_transcript_update(
    session_id: str,
    segments: list[TranscriptSegment],
    *,
    layer: TranscriptLayer = "processed",
) -> TranscriptUpdateMessage:
    return {
        "type": "transcript_update",
        "session_id": session_id,
        "layer": layer,
        "segments": segments,
    }


def _raw_transcript_segments(
    state: ServerState,
    session: SessionState,
    stream_id: StreamId,
    segments: list[WhisperTranscriptSegment],
    *,
    window_start_offset_bytes: int,
    bytes_per_sec: int,
    buf: StreamBuffer,
) -> list[TranscriptSegment]:
    epoch_sec = (buf.first_timestamp_ms / 1000.0) if buf.first_timestamp_ms else 0.0
    base_offset_sec = float(window_start_offset_bytes) / float(bytes_per_sec)
    speaker = state.get_speaker_tracker(session.session_id).stream_speakers[stream_id]
    out: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        text = str(segment.text).strip()
        if not text:
            continue
        out.append(
            {
                "id": f"raw-{stream_id}-{window_start_offset_bytes:012d}-{index:03d}",
                "start_sec": epoch_sec + base_offset_sec + float(segment.start_sec),
                "end_sec": epoch_sec + base_offset_sec + float(segment.end_sec),
                "text": text,
                "stream_tags": [stream_id],
                "speaker_id": speaker.speaker_id,
                "speaker_label": speaker.speaker_label,
                "speaker_source": speaker.speaker_source,
                "speaker_confidence": speaker.speaker_confidence,
                "is_final": True,
                "full_context_available": False,
                "final_reason": "raw_window",
            }
        )
    return out


def _assemble_transcript_segments(
    state: ServerState,
    session: SessionState,
    stream_id: StreamId,
    raw_segments: list,
    base_offset_sec: float,
    buf: StreamBuffer,
) -> list[TranscriptSegment]:
    # Convert client epoch (ms) to seconds; fall back to 0 if not set.
    epoch_sec = (buf.first_timestamp_ms / 1000.0) if buf.first_timestamp_ms else 0.0
    tracker = state.get_speaker_tracker(session.session_id)
    assembler = state.get_utterance_assembler(session.session_id, stream_id)

    out: list[TranscriptSegment] = []
    for seg in raw_segments:
        words = tuple(getattr(seg, "words", ()) or ())
        if words:
            fragments: list[tuple[float, float, str, Any]] = []
            for word in words:
                word_start = float(epoch_sec + base_offset_sec + float(word.start_sec))
                word_end = float(epoch_sec + base_offset_sec + float(word.end_sec))
                speaker = tracker.resolve(stream_id=stream_id, start_sec=word_start, end_sec=word_end)
                if fragments and fragments[-1][3].speaker_id == speaker.speaker_id:
                    previous = fragments[-1]
                    fragments[-1] = (
                        previous[0],
                        max(previous[1], word_end),
                        _join_transcript_text(previous[2], word.text),
                        previous[3],
                    )
                else:
                    fragments.append((word_start, word_end, str(word.text).strip(), speaker))
        else:
            start_sec = float(epoch_sec + base_offset_sec + float(seg.start_sec))
            end_sec = float(epoch_sec + base_offset_sec + float(seg.end_sec))
            fragments = [
                (
                    start_sec,
                    end_sec,
                    seg.text,
                    tracker.resolve(stream_id=stream_id, start_sec=start_sec, end_sec=end_sec),
                )
            ]

        for start_sec, end_sec, text, speaker in fragments:
            out.extend(
                item.to_protocol()
                for item in assembler.push(
                    stream_id=stream_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    speaker=speaker,
                )
            )
    return out


def _join_transcript_text(left: str, right: str) -> str:
    left = str(left).strip()
    right = str(right).strip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.!?:;)]}":
        return f"{left.rstrip()}{right}"
    return f"{left.rstrip()} {right.lstrip()}"


def _apply_diarization_turns(
    state: ServerState,
    session: SessionState,
    stream_id: StreamId,
    turns: list[DiarizationTurn],
    base_offset_sec: float,
    buf: StreamBuffer,
    samples: np.ndarray,
    sample_rate_hz: int,
    replace_start_sec: float | None = None,
    replace_end_sec: float | None = None,
    identity_registry: SessionSpeakerRegistry | None = None,
) -> None:
    epoch_sec = (buf.first_timestamp_ms / 1000.0) if buf.first_timestamp_ms else 0.0
    tracker = state.get_speaker_tracker(session.session_id)
    if replace_start_sec is not None and replace_end_sec is not None:
        tracker.remove_activities_in_range(
            stream_id=stream_id,
            start_sec=epoch_sec + replace_start_sec,
            end_sec=epoch_sec + replace_end_sec,
            sources={"diarization", "memory"},
        )
    if not turns:
        return
    for turn in turns:
        start_sec = float(epoch_sec + base_offset_sec + turn.start_sec)
        end_sec = float(epoch_sec + base_offset_sec + turn.end_sec)
        speaker_id = turn.speaker_id
        speaker_label = turn.speaker_label
        speaker_source = "diarization"
        speaker_confidence = turn.speaker_confidence
        match = None

        if identity_registry is not None:
            # Window-level identity resolution already decided who this is
            # (including profile matches); per-turn re-matching on short clips
            # would only second-guess it with worse evidence.
            info = identity_registry.match_info(turn.speaker_id)
            if info is not None and state.speaker_memory is not None:
                profile = state.speaker_memory.get(info[0])
                if profile is not None:
                    match = SpeakerMatch(profile=profile, confidence=info[2])
                    speaker_source = "memory"
        elif state.speaker_memory is not None:
            clip = clip_by_time(
                samples,
                sample_rate_hz,
                turn.start_sec,
                turn.end_sec,
                max_duration_sec=_env_float("SPEAKER_MEMORY_MAX_CLIP_SEC", 6.0),
            )
            match = state.speaker_memory.match(samples=clip, sample_rate_hz=sample_rate_hz)
            if match is not None:
                speaker_id = f"memory:{match.profile.profile_id}"
                speaker_label = match.profile.name
                speaker_source = "memory"
                speaker_confidence = match.confidence

        state.speaker_review.observe(
            session_id=session.session_id,
            stream_id=stream_id,
            turn=turn,
            absolute_start_sec=start_sec,
            absolute_end_sec=end_sec,
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            match=match,
        )
        tracker.apply_activity(
            {
                "stream_id": stream_id,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "speaker_source": speaker_source,
                "speaker_confidence": speaker_confidence,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )


async def _send_transcript_segments(
    state: ServerState,
    session_id: str,
    segments: list[TranscriptSegment],
    sockets: set[ServerConnection],
    *,
    layer: TranscriptLayer = "processed",
) -> None:
    if not segments or not sockets:
        return

    payload = json.dumps(_build_transcript_update(session_id, segments, layer=layer))
    dead: list[ServerConnection] = []
    for ws in sockets:
        try:
            await ws.send(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with state.lock:
            sset = state.session_sockets.get(session_id)
            if sset:
                for ws in dead:
                    sset.discard(ws)


async def _send_speaker_memory_review(
    ws: ServerConnection,
    state: ServerState,
    session_id: str,
) -> None:
    async with state.lock:
        samples = [sample.to_protocol() for sample in state.speaker_review.samples_for_session(session_id)]
    await ws.send(
        json.dumps(
            {
                "type": "speaker_memory_review",
                "session_id": session_id,
                "samples": samples,
            }
        )
    )


async def _send_speaker_memory_profiles(
    ws: ServerConnection,
    state: ServerState,
    session_id: str,
) -> None:
    async with state.lock:
        profiles = state.speaker_memory.profiles() if state.speaker_memory is not None else []
    await ws.send(
        json.dumps(
            {
                "type": "speaker_memory_profiles",
                "session_id": session_id,
                "profiles": [profile.to_protocol() for profile in profiles],
            }
        )
    )


async def _handle_speaker_memory_profiles(
    ws: ServerConnection,
    state: ServerState,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    profiles_payload = payload.get("profiles")
    if state.speaker_memory is not None and isinstance(profiles_payload, list):
        async with state.lock:
            state.speaker_memory.replace_profiles(profiles_payload)
    await _send_speaker_memory_profiles(ws, state, session_id)


async def _handle_speaker_memory_enroll(
    ws: ServerConnection,
    state: ServerState,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    if state.speaker_memory is None:
        await ws.send(
            json.dumps(
                _error(
                    session_id=session_id,
                    code="speaker_memory_disabled",
                    message="speaker memory is disabled",
                )
            )
        )
        return

    sample_id = payload.get("sample_id")
    name = payload.get("name")
    if not isinstance(sample_id, str) or not sample_id:
        await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="sample_id is required")))
        return
    if not isinstance(name, str) or not name.strip():
        await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="name is required")))
        return

    async with state.lock:
        sample = state.speaker_review.get(sample_id)
    if sample is None or sample.session_id != session_id:
        await ws.send(
            json.dumps(
                _error(
                    session_id=session_id,
                    code="sample_not_found",
                    message="speaker review sample was not found",
                )
            )
        )
        return

    try:
        profile = state.speaker_memory.enroll(name=name, samples=sample.samples, sample_rate_hz=sample.sample_rate_hz)
    except Exception as e:
        await ws.send(json.dumps(_error(session_id=session_id, code="speaker_enroll_failed", message=str(e))))
        return

    async with state.lock:
        state.speaker_review.mark_enrolled(sample_id, profile)

    await ws.send(
        json.dumps(
            {
                "type": "speaker_memory_profile",
                "session_id": session_id,
                "profile": profile.to_protocol(),
                "profiles": [item.to_protocol() for item in state.speaker_memory.profiles()],
            }
        )
    )
    await _send_speaker_memory_review(ws, state, session_id)


async def _transcribe_stream(
    state: ServerState,
    sess: SessionState,
    stream_id: StreamId,
    buf: StreamBuffer,
    min_new_audio_sec: float,
    window_sec: float,
    max_buffer_sec: float,
    *,
    force: bool = False,
) -> None:
    """Transcribe a single stream buffer and send updates."""
    async with state.get_transcription_lock(sess.session_id, stream_id):
        await _transcribe_stream_locked(
            state,
            sess,
            stream_id,
            buf,
            min_new_audio_sec,
            window_sec,
            max_buffer_sec,
            force=force,
        )


async def _transcribe_stream_locked(
    state: ServerState,
    sess: SessionState,
    stream_id: StreamId,
    buf: StreamBuffer,
    min_new_audio_sec: float,
    window_sec: float,
    max_buffer_sec: float,
    *,
    force: bool,
) -> None:
    bps = bytes_per_second(sess.sample_rate_hz, sess.num_channels)
    if bps <= 0:
        return

    buf.trim_to_max_bytes(int(max_buffer_sec * bps))
    end_off = buf.buffer_end_offset_bytes()
    idle_sec = (
        max(0.0, time.time() - buf.last_received_at)
        if buf.last_received_at is not None
        else 0.0
    )
    min_new_bytes = int(max(0.0, min_new_audio_sec) * bps)
    raw_ready = end_off > buf.last_transcribed_offset_bytes and (
        force
        or idle_sec >= max(0.0, _env_float("RAW_TRANSCRIPT_IDLE_FLUSH_SEC", 0.8))
        or eligible_for_transcription(
            end_off,
            buf.last_transcribed_offset_bytes,
            min_new_bytes,
        )
    )

    if raw_ready:
        window_start, window_end = compute_window(
            buf.last_transcribed_offset_bytes,
            end_off,
            int(window_sec * bps),
        )
        rel_start = max(0, window_start - buf.buffer_start_offset_bytes)
        rel_end = max(rel_start, window_end - buf.buffer_start_offset_bytes)
        if rel_end > rel_start:
            pcm = bytes(buf.audio_buffer[rel_start:rel_end])
            samples = _pcm_s16le_bytes_to_float32(pcm)
            has_speech = _webrtcvad_window_has_speech(
                pcm_s16le=pcm,
                sample_rate_hz=sess.sample_rate_hz,
                aggressiveness=_env_int("VAD_ML_AGGRESSIVENESS", 2),
                frame_ms=_env_int("VAD_ML_FRAME_MS", 20),
                min_speech_ratio=_env_float("VAD_ML_MIN_SPEECH_RATIO", 0.12),
            ) and _rms_energy(samples) >= _env_float("VAD_RMS_THRESHOLD", 0.003)

            raw_segments: list[WhisperTranscriptSegment] = []
            if has_speech:
                raw_segments = await asyncio.to_thread(
                    state.backend.transcribe,
                    samples,
                    sess.sample_rate_hz,
                )

            pending = PendingTranscriptWindow(
                start_offset_bytes=window_start,
                end_offset_bytes=window_end,
                segments=tuple(raw_segments),
            )
            state.get_pending_transcripts(sess.session_id, stream_id).append(pending)
            buf.last_transcribed_offset_bytes = window_end

            async with state.lock:
                raw_protocol_segments = _raw_transcript_segments(
                    state,
                    sess,
                    stream_id,
                    raw_segments,
                    window_start_offset_bytes=window_start,
                    bytes_per_sec=bps,
                    buf=buf,
                )
                raw_sockets = set(state.session_sockets.get(sess.session_id, set()))
            await _send_transcript_segments(
                state,
                sess.session_id,
                raw_protocol_segments,
                raw_sockets,
                layer="raw",
            )

    processed = await _process_pending_transcripts(
        state,
        sess,
        stream_id,
        buf,
        window_sec,
        force=force,
    )
    while force and processed:
        processed = await _process_pending_transcripts(
            state,
            sess,
            stream_id,
            buf,
            window_sec,
            force=True,
        )


async def _process_pending_transcripts(
    state: ServerState,
    sess: SessionState,
    stream_id: StreamId,
    buf: StreamBuffer,
    window_sec: float,
    *,
    force: bool,
) -> bool:
    pending = state.get_pending_transcripts(sess.session_id, stream_id)
    if not pending:
        return False

    bps = bytes_per_second(sess.sample_rate_hz, sess.num_channels)
    end_off = buf.buffer_end_offset_bytes()
    diarization_enabled = state.diarizer is not None and should_diarize_stream(stream_id)
    diarization_lag_sec = (
        max(0.0, _env_float("DIARIZATION_LAG_SEC", 4.0))
        if diarization_enabled
        else 0.0
    )
    processed_lag_sec = max(0.0, _env_float("PROCESSED_TRANSCRIPT_LAG_SEC", 3.0))
    required_lag_sec = max(processed_lag_sec, diarization_lag_sec)
    idle_sec = (
        max(0.0, time.time() - buf.last_received_at)
        if buf.last_received_at is not None
        else 0.0
    )
    remaining_lag_sec = max(0.0, required_lag_sec - idle_sec)
    required_lag_bytes = int(remaining_lag_sec * bps)
    available_end_off = end_off if force else max(
        buf.buffer_start_offset_bytes,
        end_off - required_lag_bytes,
    )

    max_window_bytes = max(1, int(window_sec * bps))
    group: list[PendingTranscriptWindow] = []
    group_start = pending[0].start_offset_bytes
    for item in pending:
        if item.end_offset_bytes > available_end_off:
            break
        if group and item.end_offset_bytes - group_start > max_window_bytes:
            break
        group.append(item)
    if not group:
        return False
    has_more_ready = (
        len(group) < len(pending)
        and pending[len(group)].end_offset_bytes <= available_end_off
    )

    min_processed_sec = max(0.0, _env_float("PROCESSED_MIN_NEW_AUDIO_SEC", 2.0))
    if diarization_enabled:
        min_processed_sec = max(
            min_processed_sec,
            _env_float("DIARIZATION_MIN_NEW_AUDIO_SEC", 4.0),
        )
    group_end = group[-1].end_offset_bytes
    if (
        not force
        and group_end - group_start < int(min_processed_sec * bps)
        and idle_sec < required_lag_sec
        and not has_more_ready
    ):
        return False

    has_transcript = any(item.segments for item in group)
    diarization_turns: list[DiarizationTurn] = []
    diarization_completed = False
    diarization_embeddings: list[np.ndarray | None] = []
    diarization_samples = np.zeros(0, dtype=np.float32)
    diarization_base_offset_sec = float(group_start) / float(bps)
    diarization_duration_sec = 0.0

    if diarization_enabled and state.diarizer is not None and has_transcript:
        diarization_lag_bytes = int(diarization_lag_sec * bps)
        diarization_context_end = min(end_off, group_end + diarization_lag_bytes)
        diarization_window_bytes = int(
            max(window_sec, _env_float("DIARIZATION_WINDOW_SEC", 12.0)) * bps
        )
        diarization_start = max(
            buf.buffer_start_offset_bytes,
            diarization_context_end - diarization_window_bytes,
        )
        relative_start = max(0, diarization_start - buf.buffer_start_offset_bytes)
        relative_end = max(
            relative_start,
            diarization_context_end - buf.buffer_start_offset_bytes,
        )
        diarization_pcm = bytes(buf.audio_buffer[relative_start:relative_end])
        diarization_samples = _pcm_s16le_bytes_to_float32(diarization_pcm)
        diarization_base_offset_sec = float(diarization_start) / float(bps)
        diarization_duration_sec = float(diarization_samples.size) / float(sess.sample_rate_hz)
        try:
            diarization_turns = await asyncio.to_thread(
                state.diarizer.diarize,
                diarization_samples,
                sess.sample_rate_hz,
            )
            diarization_completed = True
            if state.identity_enabled and diarization_turns:
                diarization_embeddings = await asyncio.to_thread(
                    turn_embeddings,
                    diarization_turns,
                    diarization_samples,
                    sess.sample_rate_hz,
                    state.voice_embedder,
                )
        except Exception:
            logger.exception(
                "diarization failed for session=%s stream=%s",
                sess.session_id,
                stream_id,
            )

    protocol_segments: list[TranscriptSegment] = []
    async with state.lock:
        if diarization_completed:
            diarization_turns = state.get_diarization_mapper(sess.session_id, stream_id).map_turns(
                diarization_turns,
                base_offset_sec=diarization_base_offset_sec,
                window_duration_sec=diarization_duration_sec,
            )
            identity_registry: SessionSpeakerRegistry | None = None
            if state.identity_enabled:
                identity_registry = state.get_identity_registry(sess.session_id, stream_id)
                # map_turns preserves order, so the per-turn embeddings
                # computed on the backend's local turns align 1:1.
                diarization_turns = identity_registry.resolve_window(
                    diarization_turns, diarization_embeddings
                )
            _apply_diarization_turns(
                state,
                sess,
                stream_id,
                diarization_turns,
                diarization_base_offset_sec,
                buf,
                diarization_samples,
                sess.sample_rate_hz,
                replace_start_sec=diarization_base_offset_sec,
                replace_end_sec=diarization_base_offset_sec + diarization_duration_sec,
                identity_registry=identity_registry,
            )
        for item in group:
            if item.segments:
                protocol_segments.extend(
                    _assemble_transcript_segments(
                        state,
                        sess,
                        stream_id,
                        list(item.segments),
                        float(item.start_offset_bytes) / float(bps),
                        buf,
                    )
                )
        sockets = set(state.session_sockets.get(sess.session_id, set()))

    del pending[: len(group)]
    buf.last_processed_offset_bytes = max(buf.last_processed_offset_bytes, group_end)
    await _send_transcript_segments(
        state,
        sess.session_id,
        protocol_segments,
        sockets,
        layer="processed",
    )
    return True


async def _transcription_loop(state: ServerState) -> None:
    interval_sec = _env_float("TRANSCRIBE_INTERVAL_SEC", 1.0)
    min_new_audio_sec = _env_float("MIN_NEW_AUDIO_SEC", 2.0)
    window_sec = _env_float("WINDOW_SEC", 8.0)
    max_buffer_sec = _env_float("MAX_BUFFER_SEC", 600.0)  # 10 minutes

    while True:
        try:
            async with state.lock:
                sessions = list(state.sessions.values())

            for sess in sessions:
                # Transcribe mic and system streams separately
                for stream_id in ("mic", "system"):
                    buf = sess.get_buffer(cast(StreamId, stream_id))
                    # Only transcribe if there's any audio in this buffer
                    if len(buf.audio_buffer) > 0:
                        await _transcribe_stream(
                            state,
                            sess,
                            cast(StreamId, stream_id),
                            buf,
                            min_new_audio_sec,
                            window_sec,
                            max_buffer_sec,
                        )

            await _flush_stale_utterances(state)

        except Exception:
            logger.exception("transcription loop error (continuing)")

        await asyncio.sleep(interval_sec)


async def _flush_stale_utterances(state: ServerState) -> None:
    sends: list[tuple[str, list[TranscriptSegment], set[ServerConnection]]] = []
    async with state.lock:
        now_sec = time.time()
        for (session_id, stream_id), assembler in state.utterance_assemblers.items():
            session = state.sessions.get(session_id)
            if session is None:
                continue
            stream_now_sec = session.get_buffer(stream_id).timeline_now_sec(
                bytes_per_sec=session.bytes_per_second(),
                wall_now_sec=now_sec,
            )
            segments = [item.to_protocol() for item in assembler.flush_if_stale(stream_now_sec)]
            if segments:
                sends.append((session_id, segments, set(state.session_sockets.get(session_id, set()))))

    for session_id, segments, sockets in sends:
        await _send_transcript_segments(state, session_id, segments, sockets)


async def _drain_session_audio(state: ServerState, session: SessionState) -> None:
    min_new_audio_sec = _env_float("MIN_NEW_AUDIO_SEC", 2.0)
    window_sec = _env_float("WINDOW_SEC", 8.0)
    max_buffer_sec = _env_float("MAX_BUFFER_SEC", 600.0)
    for stream_id in (cast(StreamId, "mic"), cast(StreamId, "system")):
        buf = session.get_buffer(stream_id)
        while buf.buffer_end_offset_bytes() > buf.last_transcribed_offset_bytes:
            before = buf.last_transcribed_offset_bytes
            await _transcribe_stream(
                state,
                session,
                stream_id,
                buf,
                min_new_audio_sec,
                window_sec,
                max_buffer_sec,
                force=True,
            )
            if buf.last_transcribed_offset_bytes <= before:
                logger.warning(
                    "forced transcription made no progress for session=%s stream=%s",
                    session.session_id,
                    stream_id,
                )
                break
        await _transcribe_stream(
            state,
            session,
            stream_id,
            buf,
            min_new_audio_sec,
            window_sec,
            max_buffer_sec,
            force=True,
        )


async def _handle_control_message(
    ws: ServerConnection,
    state: ServerState,
    msg: Any,
) -> None:
    try:
        control = _validate_control_message(msg)
    except Exception as e:
        session_id = msg.get("session_id", "unknown") if isinstance(msg, dict) else "unknown"
        await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message=str(e))))
        return

    session_id = control["session_id"]
    command = control["command"]
    payload = control.get("payload", {})

    if command == "metadata":
        async with state.lock:
            state.session_sockets.setdefault(session_id, set()).add(ws)
            state.get_speaker_tracker(session_id).apply_metadata(payload)
        return

    if command == "speaker_activity":
        async with state.lock:
            state.session_sockets.setdefault(session_id, set()).add(ws)
            state.get_speaker_tracker(session_id).apply_activity(payload)
        return

    if command == "stop":
        async with state.lock:
            state.session_sockets.setdefault(session_id, set()).add(ws)
            session = state.sessions.get(session_id)
        if session is not None:
            await _drain_session_audio(state, session)
        async with state.lock:
            segments = [
                item.to_protocol()
                for assembler in state.get_session_assemblers(session_id)
                for item in assembler.flush(reason="stop")
            ]
            sockets = set(state.session_sockets.get(session_id, set()))
        await _send_transcript_segments(state, session_id, segments, sockets)
        profiles_changed = False
        if state.speaker_memory is not None:
            async with state.lock:
                for (registry_session_id, _), registry in state.identity_registries.items():
                    if registry_session_id != session_id:
                        continue
                    profiles_changed |= registry.export_updates(
                        state.speaker_memory,
                        session_id=session_id,
                        min_auto_profile_sec=_env_float("SPEAKER_MEMORY_MIN_AUTO_PROFILE_SEC", 10.0),
                        auto_profiles_enabled=_env_bool("SPEAKER_MEMORY_AUTO_PROFILES", True),
                    )
        if profiles_changed:
            # The client persists this update, so reinforced and newly learned
            # voices survive into the next session.
            await _send_speaker_memory_profiles(ws, state, session_id)
        await _send_speaker_memory_review(ws, state, session_id)
        await ws.send(
            json.dumps(
                {
                    "type": "control",
                    "session_id": session_id,
                    "command": "stopped",
                    "payload": {},
                }
            )
        )
        return

    if command == "speaker_memory_review":
        async with state.lock:
            state.session_sockets.setdefault(session_id, set()).add(ws)
        await _send_speaker_memory_review(ws, state, session_id)
        return

    if command == "speaker_memory_profiles":
        await _handle_speaker_memory_profiles(ws, state, session_id, payload)
        return

    if command == "speaker_memory_enroll":
        await _handle_speaker_memory_enroll(ws, state, session_id, payload)
        return

    if command == "ping":
        await ws.send(
            json.dumps(
                {
                    "type": "control",
                    "session_id": session_id,
                    "command": "pong",
                    "payload": {},
                }
            )
        )


async def _handle_client(ws: ServerConnection, state: ServerState) -> None:
    peer = getattr(ws, "remote_address", None)
    logger.info("client connected: %s", peer)

    last_log_at = 0.0

    try:
        async for raw in ws:
            if not isinstance(raw, str):
                # Protocol is JSON text frames only for now.
                await ws.send(json.dumps(_error(session_id="unknown", code="bad_request", message="binary frames not supported")))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await ws.send(json.dumps(_error(session_id="unknown", code="bad_json", message=str(e))))
                continue

            msg_type = msg.get("type") if isinstance(msg, dict) else None
            if msg_type == "audio_chunk":
                try:
                    audio = _validate_audio_chunk(msg)
                except Exception as e:
                    session_id = msg.get("session_id", "unknown") if isinstance(msg, dict) else "unknown"
                    await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message=str(e))))
                    continue

                session_id = audio["session_id"]
                stream_id = cast(StreamId, audio["stream_id"])
                seq = audio["seq"]
                audio_len = len(audio["audio_base64"])
                fmt = audio["audio_format"]

                # MVP constraints
                if fmt["encoding"] != "pcm_s16le":
                    await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="only pcm_s16le supported")))
                    continue
                if int(fmt["num_channels"]) != 1:
                    await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="only mono supported")))
                    continue
                if int(fmt["sample_rate_hz"]) <= 0:
                    await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="sample_rate_hz must be > 0")))
                    continue

                try:
                    pcm_bytes = _decode_audio_base64(audio["audio_base64"])
                except Exception as e:
                    await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message=str(e))))
                    continue

                now_wall = time.time()
                async with state.lock:
                    sess = state.sessions.get(session_id)
                    if sess is None:
                        sess = SessionState(
                            session_id=session_id,
                            created_at=now_wall,
                            updated_at=now_wall,
                            sample_rate_hz=int(fmt["sample_rate_hz"]),
                            num_channels=int(fmt["num_channels"]),
                        )
                        state.sessions[session_id] = sess
                        state.session_sockets.setdefault(session_id, set()).add(ws)
                    else:
                        # Lock format for the session for now.
                        if sess.sample_rate_hz != int(fmt["sample_rate_hz"]) or sess.num_channels != int(fmt["num_channels"]):
                            await ws.send(json.dumps(_error(session_id=session_id, code="bad_request", message="audio_format cannot change within a session")))
                            continue
                        sess.updated_at = now_wall
                        state.session_sockets.setdefault(session_id, set()).add(ws)

                    # Append to the correct stream buffer
                    buf = sess.get_buffer(stream_id)
                    seq_i = int(seq)
                    # If the client seq counter jumps backwards significantly, treat it
                    # as a new stream "run" under the same session_id and reset state.
                    # This avoids sticky `last_seq` causing transcript segment ID collisions.
                    if buf.last_seq >= 0 and (seq_i + 25) < buf.last_seq:
                        if stream_id == "mic":
                            sess.mic_buffer = StreamBuffer(stream_id="mic")
                        else:
                            sess.system_buffer = StreamBuffer(stream_id="system")
                        buf = sess.get_buffer(stream_id)
                    # Capture the client's epoch on the first chunk for this stream.
                    if buf.first_timestamp_ms is None:
                        buf.first_timestamp_ms = int(audio["timestamp_ms"])
                    buf.audio_buffer.extend(pcm_bytes)
                    buf.last_received_at = now_wall
                    buf.last_seq = max(buf.last_seq, seq_i)

                now = time.monotonic()
                if now - last_log_at >= 2.0:
                    logger.info("audio_chunk: session=%s stream=%s seq=%d audio_base64_len=%d", session_id, stream_id, seq, audio_len)
                    last_log_at = now

            elif msg_type == "control":
                await _handle_control_message(ws, state, msg)

            else:
                session_id = msg.get("session_id", "unknown") if isinstance(msg, dict) else "unknown"
                await ws.send(
                    json.dumps(
                        _error(
                            session_id=session_id,
                            code="unknown_type",
                            message=f"unknown message type: {msg_type!r}",
                        )
                    )
                )

    except websockets.ConnectionClosed:
        logger.info("client disconnected: %s", peer)
    except Exception:
        logger.exception("client handler crashed (continuing)")
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        async with state.lock:
            for sset in state.session_sockets.values():
                sset.discard(ws)


async def _run_server() -> None:
    host = "0.0.0.0"
    port = _env_int("WS_PORT", 8765)
    logger.info("starting websocket server on ws://%s:%d", host, port)

    state = ServerState()
    transcriber = asyncio.create_task(_transcription_loop(state))
    try:
        async with websockets.serve(lambda ws: _handle_client(ws, state), host, port, max_size=16 * 1024 * 1024):
            await asyncio.Future()  # run forever
    finally:
        transcriber.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await transcriber


def main() -> None:
    _load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run_server())


if __name__ == "__main__":
    main()
