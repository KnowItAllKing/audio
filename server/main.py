from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from typing import Any, cast

import numpy as np
import webrtcvad
import websockets
from websockets.server import WebSocketServerProtocol

from .protocol_types import (
    AudioChunkMessage,
    ErrorMessage,
    StreamId,
    TranscriptUpdateMessage,
)
from .session_state import SessionState, StreamBuffer
from .transcription import bytes_per_second, compute_window, eligible_for_transcription
from .whisper_backend import WhisperBackend, create_backend


logger = logging.getLogger("server")


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

    vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))
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


class ServerState:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.session_sockets: dict[str, set[WebSocketServerProtocol]] = {}
        self.backend: WhisperBackend = create_backend()
        self.lock = asyncio.Lock()


def _build_transcript_update(
    session: SessionState,
    stream_id: StreamId,
    segments: list,
    base_offset_sec: float,
    buf: StreamBuffer,
) -> TranscriptUpdateMessage:
    # Convert client epoch (ms) to seconds; fall back to 0 if not set.
    epoch_sec = (buf.first_timestamp_ms / 1000.0) if buf.first_timestamp_ms else 0.0
    return {
        "type": "transcript_update",
        "session_id": session.session_id,
        "segments": [
            {
                # IMPORTANT: segment IDs must be stable identifiers for the client to
                # append/replace segments without overwriting the whole buffer.
                #
                # Using `buf.last_seq` here can cause ID collisions when a client restarts
                # its seq counter (common if the same session_id is reused). Since the
                # server currently does incremental non-overlapping windows, we can use
                # absolute timestamp ranges as a stable ID.
                "id": (
                    f"seg-{session.session_id[:8]}-{stream_id}-"
                    f"{int(round((epoch_sec + base_offset_sec + float(seg.start_sec)) * 1000.0))}-"
                    f"{int(round((epoch_sec + base_offset_sec + float(seg.end_sec)) * 1000.0))}-"
                    f"{i}"
                ),
                "start_sec": float(epoch_sec + base_offset_sec + float(seg.start_sec)),
                "end_sec": float(epoch_sec + base_offset_sec + float(seg.end_sec)),
                "text": seg.text,
                "stream_tags": [stream_id],
                "is_final": False,
                "full_context_available": False,
            }
            for i, seg in enumerate(segments)
        ],
    }


async def _transcribe_stream(
    state: ServerState,
    sess: SessionState,
    stream_id: StreamId,
    buf: StreamBuffer,
    min_new_audio_sec: float,
    window_sec: float,
    max_buffer_sec: float,
    socket_map: dict[str, set[WebSocketServerProtocol]],
) -> None:
    """Transcribe a single stream buffer and send updates."""
    bps = bytes_per_second(sess.sample_rate_hz, sess.num_channels)
    if bps <= 0:
        return

    buf.trim_to_max_bytes(int(max_buffer_sec * bps))

    end_off = buf.buffer_end_offset_bytes()
    min_new_bytes = int(min_new_audio_sec * bps)
    if not eligible_for_transcription(end_off, buf.last_transcribed_offset_bytes, min_new_bytes):
        return

    window_bytes = int(window_sec * bps)
    window_start, window_end = compute_window(buf.last_transcribed_offset_bytes, end_off, window_bytes)

    rel_start = window_start - buf.buffer_start_offset_bytes
    rel_end = window_end - buf.buffer_start_offset_bytes
    if rel_start < 0:
        rel_start = 0
    if rel_end <= rel_start:
        return

    pcm = bytes(buf.audio_buffer[rel_start:rel_end])
    samples = _pcm_s16le_bytes_to_float32(pcm)
    base_offset_sec = float(window_start) / float(bps)

    # Always-on VAD gate (Discord-style): WebRTC VAD first, RMS second.
    # We still advance last_transcribed_offset_bytes when skipping so we don't
    # keep reprocessing silence/noise.
    aggressiveness = _env_int("VAD_ML_AGGRESSIVENESS", 2)  # 0..3
    frame_ms = _env_int("VAD_ML_FRAME_MS", 20)  # 10/20/30
    min_ratio = _env_float("VAD_ML_MIN_SPEECH_RATIO", 0.12)
    if not _webrtcvad_window_has_speech(
        pcm_s16le=pcm,
        sample_rate_hz=sess.sample_rate_hz,
        aggressiveness=aggressiveness,
        frame_ms=frame_ms,
        min_speech_ratio=min_ratio,
    ):
        buf.last_transcribed_offset_bytes = window_end
        return

    thr = _env_float("VAD_RMS_THRESHOLD", 0.003)
    rms = _rms_energy(samples)
    if rms < thr:
        buf.last_transcribed_offset_bytes = window_end
        return

    # Avoid blocking the event loop.
    segments = await asyncio.to_thread(state.backend.transcribe, samples, sess.sample_rate_hz)

    sockets = socket_map.get(sess.session_id, set())
    if segments and sockets:
        update = _build_transcript_update(sess, stream_id, segments, base_offset_sec, buf)
        payload = json.dumps(update)
        dead: list[WebSocketServerProtocol] = []
        for ws in sockets:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with state.lock:
                sset = state.session_sockets.get(sess.session_id)
                if sset:
                    for ws in dead:
                        sset.discard(ws)

    # Commit everything up to window_end.
    buf.last_transcribed_offset_bytes = window_end


async def _transcription_loop(state: ServerState) -> None:
    interval_sec = _env_float("TRANSCRIBE_INTERVAL_SEC", 1.0)
    min_new_audio_sec = _env_float("MIN_NEW_AUDIO_SEC", 2.0)
    window_sec = _env_float("WINDOW_SEC", 8.0)
    max_buffer_sec = _env_float("MAX_BUFFER_SEC", 600.0)  # 10 minutes

    while True:
        try:
            async with state.lock:
                sessions = list(state.sessions.values())
                socket_map = {k: set(v) for k, v in state.session_sockets.items()}

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
                            socket_map,
                        )

        except Exception:
            logger.exception("transcription loop error (continuing)")

        await asyncio.sleep(interval_sec)


async def _handle_client(ws: WebSocketServerProtocol, state: ServerState) -> None:
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
                    buf.last_seq = max(buf.last_seq, seq_i)

                now = time.monotonic()
                if now - last_log_at >= 2.0:
                    logger.info("audio_chunk: session=%s stream=%s seq=%d audio_base64_len=%d", session_id, stream_id, seq, audio_len)
                    last_log_at = now

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
        with contextlib.suppress(Exception):
            await transcriber


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
