from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, cast

import websockets
from websockets.server import WebSocketServerProtocol

from .protocol_types import (
    AudioChunkMessage,
    ErrorMessage,
    StreamId,
    TranscriptUpdateMessage,
)


logger = logging.getLogger("server")


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


async def _handle_client(ws: WebSocketServerProtocol) -> None:
    peer = getattr(ws, "remote_address", None)
    logger.info("client connected: %s", peer)

    transcript_every_n = _env_int("TRANSCRIPT_EVERY_N", 10)
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

                now = time.monotonic()
                if now - last_log_at >= 2.0:
                    logger.info("audio_chunk: session=%s stream=%s seq=%d audio_base64_len=%d", session_id, stream_id, seq, audio_len)
                    last_log_at = now

                if transcript_every_n > 0 and (seq % transcript_every_n == 0):
                    # Fake timestamps: assume ~100ms chunks by default (client may vary).
                    end_sec = float(seq) * 0.1
                    start_sec = max(0.0, end_sec - 1.0)
                    update: TranscriptUpdateMessage = {
                        "type": "transcript_update",
                        "session_id": session_id,
                        "segments": [
                            {
                                "id": f"dummy-{stream_id}-{seq}",
                                "start_sec": start_sec,
                                "end_sec": end_sec,
                                "text": f"Dummy transcript up to seq {seq}",
                                "stream_tags": [stream_id],
                                "is_final": False,
                                "full_context_available": False,
                            }
                        ],
                    }
                    await ws.send(json.dumps(update))

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


async def _run_server() -> None:
    host = "0.0.0.0"
    port = _env_int("WS_PORT", 8765)
    logger.info("starting websocket server on ws://%s:%d", host, port)

    async with websockets.serve(_handle_client, host, port, max_size=16 * 1024 * 1024):
        await asyncio.Future()  # run forever


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

