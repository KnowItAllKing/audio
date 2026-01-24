from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


StreamId = Literal["mic", "system"]


class AudioFormat(TypedDict):
    encoding: Literal["pcm_s16le"]
    sample_rate_hz: int
    num_channels: int


class AudioChunkMessage(TypedDict):
    type: Literal["audio_chunk"]
    session_id: str
    stream_id: StreamId
    seq: int
    timestamp_ms: int | float
    audio_format: AudioFormat
    audio_base64: str


class ControlMessage(TypedDict):
    type: Literal["control"]
    session_id: str
    command: Literal["start", "stop", "ping", "pong", "metadata"]
    payload: NotRequired[dict]


class TranscriptSegment(TypedDict):
    id: str
    start_sec: float
    end_sec: float
    text: str
    stream_tags: list[StreamId]
    is_final: bool
    full_context_available: bool


class TranscriptUpdateMessage(TypedDict):
    type: Literal["transcript_update"]
    session_id: str
    segments: list[TranscriptSegment]


class ErrorMessage(TypedDict):
    type: Literal["error"]
    session_id: str
    code: str
    message: str


BaseMessage = AudioChunkMessage | ControlMessage | TranscriptUpdateMessage | ErrorMessage
