from __future__ import annotations

from typing import Literal, TypedDict, Union
from typing_extensions import NotRequired


StreamId = Literal["mic", "system"]
TranscriptLayer = Literal["raw", "processed"]
SpeakerSource = Literal["stream", "zoom", "diarization", "memory", "manual", "unknown"]


class AudioFormat(TypedDict):
    encoding: Literal["pcm_s16le"]
    sample_rate_hz: int
    num_channels: int


class AudioChunkMessage(TypedDict):
    type: Literal["audio_chunk"]
    session_id: str
    stream_id: StreamId
    seq: int
    timestamp_ms: Union[int, float]
    audio_format: AudioFormat
    audio_base64: str


class ControlMessage(TypedDict):
    type: Literal["control"]
    session_id: str
    command: Literal[
        "start",
        "stop",
        "ping",
        "pong",
        "stopped",
        "metadata",
        "speaker_activity",
        "speaker_memory_review",
        "speaker_memory_enroll",
        "speaker_memory_profiles",
    ]
    payload: NotRequired[dict]


class TranscriptSegment(TypedDict):
    id: str
    start_sec: float
    end_sec: float
    text: str
    stream_tags: list[StreamId]
    speaker_id: str
    speaker_label: str
    speaker_source: SpeakerSource
    speaker_confidence: float
    is_final: bool
    full_context_available: bool
    final_reason: NotRequired[str]


class TranscriptUpdateMessage(TypedDict):
    type: Literal["transcript_update"]
    session_id: str
    layer: TranscriptLayer
    segments: list[TranscriptSegment]


class SpeakerMemoryReviewSample(TypedDict):
    sample_id: str
    speaker_id: str
    speaker_label: str
    stream_id: StreamId
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_rate_hz: int
    audio_wav_base64: str
    matched_profile_id: NotRequired[str | None]
    matched_name: NotRequired[str | None]
    match_confidence: NotRequired[float | None]


class SpeakerMemoryReviewMessage(TypedDict):
    type: Literal["speaker_memory_review"]
    session_id: str
    samples: list[SpeakerMemoryReviewSample]


class SpeakerMemoryProfile(TypedDict):
    profile_id: str
    name: str
    fingerprint: list[float]
    sample_count: int
    created_at: float
    updated_at: float


class SpeakerMemoryProfileMessage(TypedDict):
    type: Literal["speaker_memory_profile"]
    session_id: str
    profile: SpeakerMemoryProfile
    profiles: list[SpeakerMemoryProfile]


class SpeakerMemoryProfilesMessage(TypedDict):
    type: Literal["speaker_memory_profiles"]
    session_id: str
    profiles: list[SpeakerMemoryProfile]


class ErrorMessage(TypedDict):
    type: Literal["error"]
    session_id: str
    code: str
    message: str


BaseMessage = Union[
    AudioChunkMessage,
    ControlMessage,
    TranscriptUpdateMessage,
    SpeakerMemoryReviewMessage,
    SpeakerMemoryProfileMessage,
    SpeakerMemoryProfilesMessage,
    ErrorMessage,
]
