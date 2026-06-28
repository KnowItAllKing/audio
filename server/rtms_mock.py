from __future__ import annotations

import base64
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol


RTMS_MSG_EVENT_UPDATE = 6
RTMS_MSG_CLIENT_READY_ACK = 7
RTMS_MSG_KEEP_ALIVE_REQ = 12
RTMS_MSG_KEEP_ALIVE_RESP = 13
RTMS_MSG_MEDIA_DATA_AUDIO = 14
RTMS_MSG_MEDIA_DATA_TRANSCRIPT = 17

RTMS_EVENT_FIRST_PACKET_TIMESTAMP = 1
RTMS_EVENT_ACTIVE_SPEAKER_CHANGE = 2
RTMS_EVENT_PARTICIPANT_JOINED = 3

RTMS_AUDIO_SAMPLE_RATE_HZ = 16_000
RTMS_AUDIO_CHANNELS = 1
RTMS_AUDIO_BYTES_PER_SAMPLE = 2


class JsonWebSocket(Protocol):
    async def send(self, message: str) -> None:
        ...


@dataclass(frozen=True)
class MockRtmsParticipant:
    user_id: int
    user_name: str


@dataclass(frozen=True)
class MockRtmsTurn:
    participant: MockRtmsParticipant
    text: str
    frequency_hz: float
    duration_ms: int
    pause_after_ms: int


@dataclass(frozen=True)
class MockRtmsScenario:
    meeting_uuid: str
    rtms_stream_id: str
    started_at_ms: int
    send_rate_ms: int
    participants: list[MockRtmsParticipant]
    turns: list[MockRtmsTurn]

    @property
    def expected_transcript(self) -> list[tuple[str, str]]:
        return [(turn.participant.user_name, turn.text) for turn in self.turns]


def make_random_scenario(*, seed: int, turn_count: int = 4) -> MockRtmsScenario:
    rng = random.Random(seed)
    participants = [
        MockRtmsParticipant(16778240, "John Smith"),
        MockRtmsParticipant(19778240, "Alice Lee"),
        MockRtmsParticipant(22334455, "Taylor Kim"),
    ]
    phrases = [
        "Project status is green.",
        "I found one blocker.",
        "Can we ship this tomorrow?",
        "The audio path looks stable.",
        "Please send the notes.",
        "That sounds good to me.",
    ]

    turns: list[MockRtmsTurn] = []
    used_phrases = rng.sample(phrases, k=min(turn_count, len(phrases)))
    for index, phrase in enumerate(used_phrases):
        turns.append(
            MockRtmsTurn(
                participant=rng.choice(participants),
                text=phrase,
                frequency_hz=420.0 + float(index * 90),
                duration_ms=rng.choice([420, 520, 620]),
                pause_after_ms=rng.choice([260, 340, 420]),
            )
        )

    return MockRtmsScenario(
        meeting_uuid="mock-meeting-uuid",
        rtms_stream_id="mock-stream-id",
        started_at_ms=1_737_384_349_000,
        send_rate_ms=100,
        participants=participants,
        turns=turns,
    )


def iter_rtms_messages(scenario: MockRtmsScenario) -> Iterable[dict[str, Any]]:
    yield {
        "msg_type": RTMS_MSG_EVENT_UPDATE,
        "event": {
            "event_type": RTMS_EVENT_FIRST_PACKET_TIMESTAMP,
            "timestamp": scenario.started_at_ms,
        },
    }
    yield {
        "msg_type": RTMS_MSG_EVENT_UPDATE,
        "event": {
            "event_type": RTMS_EVENT_PARTICIPANT_JOINED,
            "timestamp": scenario.started_at_ms,
            "participants": [
                {"user_id": item.user_id, "user_name": item.user_name}
                for item in scenario.participants
            ],
        },
    }

    timestamp_ms = scenario.started_at_ms
    for turn in scenario.turns:
        yield {
            "msg_type": RTMS_MSG_EVENT_UPDATE,
            "event": {
                "event_type": RTMS_EVENT_ACTIVE_SPEAKER_CHANGE,
                "timestamp": timestamp_ms,
                "user_id": turn.participant.user_id,
                "user_name": turn.participant.user_name,
            },
        }

        for packet in iter_audio_packets(turn, timestamp_ms, scenario.send_rate_ms):
            yield packet

        timestamp_ms += turn.duration_ms + turn.pause_after_ms

    yield {
        "msg_type": RTMS_MSG_MEDIA_DATA_TRANSCRIPT,
        "content": {
            "user_id": scenario.turns[-1].participant.user_id if scenario.turns else 0,
            "user_name": scenario.turns[-1].participant.user_name if scenario.turns else "",
            "start_time": scenario.started_at_ms,
            "end_time": timestamp_ms,
            "timestamp": timestamp_ms,
            "language": 9,
            "data": " ".join(turn.text for turn in scenario.turns),
        },
    }


def iter_audio_packets(
    turn: MockRtmsTurn,
    start_timestamp_ms: int,
    send_rate_ms: int,
) -> Iterable[dict[str, Any]]:
    remaining_ms = turn.duration_ms
    timestamp_ms = start_timestamp_ms
    phase = 0.0
    while remaining_ms > 0:
        packet_ms = min(send_rate_ms, remaining_ms)
        pcm, phase = generate_tone_pcm_s16le(
            frequency_hz=turn.frequency_hz,
            duration_ms=packet_ms,
            phase=phase,
        )
        yield {
            "msg_type": RTMS_MSG_MEDIA_DATA_AUDIO,
            "content": {
                "user_id": turn.participant.user_id,
                "user_name": turn.participant.user_name,
                "data": base64.b64encode(pcm).decode("ascii"),
                "length": len(pcm),
                "timestamp": timestamp_ms,
            },
        }
        remaining_ms -= packet_ms
        timestamp_ms += packet_ms


def generate_tone_pcm_s16le(
    *,
    frequency_hz: float,
    duration_ms: int,
    sample_rate_hz: int = RTMS_AUDIO_SAMPLE_RATE_HZ,
    amplitude: float = 0.32,
    phase: float = 0.0,
) -> tuple[bytes, float]:
    sample_count = int(round(sample_rate_hz * duration_ms / 1000.0))
    out = bytearray()
    step = 2.0 * math.pi * frequency_hz / sample_rate_hz
    current = phase
    for _ in range(sample_count):
        value = int(max(-1.0, min(1.0, math.sin(current) * amplitude)) * 32767.0)
        out.extend(value.to_bytes(2, byteorder="little", signed=True))
        current += step
        if current >= 2.0 * math.pi:
            current -= 2.0 * math.pi
    return bytes(out), current


class RtmsToAudioServerAdapter:
    def __init__(self, ws: JsonWebSocket, *, session_id: str, stream_id: str = "system") -> None:
        self.ws = ws
        self.session_id = session_id
        self.stream_id = stream_id
        self.seq = 0
        self.last_audio_end_timestamp_ms: Optional[int] = None

    async def handle(self, message: dict[str, Any]) -> None:
        msg_type = message.get("msg_type")
        if msg_type == RTMS_MSG_EVENT_UPDATE:
            await self._handle_event(message.get("event"))
        elif msg_type == RTMS_MSG_MEDIA_DATA_AUDIO:
            await self._handle_audio(message.get("content"))
        elif msg_type == RTMS_MSG_KEEP_ALIVE_REQ:
            await self.ws.send(
                _json_dumps(
                    {
                        "msg_type": RTMS_MSG_KEEP_ALIVE_RESP,
                        "timestamp": message.get("timestamp"),
                    }
                )
            )

    async def stop(self) -> None:
        await self.ws.send(
            _json_dumps(
                {
                    "type": "control",
                    "session_id": self.session_id,
                    "command": "stop",
                    "payload": {},
                }
            )
        )

    async def _handle_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("event_type")
        if event_type == RTMS_EVENT_PARTICIPANT_JOINED:
            await self.ws.send(
                _json_dumps(
                    {
                        "type": "control",
                        "session_id": self.session_id,
                        "command": "metadata",
                        "payload": {
                            "participants": [
                                {
                                    "id": str(item.get("user_id")),
                                    "name": item.get("user_name"),
                                }
                                for item in event.get("participants", [])
                                if isinstance(item, dict)
                            ],
                        },
                    }
                )
            )
        elif event_type == RTMS_EVENT_ACTIVE_SPEAKER_CHANGE:
            await self._send_speaker_activity(
                user_id=event.get("user_id"),
                user_name=event.get("user_name"),
                start_timestamp_ms=event.get("timestamp"),
            )

    async def _handle_audio(self, content: Any) -> None:
        if not isinstance(content, dict):
            return

        data = content.get("data")
        timestamp = content.get("timestamp")
        if not isinstance(data, str) or not isinstance(timestamp, int):
            return

        try:
            pcm = base64.b64decode(data, validate=True)
        except Exception:
            return

        duration_ms = int(round((len(pcm) / (RTMS_AUDIO_SAMPLE_RATE_HZ * RTMS_AUDIO_BYTES_PER_SAMPLE)) * 1000.0))
        if self.last_audio_end_timestamp_ms is not None and timestamp > self.last_audio_end_timestamp_ms:
            await self._send_audio_chunk(
                timestamp_ms=self.last_audio_end_timestamp_ms,
                pcm=bytes((timestamp - self.last_audio_end_timestamp_ms) * RTMS_AUDIO_BYTES_PER_SAMPLE * RTMS_AUDIO_SAMPLE_RATE_HZ // 1000),
            )

        await self._send_speaker_activity(
            user_id=content.get("user_id"),
            user_name=content.get("user_name"),
            start_timestamp_ms=timestamp,
            end_timestamp_ms=timestamp + duration_ms,
        )
        await self._send_audio_chunk(timestamp_ms=timestamp, pcm=pcm, audio_base64=data)
        self.last_audio_end_timestamp_ms = timestamp + duration_ms

    async def _send_audio_chunk(
        self,
        *,
        timestamp_ms: int,
        pcm: bytes,
        audio_base64: Optional[str] = None,
    ) -> None:
        await self.ws.send(
            _json_dumps(
                {
                    "type": "audio_chunk",
                    "session_id": self.session_id,
                    "stream_id": self.stream_id,
                    "seq": self.seq,
                    "timestamp_ms": timestamp_ms,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": RTMS_AUDIO_SAMPLE_RATE_HZ,
                        "num_channels": RTMS_AUDIO_CHANNELS,
                    },
                    "audio_base64": audio_base64 or base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        self.seq += 1

    async def _send_speaker_activity(
        self,
        *,
        user_id: Any,
        user_name: Any,
        start_timestamp_ms: Any,
        end_timestamp_ms: Optional[int] = None,
    ) -> None:
        if user_id is None or not isinstance(start_timestamp_ms, int):
            return
        payload: dict[str, Any] = {
            "participant_id": str(user_id),
            "participant_name": user_name,
            "stream_id": self.stream_id,
            "start_timestamp_ms": start_timestamp_ms,
        }
        if end_timestamp_ms is not None:
            payload["end_timestamp_ms"] = end_timestamp_ms
        await self.ws.send(
            _json_dumps(
                {
                    "type": "control",
                    "session_id": self.session_id,
                    "command": "speaker_activity",
                    "payload": payload,
                }
            )
        )


def _json_dumps(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"))
