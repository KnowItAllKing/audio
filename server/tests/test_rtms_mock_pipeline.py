from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

import numpy as np
import pytest
import websockets

from server.main import ServerState, _handle_client, _transcription_loop
from server.rtms_mock import (
    RTMS_AUDIO_SAMPLE_RATE_HZ,
    RTMS_EVENT_ACTIVE_SPEAKER_CHANGE,
    RTMS_EVENT_PARTICIPANT_JOINED,
    RTMS_MSG_EVENT_UPDATE,
    RTMS_MSG_MEDIA_DATA_AUDIO,
    RtmsToAudioServerAdapter,
    iter_rtms_messages,
    make_random_scenario,
)
from server.whisper_backend import TranscriptSegment


class ToneTranscriptBackend:
    def __init__(self, frequency_to_text: dict[float, str], *, tolerance_hz: float = 20.0) -> None:
        self.frequency_to_text = frequency_to_text
        self.tolerance_hz = tolerance_hz
        self.emitted: set[float] = set()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptSegment]:
        speech_bounds = speech_sample_bounds(samples)
        if speech_bounds is None:
            return []
        start_index, end_index = speech_bounds
        speech_samples = samples[start_index:end_index]

        frequency = dominant_frequency_hz(speech_samples, sample_rate)
        if frequency is None:
            return []

        nearest = min(self.frequency_to_text, key=lambda item: abs(item - frequency))
        if abs(nearest - frequency) > self.tolerance_hz:
            return []
        if nearest in self.emitted:
            return []

        self.emitted.add(nearest)
        return [
            TranscriptSegment(
                start_sec=float(start_index) / float(sample_rate),
                end_sec=float(end_index) / float(sample_rate),
                text=self.frequency_to_text[nearest],
                stream_tags=["system"],
            )
        ]


def speech_sample_bounds(samples: np.ndarray, *, threshold: float = 0.02) -> tuple[int, int] | None:
    above = np.where(np.abs(samples) >= threshold)[0]
    if above.size == 0:
        return None
    start = int(above[0])
    end = int(above[-1]) + 1
    return start, end


def dominant_frequency_hz(samples: np.ndarray, sample_rate: int) -> float | None:
    if samples.size < 32:
        return None
    samples = samples.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < 0.01:
        return None

    windowed = samples * np.hanning(samples.size).astype(np.float32)
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    if magnitudes.size <= 1:
        return None
    magnitudes[0] = 0.0
    peak_index = int(np.argmax(magnitudes))
    return float(np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)[peak_index])


def test_mock_rtms_messages_follow_expected_specs() -> None:
    scenario = make_random_scenario(seed=7, turn_count=2)
    messages = list(iter_rtms_messages(scenario))

    joined = messages[1]
    assert joined["msg_type"] == RTMS_MSG_EVENT_UPDATE
    assert joined["event"]["event_type"] == RTMS_EVENT_PARTICIPANT_JOINED
    assert joined["event"]["participants"][0]["user_id"] == scenario.participants[0].user_id
    assert joined["event"]["participants"][0]["user_name"] == scenario.participants[0].user_name

    active_speaker = next(
        item
        for item in messages
        if item.get("msg_type") == RTMS_MSG_EVENT_UPDATE
        and item.get("event", {}).get("event_type") == RTMS_EVENT_ACTIVE_SPEAKER_CHANGE
    )
    assert active_speaker["event"]["user_id"] == scenario.turns[0].participant.user_id
    assert active_speaker["event"]["user_name"] == scenario.turns[0].participant.user_name

    audio = next(item for item in messages if item.get("msg_type") == RTMS_MSG_MEDIA_DATA_AUDIO)
    assert audio["content"]["user_id"] == scenario.turns[0].participant.user_id
    assert isinstance(audio["content"]["data"], str)
    assert audio["content"]["length"] > 0
    assert isinstance(audio["content"]["timestamp"], int)


@pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets")
def test_mock_rtms_audio_to_transcript_pipeline() -> None:
    asyncio.run(run_mock_rtms_audio_to_transcript_pipeline())


async def run_mock_rtms_audio_to_transcript_pipeline() -> None:
    previous_env = _set_test_env()
    scenario = make_random_scenario(seed=11, turn_count=4)
    expected_texts = [text for _, text in scenario.expected_transcript]
    expected_labels_by_text = {text: label for label, text in scenario.expected_transcript}

    state = ServerState()
    state.backend = ToneTranscriptBackend({turn.frequency_hz: turn.text for turn in scenario.turns})
    transcriber = asyncio.create_task(_transcription_loop(state))

    try:
        async with websockets.serve(lambda ws: _handle_client(ws, state), "127.0.0.1", 0) as server:
            assert server.sockets
            port = int(server.sockets[0].getsockname()[1])
            uri = f"ws://127.0.0.1:{port}"

            async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
                adapter = RtmsToAudioServerAdapter(ws, session_id="mock-rtms-session")
                for message in iter_rtms_messages(scenario):
                    await adapter.handle(message)
                    if message.get("msg_type") == RTMS_MSG_MEDIA_DATA_AUDIO:
                        await asyncio.sleep(0.06)
                    else:
                        await asyncio.sleep(0)
                await adapter.stop()

                final_segments = await collect_final_segments(ws, expected_count=len(expected_texts))

        actual_texts = [item["text"] for item in final_segments]
        assert actual_texts == expected_texts
        for segment in final_segments:
            assert segment["stream_tags"] == ["system"]
            assert segment["speaker_label"] == expected_labels_by_text[segment["text"]]
            assert segment["speaker_source"] == "zoom"
            assert segment["is_final"] is True
    finally:
        transcriber.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await transcriber
        _restore_env(previous_env)


async def collect_final_segments(ws: Any, *, expected_count: int) -> list[dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    deadline = asyncio.get_running_loop().time() + 5.0
    while len(by_text) < expected_count:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        message = json.loads(raw)
        if message.get("type") != "transcript_update":
            continue
        for segment in message.get("segments", []):
            if segment.get("is_final"):
                by_text[segment["text"]] = segment
    return list(by_text.values())


def _set_test_env() -> dict[str, str | None]:
    updates = {
        "TRANSCRIBE_INTERVAL_SEC": "0.03",
        "MIN_NEW_AUDIO_SEC": "0.08",
        "WINDOW_SEC": "0.24",
        "MAX_BUFFER_SEC": "10",
        "UTTERANCE_PAUSE_SEC": "0.12",
        "UTTERANCE_MAX_SEC": "2",
        "VAD_ML_MIN_SPEECH_RATIO": "0",
        "VAD_RMS_THRESHOLD": "0",
        "SPEAKER_ACTIVITY_TTL_SEC": "2",
        "WHISPER_DISABLE": "1",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
