from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, cast

from .protocol_types import SpeakerSource, StreamId


@dataclass(frozen=True)
class SpeakerIdentity:
    speaker_id: str
    speaker_label: str
    speaker_source: SpeakerSource
    speaker_confidence: float


@dataclass(frozen=True)
class SpeakerActivity:
    participant_id: str
    participant_label: str
    stream_id: StreamId
    start_sec: float
    end_sec: Optional[float]
    speaker_source: SpeakerSource
    speaker_confidence: float


class SpeakerTracker:
    def __init__(
        self,
        *,
        local_label: str = "You",
        system_label: str = "System audio",
        active_speaker_ttl_sec: float = 15.0,
        activity_boundary_tolerance_sec: float = 0.35,
    ) -> None:
        self.active_speaker_ttl_sec = active_speaker_ttl_sec
        self.activity_boundary_tolerance_sec = max(0.0, activity_boundary_tolerance_sec)
        self.participants: dict[str, str] = {}
        self.stream_speakers: dict[StreamId, SpeakerIdentity] = {
            "mic": SpeakerIdentity("local:mic", local_label, "stream", 0.75),
            "system": SpeakerIdentity("system:unknown", system_label, "stream", 0.35),
        }
        self.activities: list[SpeakerActivity] = []

    def apply_metadata(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        participants = payload.get("participants")
        if isinstance(participants, list):
            for item in participants:
                if not isinstance(item, dict):
                    continue
                participant_id = _optional_str(item.get("id") or item.get("participant_id") or item.get("participantId"))
                name = _optional_str(item.get("name") or item.get("display_name") or item.get("displayName"))
                if participant_id and name:
                    self.participants[participant_id] = name

        speaker_labels = payload.get("speaker_labels")
        if isinstance(speaker_labels, dict):
            for stream_id in ("mic", "system"):
                label = _optional_str(speaker_labels.get(stream_id))
                if label:
                    self.set_stream_speaker(cast(StreamId, stream_id), label=label, source="manual")

        stream_speakers = payload.get("stream_speakers")
        if isinstance(stream_speakers, dict):
            for stream_id in ("mic", "system"):
                item = stream_speakers.get(stream_id)
                if isinstance(item, str):
                    self.set_stream_speaker(cast(StreamId, stream_id), label=item, source="manual")
                elif isinstance(item, dict):
                    label = _optional_str(
                        item.get("speaker_label")
                        or item.get("label")
                        or item.get("name")
                    )
                    speaker_id = _optional_str(item.get("speaker_id") or item.get("id"))
                    source = _speaker_source(item.get("speaker_source") or item.get("source"), default="manual")
                    confidence_raw = item["speaker_confidence"] if "speaker_confidence" in item else item.get("confidence")
                    confidence = _optional_float(confidence_raw)
                    if label:
                        self.set_stream_speaker(
                            cast(StreamId, stream_id),
                            label=label,
                            speaker_id=speaker_id,
                            source=source,
                            confidence=confidence,
                        )

    def set_stream_speaker(
        self,
        stream_id: StreamId,
        *,
        label: str,
        speaker_id: Optional[str] = None,
        source: SpeakerSource = "manual",
        confidence: Optional[float] = None,
    ) -> None:
        if not speaker_id:
            speaker_id = "local:mic" if stream_id == "mic" else "system:unknown"
        if confidence is None:
            confidence = 0.85 if source == "manual" else 0.75
        self.stream_speakers[stream_id] = SpeakerIdentity(
            speaker_id=speaker_id,
            speaker_label=label,
            speaker_source=source,
            speaker_confidence=_clamp_confidence(confidence),
        )

    def apply_activity(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        stream_id = payload.get("stream_id") or payload.get("streamId") or "system"
        if stream_id not in ("mic", "system"):
            return

        participant_id = _optional_str(
            payload.get("participant_id")
            or payload.get("participantId")
            or payload.get("speaker_id")
            or payload.get("speakerId")
            or payload.get("id")
        )
        if not participant_id:
            return

        label = _optional_str(
            payload.get("participant_name")
            or payload.get("participantName")
            or payload.get("speaker_label")
            or payload.get("speakerLabel")
            or payload.get("name")
        )
        if not label:
            label = self.participants.get(participant_id, participant_id)

        start_sec = _read_time_sec(payload, "start")
        if start_sec is None:
            return
        end_sec = _read_time_sec(payload, "end")
        if end_sec is not None and end_sec < start_sec:
            end_sec = start_sec
        source = _speaker_source(payload.get("speaker_source") or payload.get("source"), default="zoom")
        confidence = _optional_float(payload.get("speaker_confidence") or payload.get("confidence"))

        self.activities.append(
            SpeakerActivity(
                participant_id=participant_id,
                participant_label=label,
                stream_id=cast(StreamId, stream_id),
                start_sec=start_sec,
                end_sec=end_sec,
                speaker_source=source,
                speaker_confidence=_clamp_confidence(confidence if confidence is not None else 0.95),
            )
        )
        if len(self.activities) > 500:
            self.activities = self.activities[-500:]

    def remove_activities_in_range(
        self,
        *,
        stream_id: StreamId,
        start_sec: float,
        end_sec: float,
        sources: set[SpeakerSource],
    ) -> None:
        """Remove replaceable inferred turns while preserving manual/Zoom metadata.

        Activities that straddle a boundary of the range are trimmed, not
        dropped: the replacing window only re-covers its own span, so deleting
        a turn that pokes past the boundary would lose the outside part for
        good (the next windows start after it and never restore it)."""
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        kept: list[SpeakerActivity] = []
        for activity in self.activities:
            activity_end = activity.end_sec if activity.end_sec is not None else float("inf")
            overlaps = activity.start_sec < end_sec and activity_end > start_sec
            if activity.stream_id != stream_id or activity.speaker_source not in sources or not overlaps:
                kept.append(activity)
                continue
            if activity.start_sec < start_sec:
                kept.append(replace(activity, end_sec=start_sec))
            if activity_end > end_sec and activity.end_sec is not None:
                kept.append(replace(activity, start_sec=end_sec))
        self.activities = kept

    def resolve(self, *, stream_id: StreamId, start_sec: float, end_sec: float) -> SpeakerIdentity:
        activity = self._activity_for(stream_id=stream_id, start_sec=start_sec, end_sec=end_sec)
        if activity:
            speaker_id = activity.participant_id
            if ":" not in speaker_id:
                speaker_id = f"{activity.speaker_source}:{speaker_id}"
            return SpeakerIdentity(
                speaker_id=speaker_id,
                speaker_label=activity.participant_label,
                speaker_source=activity.speaker_source,
                speaker_confidence=activity.speaker_confidence,
            )
        return self.stream_speakers[stream_id]

    def _activity_for(self, *, stream_id: StreamId, start_sec: float, end_sec: float) -> Optional[SpeakerActivity]:
        # Streams without diarization/Zoom/manual activity simply have no
        # activities recorded and fall back to the stream default (mic → You).
        # Gating on the stream name here would silently break
        # DIARIZATION_STREAMS=mic (a laptop mic hearing a whole room).
        mid_sec = start_sec + max(0.0, end_sec - start_sec) / 2.0
        exact_matches: list[SpeakerActivity] = []
        boundary_matches: list[SpeakerActivity] = []
        for activity in self.activities:
            if activity.stream_id != stream_id:
                continue
            if activity.start_sec - self.activity_boundary_tolerance_sec > mid_sec:
                continue
            if activity.end_sec is None:
                ttl_sec = 3600.0 if activity.speaker_source == "manual" else self.active_speaker_ttl_sec
                if mid_sec - activity.start_sec > ttl_sec + self.activity_boundary_tolerance_sec:
                    continue
                is_exact = activity.start_sec <= mid_sec and mid_sec - activity.start_sec <= ttl_sec
            elif activity.end_sec + self.activity_boundary_tolerance_sec < mid_sec:
                continue
            else:
                is_exact = activity.start_sec <= mid_sec <= activity.end_sec
            (exact_matches if is_exact else boundary_matches).append(activity)
        matches = exact_matches or boundary_matches
        if not matches:
            return None
        return max(matches, key=lambda item: (_activity_priority(item), item.start_sec))


def _activity_priority(activity: SpeakerActivity) -> int:
    if activity.speaker_source in ("manual", "zoom"):
        return 3
    if activity.speaker_source == "memory":
        return 2
    if activity.speaker_source == "diarization":
        return 1
    return 1


def _read_time_sec(payload: dict[str, Any], prefix: Literal["start", "end"]) -> Optional[float]:
    sec_keys = [f"{prefix}_sec", f"{prefix}Sec"]
    ms_keys = [f"{prefix}_timestamp_ms", f"{prefix}TimestampMs", f"{prefix}Time", f"{prefix}_time"]
    for key in sec_keys:
        value = _optional_float(payload.get(key))
        if value is not None:
            return value
    for key in ms_keys:
        value = _optional_float(payload.get(key))
        if value is not None:
            return value / 1000.0
    return None


def _optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
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


def _speaker_source(value: Any, *, default: SpeakerSource) -> SpeakerSource:
    if value in ("stream", "zoom", "diarization", "memory", "manual", "unknown"):
        return cast(SpeakerSource, value)
    return default


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
