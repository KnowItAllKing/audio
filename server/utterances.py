from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .protocol_types import SpeakerSource, StreamId, TranscriptSegment
from .speakers import SpeakerIdentity


@dataclass(frozen=True)
class UtteranceAssemblerConfig:
    pause_sec: float = 1.2
    max_duration_sec: float = 14.0
    emit_partials: bool = True


@dataclass(frozen=True)
class AssembledSegment:
    id: str
    start_sec: float
    end_sec: float
    text: str
    stream_tags: list[StreamId]
    is_final: bool
    full_context_available: bool
    speaker_id: str
    speaker_label: str
    speaker_source: SpeakerSource
    speaker_confidence: float
    final_reason: Optional[str] = None

    def to_protocol(self) -> TranscriptSegment:
        out: TranscriptSegment = {
            "id": self.id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "stream_tags": self.stream_tags,
            "is_final": self.is_final,
            "full_context_available": self.full_context_available,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "speaker_source": self.speaker_source,
            "speaker_confidence": self.speaker_confidence,
        }
        if self.final_reason:
            out["final_reason"] = self.final_reason
        return out


@dataclass
class _OpenUtterance:
    id: str
    start_sec: float
    end_sec: float
    text: str
    stream_tags: set[StreamId] = field(default_factory=set)
    speaker: SpeakerIdentity = field(
        default_factory=lambda: SpeakerIdentity("unknown", "Unknown", "unknown", 0.0)
    )


class UtteranceAssembler:
    def __init__(self, *, session_id: str, config: UtteranceAssemblerConfig | None = None) -> None:
        self.session_id = session_id
        self.config = config or UtteranceAssemblerConfig()
        self._current: Optional[_OpenUtterance] = None
        self._counter = 0

    def push(
        self,
        *,
        stream_id: StreamId,
        start_sec: float,
        end_sec: float,
        text: str,
        speaker: SpeakerIdentity,
    ) -> list[AssembledSegment]:
        text = _normalize_text(text)
        if not text:
            return []

        if end_sec < start_sec:
            end_sec = start_sec

        out: list[AssembledSegment] = []
        if self._current and self._should_break(stream_id=stream_id, start_sec=start_sec, speaker=speaker):
            out.append(self._finish(_break_reason(self._current, stream_id=stream_id, start_sec=start_sec, speaker=speaker)))

        if self._current is None:
            self._current = self._new_current(stream_id=stream_id, start_sec=start_sec, end_sec=end_sec, text="", speaker=speaker)

        current = self._current
        current.text = _join_text(current.text, text)
        current.end_sec = max(current.end_sec, end_sec)
        current.stream_tags.add(stream_id)

        out.extend(self._finish_complete_sentences())

        if self._current and (self._current.end_sec - self._current.start_sec) >= self.config.max_duration_sec:
            out.append(self._finish("max_duration"))

        if self._current and self.config.emit_partials:
            out.append(self._snapshot(self._current, is_final=False, final_reason=None))

        return out

    def flush(self, *, reason: str = "flush") -> list[AssembledSegment]:
        if self._current is None:
            return []
        return [self._finish(reason)]

    def flush_if_stale(self, now_sec: float) -> list[AssembledSegment]:
        if self._current is None:
            return []
        if (now_sec - self._current.end_sec) < self.config.pause_sec:
            return []
        return [self._finish("pause")]

    def _new_current(
        self,
        *,
        stream_id: StreamId,
        start_sec: float,
        end_sec: float,
        text: str,
        speaker: SpeakerIdentity,
    ) -> _OpenUtterance:
        current = _OpenUtterance(
            id=f"utt-{self.session_id[:8]}-{self._counter:06d}",
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            stream_tags={stream_id},
            speaker=speaker,
        )
        self._counter += 1
        return current

    def _should_break(self, *, stream_id: StreamId, start_sec: float, speaker: SpeakerIdentity) -> bool:
        current = self._current
        if current is None:
            return False
        if current.speaker.speaker_id != speaker.speaker_id:
            return True
        if stream_id not in current.stream_tags:
            return True
        return (start_sec - current.end_sec) >= self.config.pause_sec

    def _finish_complete_sentences(self) -> list[AssembledSegment]:
        current = self._current
        if current is None:
            return []

        pieces, remainder, consumed_chars = _complete_sentence_pieces(current.text)
        if not pieces:
            return []

        total_chars = max(1, len(current.text))
        span_sec = max(0.0, current.end_sec - current.start_sec)
        base_start = current.start_sec
        original_end_sec = current.end_sec
        original_tags = set(current.stream_tags)
        original_speaker = current.speaker
        out: list[AssembledSegment] = []

        for index, (piece, end_idx) in enumerate(pieces):
            current = self._current
            if current is None:
                break
            estimated_end = base_start + span_sec * (end_idx / total_chars)
            current.text = piece
            current.end_sec = max(current.start_sec, estimated_end)
            out.append(self._snapshot(current, is_final=True, final_reason="punctuation"))
            if index < len(pieces) - 1:
                tags = set(current.stream_tags)
                current = self._new_current(
                    stream_id=next(iter(current.stream_tags)),
                    start_sec=current.end_sec,
                    end_sec=current.end_sec,
                    text="",
                    speaker=current.speaker,
                )
                current.stream_tags = tags
                self._current = current

        if remainder:
            last_end = base_start + span_sec * (consumed_chars / total_chars)
            self._current = self._new_current(
                stream_id=next(iter(original_tags)),
                start_sec=max(base_start, last_end),
                end_sec=original_end_sec,
                text=remainder,
                speaker=original_speaker,
            )
            self._current.stream_tags = original_tags
        else:
            self._current = None

        return out

    def _finish(self, reason: str) -> AssembledSegment:
        if self._current is None:
            raise RuntimeError("no open utterance")
        current = self._current
        self._current = None
        return self._snapshot(current, is_final=True, final_reason=reason)

    def _snapshot(
        self,
        current: _OpenUtterance,
        *,
        is_final: bool,
        final_reason: Optional[str],
    ) -> AssembledSegment:
        return AssembledSegment(
            id=current.id,
            start_sec=current.start_sec,
            end_sec=current.end_sec,
            text=current.text,
            stream_tags=_ordered_stream_tags(current.stream_tags),
            is_final=is_final,
            full_context_available=is_final,
            speaker_id=current.speaker.speaker_id,
            speaker_label=current.speaker.speaker_label,
            speaker_source=current.speaker.speaker_source,
            speaker_confidence=current.speaker.speaker_confidence,
            final_reason=final_reason,
        )


def _break_reason(current: _OpenUtterance, *, stream_id: StreamId, start_sec: float, speaker: SpeakerIdentity) -> str:
    if current.speaker.speaker_id != speaker.speaker_id:
        return "speaker_change"
    if stream_id not in current.stream_tags:
        return "stream_change"
    if start_sec - current.end_sec >= 0:
        return "pause"
    return "boundary"


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.!?:;)]}":
        return f"{left.rstrip()}{right}"
    return f"{left.rstrip()} {right.lstrip()}"


def _complete_sentence_pieces(text: str) -> tuple[list[tuple[str, int]], str, int]:
    pieces: list[tuple[str, int]] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in ".?!":
            if text[i] == "." and (
                (i > 0 and text[i - 1] == ".")
                or (i + 1 < len(text) and text[i + 1] == ".")
            ):
                i += 1
                continue
            end = i + 1
            while end < len(text) and text[end] in "\"')]}":
                end += 1
            if end == len(text) or text[end].isspace():
                sentence = text[start:end].strip()
                if sentence:
                    pieces.append((sentence, end))
                while end < len(text) and text[end].isspace():
                    end += 1
                start = end
                i = end
                continue
        i += 1
    remainder = text[start:].strip()
    consumed_chars = start
    return pieces, remainder, consumed_chars


def _ordered_stream_tags(tags: set[StreamId]) -> list[StreamId]:
    out: list[StreamId] = []
    if "mic" in tags:
        out.append("mic")
    if "system" in tags:
        out.append("system")
    return out
