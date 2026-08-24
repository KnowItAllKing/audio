"""Session-level speaker identity on top of windowed diarization.

The DiarizationSpeakerMapper keeps ids stable only through time overlap
between consecutive windows — continuity, not identity. A speaker who is
silent past the overlap horizon comes back as a brand-new "Speaker N", and a
window whose internal clustering merges two similar voices drags one
speaker's turns under the other's id.

This registry adds identity at the right granularity: sherpa's turn
segmentation is reliable while its within-window clustering is not, so every
individual diarized turn with enough audio is voice-embedded and matched
against the session's known voices (and the voice profiles the client sent at
session start). A returning voice reuses its session id no matter how long it
was silent; a voice matching a stored profile is labeled with the profile's
name immediately; short turns fall back to window-cluster continuity.

Reinforcement: confidently-assigned turn embeddings are folded into the
entry's exemplar bank, so one person may accumulate several distinct
fingerprints (different mics, rooms, days). At session stop the accumulated
exemplars are folded back into the client-persisted profiles — matched
profiles learn the new rendition of the voice, and unnamed session speakers
can be exported as auto profiles so the next session recognizes them.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .diarization import DiarizationTurn
from .speaker_memory import (
    SpeakerMemory,
    SpeakerProfile,
    bank_similarity,
    merge_into_bank,
)

_LABEL_NUMBER = re.compile(r"^Speaker (\d+)$")


@dataclass
class IdentityEntry:
    speaker_id: str
    label: str
    source: str  # "diarization" | "memory"
    profile_id: Optional[str]
    exemplars: list[np.ndarray] = field(default_factory=list)
    session_exemplars: list[np.ndarray] = field(default_factory=list)
    duration_sec: float = 0.0
    last_confidence: float = 0.0


class SessionSpeakerRegistry:
    """Per-(session, stream) voice identity registry.

    `resolve_window` receives the mapper's provisionally-mapped turns plus one
    voice embedding per turn (None where the turn is too short to embed) and
    rewrites ids/labels to stable identities.
    """

    def __init__(
        self,
        *,
        match_threshold: float = 0.68,
        match_margin: float = 0.05,
        reinforce_floor: float = 0.55,
        merge_similarity: float = 0.90,
        max_exemplars: int = 8,
        min_new_speaker_sec: float = 2.0,
        min_reinforce_sec: float = 2.0,
    ) -> None:
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.reinforce_floor = reinforce_floor
        self.merge_similarity = merge_similarity
        self.max_exemplars = max_exemplars
        self.min_new_speaker_sec = min_new_speaker_sec
        self.min_reinforce_sec = min_reinforce_sec
        self._entries: dict[str, IdentityEntry] = {}
        self._aliases: dict[str, str] = {}
        self._seeded_profile_ids: set[str] = set()

    def seed_profiles(self, profiles: list[SpeakerProfile]) -> None:
        for profile in profiles:
            if profile.profile_id in self._seeded_profile_ids:
                continue
            self._seeded_profile_ids.add(profile.profile_id)
            speaker_id = f"memory:{profile.profile_id}"
            self._entries[speaker_id] = IdentityEntry(
                speaker_id=speaker_id,
                label=self._session_label(profile.name),
                source="memory",
                profile_id=profile.profile_id,
                exemplars=[exemplar.astype(np.float32) for exemplar in profile.bank()],
            )

    def resolve_window(
        self,
        turns: list[DiarizationTurn],
        embeddings: list[Optional[np.ndarray]],
    ) -> list[DiarizationTurn]:
        """Rewrite the mapper's provisional ids to stable voice identities.

        Identity is decided per turn: within-window clustering may merge two
        similar voices into one cluster, but individual turns are single-voice,
        so each turn with enough audio is matched on its own embedding. Turns
        too short to embed follow their window cluster's duration-weighted
        majority, then continuity, then a new speaker.
        """
        if len(embeddings) < len(turns):
            embeddings = list(embeddings) + [None] * (len(turns) - len(embeddings))

        assigned: list[Optional[IdentityEntry]] = [None] * len(turns)
        for index, turn in enumerate(turns):
            embedding = embeddings[index]
            if embedding is None:
                continue
            duration = max(0.0, turn.end_sec - turn.start_sec)
            matched, best, second = self._best_match(embedding)
            if matched is not None and best >= self.match_threshold and (best - second) >= self.match_margin:
                assigned[index] = matched
                matched.last_confidence = best
                self._note(matched, embedding, duration)
            elif (matched is None or best < self.match_threshold) and duration >= self.min_new_speaker_sec:
                entry = self._create_entry()
                assigned[index] = entry
                self._note(entry, embedding, duration)
            # else: ambiguous (close seconds, or confident-but-short) — leave
            # for the fallback pass so continuity can decide.

        # Duration-weighted majority entry per provisional cluster id.
        majority: dict[str, IdentityEntry] = {}
        weight: dict[str, dict[str, float]] = {}
        for index, turn in enumerate(turns):
            entry = assigned[index]
            if entry is None:
                continue
            per_entry = weight.setdefault(turn.speaker_id, {})
            per_entry[entry.speaker_id] = per_entry.get(entry.speaker_id, 0.0) + max(
                0.0, turn.end_sec - turn.start_sec
            )
        for provisional_id, weights in weight.items():
            majority[provisional_id] = self._entries[max(weights, key=lambda key: weights[key])]

        for index, turn in enumerate(turns):
            if assigned[index] is not None:
                continue
            entry = majority.get(turn.speaker_id)
            if entry is None:
                known_id = self._aliases.get(turn.speaker_id)
                entry = self._entries.get(known_id) if known_id is not None else None
            if entry is None:
                entry = self._create_entry()
            assigned[index] = entry
            entry.duration_sec += max(0.0, turn.end_sec - turn.start_sec)

        # Continuity for the next window's short turns: remember where this
        # window's clusters predominantly landed.
        for provisional_id, entry in majority.items():
            self._aliases[provisional_id] = entry.speaker_id

        out: list[DiarizationTurn] = []
        for index, turn in enumerate(turns):
            entry = assigned[index]
            assert entry is not None
            confidence = (
                entry.last_confidence
                if entry.source == "memory" and entry.last_confidence > 0.0
                else turn.speaker_confidence
            )
            out.append(
                DiarizationTurn(
                    start_sec=turn.start_sec,
                    end_sec=turn.end_sec,
                    speaker_id=entry.speaker_id,
                    speaker_label=entry.label,
                    speaker_confidence=confidence,
                )
            )
        return out

    def match_info(self, speaker_id: str) -> Optional[tuple[str, str, float]]:
        entry = self._entries.get(speaker_id)
        if entry is None or entry.source != "memory" or entry.profile_id is None:
            return None
        return (entry.profile_id, entry.label, entry.last_confidence)

    def session_entries(self) -> list[IdentityEntry]:
        return list(self._entries.values())

    def export_updates(
        self,
        memory: SpeakerMemory,
        *,
        session_id: str,
        min_auto_profile_sec: float = 10.0,
        auto_profiles_enabled: bool = True,
    ) -> bool:
        """Fold this session's voice evidence back into the profile store.

        Matched profiles are reinforced with the session's exemplars; unnamed
        session speakers with enough speech become auto profiles ("worst case
        one person has several diarized voices" — until the user names them,
        at which point enrolling merges banks under the named profile).
        """
        changed = False
        now = time.time()
        for entry in self._entries.values():
            if not entry.session_exemplars:
                continue
            if entry.source == "memory" and entry.profile_id is not None:
                if memory.reinforce(entry.profile_id, entry.session_exemplars) is not None:
                    changed = True
                continue
            if not auto_profiles_enabled or entry.duration_sec < min_auto_profile_sec:
                continue
            digest = hashlib.sha256(f"{session_id}|{entry.speaker_id}".encode("utf-8")).hexdigest()[:12]
            bank = list(entry.session_exemplars)
            mean = np.sum(bank, axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 1e-8:
                continue
            memory.add_profile(
                SpeakerProfile(
                    profile_id=f"auto-{digest}",
                    name=entry.label,
                    embedding=(mean / norm).astype(np.float32),
                    sample_count=len(bank),
                    created_at=now,
                    updated_at=now,
                    exemplars=tuple(bank),
                    kind="auto",
                )
            )
            changed = True
        return changed

    def _create_entry(self) -> IdentityEntry:
        entry = IdentityEntry(
            speaker_id=f"diarization:identity-{len(self._entries) + 1:03d}",
            label=self._session_label(None),
            source="diarization",
            profile_id=None,
        )
        self._entries[entry.speaker_id] = entry
        return entry

    def _note(self, entry: IdentityEntry, embedding: np.ndarray, duration_sec: float) -> None:
        entry.duration_sec += duration_sec
        if duration_sec < self.min_reinforce_sec:
            return
        vector = embedding.astype(np.float32)
        if entry.exemplars and bank_similarity(entry.exemplars, vector) < self.reinforce_floor:
            # Too dissimilar to trust for learning; keep the assignment but do
            # not pollute the bank.
            return
        entry.exemplars = merge_into_bank(
            entry.exemplars,
            vector,
            merge_similarity=self.merge_similarity,
            max_exemplars=self.max_exemplars,
        )
        entry.session_exemplars = merge_into_bank(
            entry.session_exemplars,
            vector,
            merge_similarity=self.merge_similarity,
            max_exemplars=self.max_exemplars,
        )

    def _best_match(self, embedding: np.ndarray) -> tuple[Optional[IdentityEntry], float, float]:
        best_entry: Optional[IdentityEntry] = None
        best = 0.0
        second = 0.0
        for entry in self._entries.values():
            if not entry.exemplars:
                continue
            similarity = bank_similarity(entry.exemplars, embedding)
            if similarity > best:
                best_entry, second, best = entry, best, similarity
            elif similarity > second:
                second = similarity
        return best_entry, best, second

    def _session_label(self, preferred: Optional[str]) -> str:
        used = {entry.label for entry in self._entries.values()}
        if preferred is not None:
            if preferred not in used:
                return preferred
            if _LABEL_NUMBER.match(preferred) is None:
                return preferred  # named profiles keep their name even if repeated
        numbers = set()
        for label in used:
            found = _LABEL_NUMBER.match(label)
            if found is not None:
                numbers.add(int(found.group(1)))
        number = 1
        while number in numbers:
            number += 1
        return f"Speaker {number}"


def turn_embeddings(
    turns: list[DiarizationTurn],
    samples: np.ndarray,
    sample_rate_hz: int,
    embedder: Any,
    *,
    min_turn_sec: float = 1.0,
    max_turn_sec: float = 12.0,
) -> list[Optional[np.ndarray]]:
    """One voice embedding per diarized turn (None where the turn is too
    short). Per-turn granularity matters: within-window clustering may merge
    two similar voices, but an individual turn is single-voice."""
    out: list[Optional[np.ndarray]] = []
    for turn in turns:
        length = max(0.0, turn.end_sec - turn.start_sec)
        if length < min_turn_sec:
            out.append(None)
            continue
        start = max(0, int(turn.start_sec * sample_rate_hz))
        end = min(samples.size, int(turn.end_sec * sample_rate_hz))
        limit = start + int(max_turn_sec * sample_rate_hz)
        end = min(end, limit)
        if end <= start:
            out.append(None)
            continue
        try:
            out.append(embedder.embed(samples[start:end], sample_rate_hz))
        except ValueError:
            out.append(None)
    return out
