from __future__ import annotations

import numpy as np

from server.diarization import DiarizationTurn
from server.speaker_identity import SessionSpeakerRegistry, turn_embeddings
from server.speaker_memory import (
    SpeakerMemory,
    SpeakerProfile,
    merge_into_bank,
    speaker_profile_from_protocol,
)


def unit(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# Three well-separated synthetic voices plus a slight variation of the first.
ALICE = unit([1.0, 0.1, 0.0, 0.0])
ALICE_OTHER_MIC = unit([0.9, 0.35, 0.0, 0.1])
BOB = unit([0.0, 1.0, 0.1, 0.0])
CARA = unit([0.0, 0.0, 1.0, 0.1])


def turn(start: float, end: float, speaker: str) -> DiarizationTurn:
    return DiarizationTurn(start_sec=start, end_sec=end, speaker_id=speaker, speaker_label=speaker)


def test_returning_speaker_reuses_identity_after_silence() -> None:
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.03)

    first = registry.resolve_window([turn(0.0, 4.0, "p1")], [ALICE])
    assert first[0].speaker_id == "diarization:identity-001"
    assert first[0].speaker_label == "Speaker 1"

    # Bob speaks alone for a while; the mapper minted a fresh provisional id.
    second = registry.resolve_window([turn(0.0, 4.0, "p2")], [BOB])
    assert second[0].speaker_id == "diarization:identity-002"

    # Alice returns much later under yet another provisional id (no overlap
    # continuity left) — her voice must bring back the same session identity.
    third = registry.resolve_window([turn(0.0, 3.0, "p7")], [ALICE])
    assert third[0].speaker_id == "diarization:identity-001"
    assert third[0].speaker_label == "Speaker 1"


def test_short_turns_follow_cluster_majority_and_continuity() -> None:
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.03)
    # One provisional cluster: a long Alice turn plus a short unembeddable one.
    resolved = registry.resolve_window(
        [turn(0.0, 4.0, "p1"), turn(5.0, 5.4, "p1")],
        [ALICE, None],
    )
    assert resolved[0].speaker_id == resolved[1].speaker_id

    # Next window: only a short turn under the same provisional id — follows
    # the continuity alias even without an embedding.
    follow = registry.resolve_window([turn(0.0, 0.5, "p1")], [None])
    assert follow[0].speaker_id == resolved[0].speaker_id


def test_merged_cluster_is_split_per_turn() -> None:
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.03)
    registry.resolve_window([turn(0.0, 4.0, "p1")], [ALICE])
    registry.resolve_window([turn(0.0, 4.0, "p2")], [BOB])

    # The backend glued an Alice turn and a Bob turn into ONE cluster; per-turn
    # embeddings must still route each turn to its own identity.
    resolved = registry.resolve_window(
        [turn(0.0, 3.0, "pX"), turn(4.0, 7.0, "pX")],
        [ALICE, BOB],
    )
    assert resolved[0].speaker_id == "diarization:identity-001"
    assert resolved[1].speaker_id == "diarization:identity-002"


def test_ambiguous_voice_does_not_steal_identity() -> None:
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.2)
    registry.resolve_window([turn(0.0, 4.0, "p1")], [ALICE])
    registry.resolve_window([turn(0.0, 4.0, "p2")], [ALICE_OTHER_MIC])
    # ALICE_OTHER_MIC matches ALICE at ~0.93 but margin over the second-best
    # candidate is enforced only when two entries are close; with a huge margin
    # requirement and one entry, the match still lands on Alice's entry.
    entries = registry.session_entries()
    assert len(entries) == 1


def test_seeded_profile_wins_and_reinforces() -> None:
    profile = SpeakerProfile(
        profile_id="john-abc",
        name="John",
        embedding=ALICE,
        sample_count=1,
        created_at=1.0,
        updated_at=1.0,
        exemplars=(ALICE,),
    )
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.03, reinforce_floor=0.8)
    registry.seed_profiles([profile])

    resolved = registry.resolve_window([turn(0.0, 5.0, "p1")], [ALICE])
    assert resolved[0].speaker_id == "memory:john-abc"
    assert resolved[0].speaker_label == "John"
    assert registry.match_info("memory:john-abc") is not None

    memory = SpeakerMemory(threshold=0.9)
    memory.add_profile(profile)
    assert registry.export_updates(memory, session_id="s1") is True
    reinforced = memory.get("john-abc")
    assert reinforced is not None and reinforced.sample_count > 1


def test_low_similarity_session_voice_becomes_second_exemplar_when_named() -> None:
    """The 'John has two voices' path: a session voice that does NOT match
    John's stored fingerprint stays separate, and enrolling it under John's
    name merges the banks instead of averaging the voices together."""
    memory = SpeakerMemory(threshold=0.9, fingerprinter=_StubFingerprinter(ALICE))
    john = memory.enroll(name="John", samples=np.ones(16_000, dtype=np.float32), sample_rate_hz=16_000)
    assert len(john.bank()) == 1

    # A very different rendition of John's voice enrolled under the same name.
    memory.fingerprinter = _StubFingerprinter(BOB)
    updated = memory.enroll(name="John", samples=np.ones(16_000, dtype=np.float32), sample_rate_hz=16_000)
    assert updated.profile_id == john.profile_id
    assert len(updated.bank()) == 2  # two distinct exemplars, not one blurred average

    # Both renditions now match the single named profile.
    for voice in (ALICE, BOB):
        match = memory.match_embedding(voice)
        assert match is not None and match.profile.name == "John"


def test_auto_profiles_export_only_substantial_speakers() -> None:
    registry = SessionSpeakerRegistry(match_threshold=0.9, match_margin=0.03)
    registry.resolve_window([turn(0.0, 20.0, "p1")], [ALICE])
    registry.resolve_window([turn(0.0, 3.0, "p2")], [BOB])

    memory = SpeakerMemory(threshold=0.9)
    assert registry.export_updates(memory, session_id="s1", min_auto_profile_sec=10.0) is True
    profiles = memory.profiles()
    assert len(profiles) == 1
    assert profiles[0].kind == "auto"
    assert profiles[0].name == "Speaker 1"

    # Exports are deterministic per session+speaker, so a second stop does not
    # duplicate the profile.
    registry.export_updates(memory, session_id="s1", min_auto_profile_sec=10.0)
    assert len(memory.profiles()) == 1


def test_profile_bank_round_trips_through_protocol() -> None:
    profile = SpeakerProfile(
        profile_id="p",
        name="Pat",
        embedding=ALICE,
        sample_count=3,
        created_at=1.0,
        updated_at=2.0,
        exemplars=(ALICE, BOB),
        kind="auto",
    )
    parsed = speaker_profile_from_protocol(profile.to_protocol())
    assert parsed is not None
    assert parsed.kind == "auto"
    assert len(parsed.bank()) == 2
    assert np.allclose(parsed.bank()[0], ALICE, atol=1e-6)
    assert np.allclose(parsed.bank()[1], BOB, atol=1e-6)

    # Old single-fingerprint payloads still parse.
    legacy = {"profile_id": "q", "name": "Quinn", "fingerprint": [float(v) for v in CARA]}
    parsed_legacy = speaker_profile_from_protocol(legacy)
    assert parsed_legacy is not None
    assert parsed_legacy.kind == "named"
    assert len(parsed_legacy.bank()) == 1


def test_merge_into_bank_caps_and_merges() -> None:
    bank = [ALICE]
    bank = merge_into_bank(bank, unit([0.98, 0.15, 0.0, 0.0]))  # near-duplicate merges
    assert len(bank) == 1
    bank = merge_into_bank(bank, BOB)  # distinct voice appended
    assert len(bank) == 2
    for index in range(10):  # overflow keeps the bank bounded
        bank = merge_into_bank(bank, unit([1.0, float(index) / 10.0, 0.5, 0.0]), max_exemplars=4)
    assert len(bank) <= 4


def test_turn_embeddings_skips_short_turns() -> None:
    class CountingEmbedder:
        kind = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
            self.calls += 1
            return ALICE

    embedder = CountingEmbedder()
    samples = np.zeros(16_000 * 10, dtype=np.float32)
    turns = [turn(0.0, 0.5, "a"), turn(1.0, 4.0, "a"), turn(5.0, 9.0, "b")]
    out = turn_embeddings(turns, samples, 16_000, embedder)
    assert out[0] is None
    assert out[1] is not None and out[2] is not None
    assert embedder.calls == 2


class _StubFingerprinter:
    kind = "stub"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def embed(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        return self.vector
