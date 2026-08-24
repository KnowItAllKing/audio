"""Speaker-identity validation across sliding diarization windows.

Reproduces the server's live loop faithfully: a long multi-speaker
conversation is diarized in overlapping windows (DIARIZATION_WINDOW_SEC
wide, advancing by DIARIZATION_MIN_NEW_AUDIO_SEC), each window's turns go
through the DiarizationSpeakerMapper, and later windows replace earlier
activity in their range — exactly what _apply_diarization_turns does.

The assertion is identity, not just separation: every ground-truth speaker
must map to ONE stable session speaker id across all their non-contiguous
turns, even after they have been silent for several windows.

Opt-in like the other real-model tests:

    RUN_REAL_DIARIZATION=1 DIARIZATION_BACKEND=sherpa-onnx \
    PYTHONPATH=. uv run --project server --extra dev --extra diarization \
    pytest server/tests/test_diarization_identity.py -q -rs -s
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from server.diarization import DiarizationSpeakerMapper, DiarizationTurn
from server.speaker_memory import VoiceFingerprint, cosine_similarity


# Captured at import time: the autouse conftest fixture overwrites
# DIARIZATION_BACKEND=off for test isolation before the test body runs.
BACKEND_NAME = os.environ.get("DIARIZATION_BACKEND", "sherpa-onnx").lower()

SAMPLE_RATE = 16_000
WINDOW_SEC = 12.0
HOP_SEC = 4.0
GAP_SEC = 0.8

# Four timbre-diverse macOS voices; each speaker returns three times with
# other speakers (and multiple whole windows) in between.
VOICES = {
    "samantha": "Samantha",
    "daniel": "Daniel",
    "karen": "Karen",
    "rishi": "Rishi",
}

CONVERSATION: list[tuple[str, str]] = [
    ("samantha", "Alright everyone, let's get started with the quarterly planning review for the platform team."),
    ("daniel", "Thanks. The migration finished last week and the error rate has stayed flat since the cutover."),
    ("karen", "From the product side we still need the billing dashboard before the end of the month."),
    ("rishi", "I can take the dashboard work, but I will need the schema changes merged first."),
    ("karen", "The schema changes are already reviewed, so you should be unblocked by tomorrow morning."),
    ("samantha", "Good. Let's make sure the rollout plan includes a staged deployment across the regions."),
    ("rishi", "Staged rollout is fine, though we should keep the feature flag until the load test passes."),
    ("daniel", "I'll run the load test on Thursday and publish the results to the engineering channel."),
    ("daniel", "One more thing, the on-call rotation needs a volunteer for the holiday weekend."),
    ("rishi", "I did the last holiday rotation, so somebody else should pick this one up."),
    ("samantha", "I'll take the holiday on-call shift and swap with whoever has the following week."),
    ("karen", "Works for me. I'll update the rotation calendar right after this meeting ends."),
    ("rishi", "Before we wrap up, the vendor renewal quote came in ten percent above last year."),
    ("samantha", "Push back on the vendor quote and ask for the multi-year discount we discussed."),
    ("daniel", "Agreed, and if they will not move on price we should trial the open source alternative."),
    ("karen", "Alright, sending the meeting notes with all the action items this afternoon."),
]


@dataclass(frozen=True)
class Activity:
    start_sec: float
    end_sec: float
    speaker_id: str
    speaker_label: str


def test_speaker_identity_across_sliding_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_REAL_DIARIZATION") != "1":
        pytest.skip("set RUN_REAL_DIARIZATION=1 to run real diarization")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("macOS say and afconvert are required to synthesize local speech audio")
    # The identity embedder is always sherpa ERes2Net, regardless of which
    # backend produces the diarization turns.
    pytest.importorskip("sherpa_onnx")
    backend_name = BACKEND_NAME
    if backend_name in ("pyannote", "community"):
        pytest.importorskip("pyannote.audio")
        if not any(
            os.environ.get(key)
            for key in ("DIARIZATION_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
        ):
            pytest.skip("DIARIZATION_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN is required")

    # Realistic server defaults: speaker count is NOT known ahead of time.
    monkeypatch.delenv("DIARIZATION_MIN_SPEAKERS", raising=False)
    monkeypatch.delenv("DIARIZATION_MAX_SPEAKERS", raising=False)
    monkeypatch.setenv("DIARIZATION_MIN_TURN_SEC", "0.2")
    # With an identity layer on top, within-window clustering should
    # over-split rather than merge distinct voices; the registry re-merges
    # clusters of the same voice. Mirrors the server default when identity
    # resolution is enabled.
    monkeypatch.setenv(
        "DIARIZATION_CLUSTER_THRESHOLD", os.environ.get("DIARIZATION_CLUSTER_THRESHOLD", "0.35")
    )

    from server.diarization import PyannoteDiarizationBackend, SherpaOnnxDiarizationBackend
    from server.speaker_identity import SessionSpeakerRegistry
    from server.speaker_memory import SpeakerMemory, SherpaVoiceEmbedder, default_match_threshold

    samples, truth = build_conversation(tmp_path)
    total_sec = samples.size / SAMPLE_RATE
    report_embedding_separability(samples, truth)

    if backend_name in ("pyannote", "community"):
        backend = PyannoteDiarizationBackend(device=os.environ.get("DIARIZATION_DEVICE", "cpu"))
    else:
        backend = SherpaOnnxDiarizationBackend()
    print(f"\ndiarization backend: {backend.model_name}")
    embedder = SherpaVoiceEmbedder()
    registry = SessionSpeakerRegistry()
    activities = simulate_session(samples, backend, DiarizationSpeakerMapper(), embedder, registry)

    print(f"\nconversation: {total_sec:.1f}s, {len(truth)} turns, {len(VOICES)} true speakers")
    dominant = report_identity(activities, truth)

    ids_by_speaker: dict[str, set[str]] = {}
    for (speaker, _, _), mapped_id in dominant.items():
        assert mapped_id is not None, f"no diarization coverage for turn {speaker}"
        ids_by_speaker.setdefault(speaker, set()).add(mapped_id)

    for speaker, ids in sorted(ids_by_speaker.items()):
        assert len(ids) == 1, (
            f"{speaker} fragmented into {len(ids)} session ids across returning turns: {sorted(ids)}"
        )
    all_ids = [next(iter(ids)) for ids in ids_by_speaker.values()]
    assert len(set(all_ids)) == len(VOICES), (
        f"speakers merged: {len(set(all_ids))} distinct ids for {len(VOICES)} speakers: {ids_by_speaker}"
    )

    # --- Cross-session identity: stop exports auto profiles; a new session
    # seeded with them must recognize the same voices under the same labels.
    memory = SpeakerMemory(threshold=default_match_threshold(embedder), fingerprinter=embedder)
    assert registry.export_updates(memory, session_id="session-one")
    exported = memory.profiles()
    print(f"\nexported auto profiles: {[(p.name, p.kind, len(p.bank())) for p in exported]}")
    assert len(exported) == len(VOICES)

    label_by_true_speaker: dict[str, str] = {}
    for (speaker, _, _), mapped_id in dominant.items():
        label = next(a.speaker_label for a in activities if a.speaker_id == mapped_id)
        label_by_true_speaker[speaker] = label

    rejoined = SessionSpeakerRegistry()
    rejoined.seed_profiles(exported)
    second_cut = samples[: int(35.0 * SAMPLE_RATE)]
    second_activities = simulate_session(
        second_cut, backend, DiarizationSpeakerMapper(), embedder, rejoined
    )
    print("\nsecond session (same voices, seeded from exported profiles):")
    second_dominant = report_identity(second_activities, [t for t in truth if t[2] <= 35.0])
    for (speaker, _, _), mapped_id in second_dominant.items():
        assert mapped_id is not None
        assert mapped_id.startswith("memory:auto-"), (
            f"{speaker} was not recognized from the stored profile: {mapped_id}"
        )
        label = next(a.speaker_label for a in second_activities if a.speaker_id == mapped_id)
        assert label == label_by_true_speaker[speaker], (
            f"{speaker} label changed across sessions: {label!r} vs {label_by_true_speaker[speaker]!r}"
        )


def build_conversation(tmp_path: Path) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    pcm_by_turn: list[tuple[str, bytes]] = []
    for index, (speaker, text) in enumerate(CONVERSATION):
        pcm, rate = synthesize_speech_pcm(tmp_path, f"turn-{index:02d}", text, voice=VOICES[speaker])
        assert rate == SAMPLE_RATE
        pcm_by_turn.append((speaker, pcm))

    out = bytearray()
    truth: list[tuple[str, float, float]] = []
    gap = b"\x00" * int(SAMPLE_RATE * GAP_SEC) * 2
    bytes_per_sec = SAMPLE_RATE * 2
    for index, (speaker, pcm) in enumerate(pcm_by_turn):
        if index > 0:
            out.extend(gap)
        start_sec = len(out) / bytes_per_sec
        out.extend(pcm)
        truth.append((speaker, start_sec, len(out) / bytes_per_sec))
    samples = np.frombuffer(bytes(out), dtype="<i2").astype(np.float32) / 32768.0
    return samples, truth


def simulate_session(
    samples: np.ndarray,
    backend,
    mapper: DiarizationSpeakerMapper,
    embedder,
    registry,
) -> list[Activity]:
    """Mirror main.py: diarize the trailing window on each flush, map local
    cluster ids to session ids, resolve voice identity, and let the newest
    window replace overlapping diarization activity."""
    from server.speaker_identity import turn_embeddings

    total_sec = samples.size / SAMPLE_RATE
    activities: list[Activity] = []
    end = HOP_SEC
    while True:
        end = min(end, total_sec)
        start = max(0.0, end - WINDOW_SEC)
        window = samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        turns = backend.diarize(window, SAMPLE_RATE)
        embeddings = turn_embeddings(turns, window, SAMPLE_RATE, embedder)
        mapped = mapper.map_turns(
            turns,
            base_offset_sec=start,
            window_duration_sec=window.size / SAMPLE_RATE,
        )
        mapped = registry.resolve_window(mapped, embeddings)
        replaced: list[Activity] = []
        for activity in activities:
            if not (activity.start_sec < end and activity.end_sec > start):
                replaced.append(activity)
                continue
            # Trim boundary-straddling activity instead of dropping it,
            # mirroring SpeakerTracker.remove_activities_in_range.
            if activity.start_sec < start:
                replaced.append(
                    Activity(activity.start_sec, start, activity.speaker_id, activity.speaker_label)
                )
            if activity.end_sec > end:
                replaced.append(
                    Activity(end, activity.end_sec, activity.speaker_id, activity.speaker_label)
                )
        activities = replaced
        activities.extend(
            Activity(
                start_sec=start + turn.start_sec,
                end_sec=start + turn.end_sec,
                speaker_id=turn.speaker_id,
                speaker_label=turn.speaker_label,
            )
            for turn in mapped
        )
        if end >= total_sec:
            return sorted(activities, key=lambda item: item.start_sec)
        end += HOP_SEC


def report_identity(
    activities: list[Activity],
    truth: list[tuple[str, float, float]],
) -> dict[tuple[str, float, float], str | None]:
    dominant: dict[tuple[str, float, float], str | None] = {}
    label_by_id = {activity.speaker_id: activity.speaker_label for activity in activities}
    print("turn  speaker    mapped session id (dominant)")
    for speaker, start, end in truth:
        overlaps: dict[str, float] = {}
        for activity in activities:
            overlap = max(0.0, min(activity.end_sec, end) - max(activity.start_sec, start))
            if overlap > 0:
                overlaps[activity.speaker_id] = overlaps.get(activity.speaker_id, 0.0) + overlap
        best = max(overlaps, key=lambda item: overlaps[item]) if overlaps else None
        dominant[(speaker, start, end)] = best
        label = label_by_id.get(best, "-") if best else "-"
        print(f"{start:5.1f}s {speaker:<10} {best or 'NONE'} ({label})")

    minted = {activity.speaker_id for activity in activities}
    print(f"distinct session ids on final timeline: {len(minted)}")
    return dominant


def report_embedding_separability(samples: np.ndarray, truth: list[tuple[str, float, float]]) -> None:
    """Sanity-check the shared VoiceFingerprint on this material: within-speaker
    similarity across turns must exceed cross-speaker similarity."""
    fingerprinter = VoiceFingerprint()
    embeddings: dict[str, list[np.ndarray]] = {}
    for speaker, start, end in truth:
        clip = samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        embeddings.setdefault(speaker, []).append(fingerprinter.embed(clip, SAMPLE_RATE))

    speakers = sorted(embeddings)
    print("\nVoiceFingerprint separability (mean cosine):")
    print("           " + "  ".join(f"{item:<9}" for item in speakers))
    for left in speakers:
        row = []
        for right in speakers:
            values = [
                cosine_similarity(a, b)
                for i, a in enumerate(embeddings[left])
                for j, b in enumerate(embeddings[right])
                if left != right or i < j
            ]
            row.append(float(np.mean(values)) if values else 0.0)
        print(f"{left:<10} " + "  ".join(f"{value:.3f}    " for value in row))


def synthesize_speech_pcm(tmp_path: Path, name: str, text: str, *, voice: str) -> tuple[bytes, int]:
    aiff_path = tmp_path / f"{name}.aiff"
    wav_path = tmp_path / f"{name}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(wav_path)],
        check=True,
        capture_output=True,
    )
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == SAMPLE_RATE
        return wf.readframes(wf.getnframes()), wf.getframerate()
