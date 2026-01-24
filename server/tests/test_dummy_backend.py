from __future__ import annotations

import numpy as np

from server.whisper_backend import LocalWhisperBackend


def test_local_backend_returns_deterministic_segment() -> None:
    backend = LocalWhisperBackend()
    samples = np.zeros(16000, dtype=np.float32)
    segs = backend.transcribe(samples, sample_rate=16000)
    assert len(segs) == 1
    assert segs[0].text == "Whisper not wired yet"
    assert segs[0].start_sec == 0.0
    assert 0.9 <= segs[0].end_sec <= 1.1

