from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _disable_default_diarization_in_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIARIZATION_BACKEND", "off")
    monkeypatch.setenv("SPEAKER_MEMORY_ENABLED", "0")


@pytest.fixture
def require_loopback_bind() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
    except OSError as exc:
        pytest.skip(f"loopback bind is unavailable in this environment: {exc}")
