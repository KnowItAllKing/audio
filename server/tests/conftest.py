from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_default_diarization_in_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIARIZATION_BACKEND", "off")
