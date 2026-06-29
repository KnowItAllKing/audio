WHISPER_E2E_MODEL ?= tiny

.PHONY: server-test client-test client-typecheck real-whisper-e2e real-whisper-hard-e2e real-diarization speaker-memory-tts security-audit verify

server-test:
	PYTHONPATH=. uv run --project server --extra dev pytest -q

client-typecheck:
	pnpm --filter audio-client exec tsc --noEmit

client-test:
	pnpm --filter audio-client run test:audio-filter

real-whisper-e2e:
	RUN_REAL_WHISPER_E2E=1 WHISPER_E2E_MODEL=$(WHISPER_E2E_MODEL) PYTHONPATH=. uv run --project server --extra dev pytest server/tests/test_end_to_end_real_whisper.py::test_end_to_end_real_whisper_local_speech -q

real-whisper-hard-e2e:
	RUN_REAL_WHISPER_HARD_E2E=1 WHISPER_E2E_MODEL=$(WHISPER_E2E_MODEL) PYTHONPATH=. uv run --project server --extra dev pytest server/tests/test_end_to_end_real_whisper.py::test_hard_end_to_end_real_whisper_cases -q -rx

real-diarization:
	set -a; [ ! -f server/.env ] || . server/.env; set +a; RUN_REAL_DIARIZATION=1 PYTHONPATH=. uv run --project server --extra dev --extra diarization pytest server/tests/test_real_diarization.py -q -rs

speaker-memory-tts:
	RUN_SPEAKER_MEMORY_TTS=1 PYTHONPATH=. uv run --project server --extra dev pytest server/tests/test_speaker_memory_tts.py -q -rs

security-audit:
	pnpm audit --audit-level=moderate
	uv audit --project server --locked --no-extra diarization

verify: server-test client-test client-typecheck
