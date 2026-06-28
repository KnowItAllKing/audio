WHISPER_E2E_MODEL ?= tiny

.PHONY: server-test client-test client-typecheck real-whisper-e2e real-whisper-hard-e2e security-audit verify

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

security-audit:
	pnpm audit --audit-level=moderate
	uv audit --project server --locked

verify: server-test client-test client-typecheck security-audit
