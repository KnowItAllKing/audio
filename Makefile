.PHONY: server-test client-typecheck security-audit verify

server-test:
	PYTHONPATH=. uv run --project server --extra dev pytest -q

client-typecheck:
	pnpm --filter audio-client exec tsc --noEmit

security-audit:
	pnpm audit --audit-level=moderate
	uv audit --project server --locked

verify: server-test client-typecheck security-audit
