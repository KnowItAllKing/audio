.PHONY: server-test

server-test:
	uv run --project server --extra dev pytest

