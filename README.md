# audio

Monorepo for **audio streaming + transcription over WebSockets**.

## Purpose

- Stream raw audio from a client (mic + system audio)
- Transcribe on a Python server (Whisper)
- Send transcript updates back to the client in a consistent, future-proof JSON protocol

## Layout

- `server/`: Python WebSocket server + Whisper transcription
- `client/`: Node/Electron app (capture audio, stream to server, show transcript)
- `shared/`: Protocol docs and (later) shared types

## Tech stack (planned)

- **Server**: Python (3.11+) + WebSockets + Whisper
- **Client**: Electron + TypeScript
- **Deployment network**: Tailscale (for connecting client ↔ server in production)

## Protocol

See `shared/protocol.md` for the WebSocket message schemas and expectations.

## Phase 1: run the dummy pipeline (localhost)

This repo is set up as a **pnpm workspace** with **Turborepo** (`turbo.json`).

### Server (Python)

```bash
# Install uv (see: https://docs.astral.sh/uv/)
# Then from repo root:
uv sync --project server

# Optional: WS_PORT=8765 (default), TRANSCRIPT_EVERY_N=10 (default)
uv run --project server python -m server.main
```

### Client (Node/TypeScript)

```bash
# One-time monorepo install (requires Node + pnpm)
pnpm install

# Run the headless test client directly
cd client
# Optional:
#   --interval-ms=100
#   --duration-sec=600
#   --url=ws://localhost:8765
pnpm dev:test-client
```

### Optional: run via Turbo

Once you add Turborepo as a dev dependency (e.g. `pnpm add -D turbo -w`), you can run:

```bash
# from repo root
pnpm dev:server
pnpm dev:client
pnpm dev
```
