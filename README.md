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

# Optional: WS_PORT=8765 (default)
# Phase 3 knobs:
#   TRANSCRIBE_INTERVAL_SEC=1.0
#   MIN_NEW_AUDIO_SEC=2.0
#   WINDOW_SEC=8.0
#   MAX_BUFFER_SEC=600.0
# Enable real local Whisper (Phase 3.4, OpenAI reference whisper):
#   (Requires Python 3.9.x for this repo)
#   WHISPER_MODEL=base   (or small/medium/large-v3)
#   WHISPER_DEVICE=cpu|cuda
PYTHONPATH=.. uv run --project server python -m server.main
```

### Optional: VAD / noise filtering (server-side)

The server runs an always-on VAD gate to avoid transcribing near-silence (reduces hallucinated short tokens).

- **Tune WebRTC VAD**
  - `VAD_ML_AGGRESSIVENESS=0..3` (default `2`, higher = more aggressive)
  - `VAD_ML_FRAME_MS=10|20|30` (default `20`)
  - `VAD_ML_MIN_SPEECH_RATIO=0..1` (default `0.12`)
- **Tune RMS gate (runs after WebRTC VAD)**
  - `VAD_RMS_THRESHOLD` (default `0.003`, higher = more aggressive)

### Client (Node/TypeScript)

```bash
# One-time monorepo install (requires Node + pnpm)
pnpm install

# Run the headless test client directly
cd client
# Localhost (default):
#   pnpm dev:test-client
#
# Or explicitly:
#   WS_URL=ws://localhost:8765 pnpm dev:test-client
#
# Over Tailscale:
#   WS_URL=ws://<server-tailnet-hostname>:8765 pnpm dev:test-client
#
# Optional:
#   --interval-ms=100
#   --duration-sec=600
#   --url=ws://localhost:8765
pnpm dev:test-client
```

## Phase 4: Electron GUI client

The Electron app captures **mic** and a user-selected **“system”** device (typically a virtual loopback input), streams both to the server, renders transcript updates, and saves transcript JSON on stop.

```bash
cd client
pnpm install
pnpm dev
```

Legacy headless client (kept for quick protocol testing):

```bash
cd client
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
