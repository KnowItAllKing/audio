# audio

Monorepo for **audio streaming + transcription over WebSockets**.

## Purpose

- Stream raw audio from a client (mic + system audio)
- Transcribe on a Python server (Whisper)
- Send speaker-labeled phrase/sentence transcript updates back to the client

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

## Security baseline

Dependency installs are intentionally conservative: npm packages must be at
least 7 days old, npm dependency install scripts are disabled, and Python
package resolution uses the same 7-day cooling-off period through `uv`. See
`SECURITY.md` for the full policy and audit commands.

## Phase 1: run the dummy pipeline (localhost)

This repo is set up as a **pnpm workspace** with **Turborepo** (`turbo.json`).

### Server (Python)

```bash
# Install uv (see: https://docs.astral.sh/uv/)
# Then from repo root:
uv sync --project server --extra dev

# Optional: WS_PORT=8765 (default)
# Transcription scheduling knobs:
#   TRANSCRIBE_INTERVAL_SEC=1.0
#   MIN_NEW_AUDIO_SEC=2.0
#   WINDOW_SEC=8.0
#   MAX_BUFFER_SEC=600.0
# Utterance grouping knobs:
#   UTTERANCE_PAUSE_SEC=1.2
#   UTTERANCE_MAX_SEC=14.0
#   UTTERANCE_EMIT_PARTIALS=1
# Enable real local Whisper (Phase 3.4, OpenAI reference whisper):
#   WHISPER_MODEL=tiny   (default; use base/small/medium/large-v3 when resources allow)
#   WHISPER_DEVICE=cpu|cuda
# Hallucination controls:
#   WHISPER_CONDITION_ON_PREVIOUS_TEXT=0
#   WHISPER_NO_SPEECH_THRESHOLD=0.6
#   WHISPER_LOGPROB_THRESHOLD=-1.0
#   WHISPER_COMPRESSION_RATIO_THRESHOLD=2.4
PYTHONPATH=. uv run --project server python -m server.main
```

### Optional: VAD / noise filtering (server-side)

The server runs an always-on VAD gate to avoid transcribing near-silence (reduces hallucinated short tokens).

- **Tune WebRTC VAD**
  - `VAD_ML_AGGRESSIVENESS=0..3` (default `2`, higher = more aggressive)
  - `VAD_ML_FRAME_MS=10|20|30` (default `20`)
  - `VAD_ML_MIN_SPEECH_RATIO=0..1` (default `0.12`)
- **Tune RMS gate (runs after WebRTC VAD)**
  - `VAD_RMS_THRESHOLD` (default `0.003`, higher = more aggressive)

### Speaker labels and utterances

The protocol separates audio source tags from speaker identity:

- `stream_tags`: audio provenance, currently `mic` and/or `system`
- `speaker_id` / `speaker_label`: person label when known
- `speaker_source`: `manual`, `stream`, `zoom`, `diarization`, or `unknown`

Without Zoom metadata or diarization, mic defaults to `You` and system audio
defaults to `System audio`. The Electron client sends editable local labels.
Future Zoom RTMS events should be forwarded as `control:speaker_activity` so
the server can map participant names onto system-audio transcript utterances.

### Mock Zoom RTMS testing

`server/rtms_mock.py` provides a spec-shaped Zoom RTMS mock for local tests:

- `msg_type: 6` event updates for participants and active speaker changes
- `msg_type: 14` L16/16k/mono audio payloads
- `msg_type: 17` transcript payload shape for parity checks
- deterministic random meeting scripts and tone-coded synthetic audio

The mock pipeline test adapts those RTMS messages into the local websocket
protocol, decodes the generated tone audio with a fake ASR backend, and asserts
the final transcript text and Zoom speaker labels match the generated script.

```bash
PYTHONPATH=. uv run --project server --extra dev pytest server/tests/test_rtms_mock_pipeline.py -q
```

### Real Whisper end-to-end test

`make real-whisper-e2e` runs an opt-in local end-to-end smoke test with the
actual server websocket path, generated macOS speech audio, and real Whisper
transcription using the `tiny` model by default. It is not part of `make verify`
because it loads a real model and requires `say` plus `afconvert`.

```bash
make real-whisper-e2e
```

`make real-whisper-hard-e2e` runs harder model-quality cases through the same
real server path. These cover acronyms, product names, numbers, pause-based
phrase boundaries, background chatter, and noise. Tiny-model failures are
reported as expected failures by default; set `WHISPER_E2E_HARD_STRICT=1` to
make them fail the command. Use a larger model by overriding
`WHISPER_E2E_MODEL`.

```bash
make real-whisper-hard-e2e
make real-whisper-hard-e2e WHISPER_E2E_MODEL=turbo
WHISPER_E2E_HARD_STRICT=1 make real-whisper-hard-e2e
```

### Client (Node/TypeScript)

```bash
# One-time monorepo install from repo root (requires Node + pnpm)
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
The mic path has a mute button plus a client-side RMS gate, so quiet room noise
is dropped before it reaches Whisper. A pause finalizes the current phrase even
when the client stops sending silence.

```bash
pnpm install
pnpm --filter audio-client dev
```

Dependency install scripts are disabled. Electron's npm package may fetch its
app bundle the first time `electron` runs; treat that as an explicit local
runtime setup step. On Tart shared volumes, the downloaded `.app` can fail
macOS code-signature checks, so use a VM-local Electron bundle via
`ELECTRON_OVERRIDE_DIST_PATH` if the GUI cannot launch.

Legacy headless client (kept for quick protocol testing):

```bash
cd client
pnpm dev:test-client
```

### Optional: run via Turbo

Turbo is available from the root package:

```bash
# from repo root
pnpm dev:server
pnpm dev:client
pnpm dev
```
