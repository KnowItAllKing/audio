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
#   MIN_NEW_AUDIO_SEC=2.0              (raw ASR window threshold)
#   RAW_TRANSCRIPT_IDLE_FLUSH_SEC=0.8  (flush a short phrase after input goes quiet)
#   PROCESSED_TRANSCRIPT_LAG_SEC=3.0   (cleanup/future-context delay)
#   PROCESSED_MIN_NEW_AUDIO_SEC=2.0
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

`server.main` loads `server/.env` on startup. Existing shell environment values
win over `.env` values.

### Optional: VAD / noise filtering (server-side)

The server runs an always-on VAD gate to avoid transcribing near-silence (reduces hallucinated short tokens).

- **Tune WebRTC VAD**
  - `VAD_ML_AGGRESSIVENESS=0..3` (default `2`, higher = more aggressive)
  - `VAD_ML_FRAME_MS=10|20|30` (default `20`)
  - `VAD_ML_MIN_SPEECH_RATIO=0..1` (default `0.12`)
- **Tune RMS gate (runs after WebRTC VAD)**
  - `VAD_RMS_THRESHOLD` (default `0.003`, higher = more aggressive)

### Speaker labels and utterances

Cadence exposes two transcript layers without transcribing the same audio
twice:

- **Raw** publishes each Whisper window immediately with source-level labels.
- **Processed** reuses that ASR result after a short lag, applies future-context
  diarization, splits at word-level speaker changes, and joins fragments into
  phrases. This is the canonical saved transcript.

Saved JSON keeps the processed layer in `segments` for compatibility and also
includes the immediate layer in `raw_segments`. PCM audio is not saved.
When the microphone hears the same meeting audio as the system loopback,
the client reconciles time-aligned matching text so the dialogue is shown and
saved once; later repetition and different simultaneous speech remain separate.
Use **Copy clean** in the Electron client to copy the Processed transcript as
plain speaker turns (`Name: dialogue`) with no timestamps, IDs, or JSON fields.

### AI Q&A threads

The Electron client can open persistent transcript Q&A tabs backed by an
installed `codex` or `claude` CLI. Choose a provider and optional model, start a
thread, and ask a question; Cadence sends the same clean Processed transcript
used by **Copy clean**. Later questions resume the provider thread and only send
the transcript again when the Processed snapshot changes. **Sync transcript**
can push a changed snapshot before the next question.

Cadence starts both CLIs without a shell, disables tools, uses a dedicated empty
working directory, bounds run time/output, and exposes cancel and visible error
states. The app stores tab metadata and displayed Q&A messages locally, but not
the complete transcript snapshot. Closing a tab removes Cadence's local handle
to that provider thread.

The protocol separates audio source tags from speaker identity:

- `stream_tags`: audio provenance, currently `mic` and/or `system`
- `speaker_id` / `speaker_label`: person label when known
- `speaker_source`: `manual`, `stream`, `zoom`, `diarization`, `memory`, or `unknown`

Without Zoom metadata or diarization, mic defaults to `You` and system audio
defaults to `System audio`. The Electron client relies on server-provided
speaker identities and does not override diarization labels.

### Archive jots (live capture)

With the **Jot to archive** toggle on (off by default — it spends LLM tokens
every minute), the Electron client periodically runs a sliding window of the
Processed transcript through an AI CLI and pushes moments worth keeping
(decisions, action items, claims to verify, names/dates/numbers) into the
[`archive`](https://github.com/KnowItAllKing/archive) CLI as jots. One
meeting's jots share a batch named `cadence-YYYY-MM-DD-<slug of the session
title>`, so `archive list --batch NAME` shows them grouped and an archivist
session can distill the meeting as a unit. Each jot carries
`--source cadence:<session_id>`.

The extraction prompt reuses the AI Q&A fencing: the transcript window is
untrusted source material, and instructions spoken into a meeting are never
followed. Overlapping windows are deduplicated twice — the prompt lists what
was already jotted, and the main process keeps a per-batch memory of pushed
texts. Stopping a session runs one final extraction over the tail window after
the server flushes, then nothing further is pushed. Stopping also auto-saves
the transcript JSON to `userData/transcripts/` before the save dialog, so the
jots are never the only durable record of a meeting.

Configuration (environment variables): `CADENCE_ARCHIVE_JOTS=1` turns the
toggle on by default, `CADENCE_ARCHIVE_EXTRACT_PROVIDER` (`claude`, default,
or `codex`) and `CADENCE_ARCHIVE_EXTRACT_MODEL` (default `haiku`) pick the
extraction model — a cheap/fast model is the right choice —
`CADENCE_ARCHIVE_WINDOW_SEC` (default 150) and `CADENCE_ARCHIVE_TICK_SEC`
(default 60) shape the loop, and `CADENCE_ARCHIVE_PATH` points at the
`archive` binary when it is not on `PATH`.

### Speaker diarization

Diarization labels mixed system audio as `Speaker 1`, `Speaker 2`, etc. The
default local setup is Sherpa-ONNX and does not require a Hugging Face token:

```bash
make setup-diarization
PYTHONPATH=. uv run --project server --extra diarization python -m server.main
```

The setup target installs Sherpa-ONNX and downloads checksum-verified
segmentation and speaker-embedding models into
`~/.cache/cadence/diarization`. In `auto` mode, the server uses pyannote when
it is installed and a Hugging Face token is available; otherwise it uses the
local Sherpa models when present. Manual and Zoom speaker labels still take
priority when explicitly used.

For pyannote Community-1 instead, install its separate extra and set
`DIARIZATION_HF_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_TOKEN` (normal local use
can keep it in `server/.env`):

```bash
uv sync --project server --extra dev --extra diarization-pyannote
DIARIZATION_BACKEND=pyannote \
PYTHONPATH=. uv run --project server --extra diarization-pyannote python -m server.main
```

Useful knobs:

- `DIARIZATION_BACKEND=auto|sherpa-onnx|pyannote|off`
- `DIARIZATION_STREAMS=system`
- `DIARIZATION_MIN_SPEAKERS`
- `DIARIZATION_MAX_SPEAKERS`
- `DIARIZATION_MIN_TURN_SEC=0.2`
- `DIARIZATION_MIN_NEW_AUDIO_SEC=4`
- `DIARIZATION_WINDOW_SEC=12`
- `DIARIZATION_LAG_SEC=4` (future context improves turn boundaries at the cost of latency)
- `DIARIZATION_CLUSTER_THRESHOLD` (Sherpa speaker clustering sensitivity)
- `SPEAKER_ACTIVITY_BOUNDARY_TOLERANCE_SEC=0.35`
- `DIARIZATION_STRICT=1` to fail server startup if diarization cannot load

pyannote models may require accepting gated Hugging Face model terms for
`pyannote/speaker-diarization-community-1`.

### Speaker identity across windows and sessions

Windowed diarization alone only keeps ids stable through time overlap between
consecutive windows — a speaker who is silent for a while used to come back as
a brand-new "Speaker N". With the sherpa models installed, the server now
voice-embeds every diarized turn (3D-Speaker ERes2Net, the same model
diarization already uses) and matches it against the session's known voices,
so a returning voice reuses its id and label no matter how long it was silent.
Turn-level matching also splits window clusters that glued two similar voices
together.

Identity extends across sessions through speaker memory: at session stop,
profiles that matched during the session are reinforced with that session's
voice exemplars (one person's profile holds up to 8 distinct fingerprints —
different mics, rooms, days — and matching uses the closest one), and unnamed
diarized speakers with enough speech are exported as `auto` profiles. The
client persists them, sends them back at the next session start, and the
returning voice is recognized under its previous label. Naming an auto voice
via **Review voices** merges its bank into the named profile.

Knobs:

- `DIARIZATION_IDENTITY_ENABLED=0` to disable identity resolution
- `DIARIZATION_IDENTITY_THRESHOLD=0.68` (min cosine to reuse a known voice)
- `DIARIZATION_IDENTITY_MARGIN=0.05` (required lead over the runner-up voice)
- `SPEAKER_MEMORY_AUTO_PROFILES=0` to stop persisting unnamed session voices
- `SPEAKER_MEMORY_MIN_AUTO_PROFILE_SEC=10` (min speech before an auto profile)
- When identity is enabled, `DIARIZATION_CLUSTER_THRESHOLD` defaults to `0.35`
  so within-window clustering over-splits; the identity layer re-merges
  clusters of the same voice.

Real diarization smoke test:

```bash
make real-diarization
make real-pyannote-diarization  # uses a token already present in the environment or server/.env
```

Speaker-identity validation (3–4 distinct TTS voices, many sliding windows,
asserts one stable id per speaker within a session and recognition across
sessions):

```bash
make real-diarization-identity
make real-pyannote-diarization-identity  # same validation, pyannote backend
```

Whisper word timestamps are enabled by default (`WHISPER_WORD_TIMESTAMPS=1`).
They let one ASR result be split at a speaker change. Consecutive fragments for
the same stream and speaker are then joined into a phrase until punctuation, a
pause, a speaker change, the maximum utterance duration, or Stop finalizes it.
The mic and system streams have independent phrase assemblers, so activity on
one stream cannot prematurely finalize the other.

### Speaker memory

After a meeting, diarized speakers can be reviewed as short local audio clips.
Name a clip once, and future matching can label similar speakers with that saved
name. This is a local voice-profile layer on top of diarization; it does not
make diarization itself know real names.

Named voice fingerprints are persisted atomically in
`speaker-memory-profiles.json` under Electron's `app.getPath("userData")`
directory. Existing renderer-localStorage profiles migrate into this file on
the first launch after upgrading. The client sends the profiles to the server
at session start; the server only needs an in-memory working copy.

Defaults:

- review clips are temporary server-memory samples for playback and enrollment
- saved profiles store local voice fingerprints, not raw audio clips
- the Electron client owns persisted fingerprints in client-side app storage
- the server receives fingerprints from the client and keeps them in memory only
- set `SPEAKER_MEMORY_ENABLED=0` to disable speaker memory
- set `SPEAKER_MEMORY_THRESHOLD` to tune match confidence

Manual TTS smoke harness:

```bash
make speaker-memory-tts
```

This uses macOS `say` voices as a repeatable sanity check. Passing it is useful,
but it is not proof that real meeting voices will always match correctly.

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

The Electron app captures **mic** and a user-selected **“system”** device (typically a virtual loopback input), streams both to the server, renders transcript updates, and saves transcript JSON on stop. Stop first drains pending audio and waits for the server's final transcript acknowledgement, so the saved file includes the last phrase.
The mic path has a mute button plus a client-side RMS gate, so quiet room noise
is dropped before it reaches Whisper. A pause finalizes the current phrase even
when the client stops sending silence.
For Zoom without a Zoom app, use [Local Zoom Mode](shared/local-zoom-mode.md).

```bash
pnpm install
pnpm --filter audio-client dev
```

Build a local unsigned macOS `.app` bundle:

```bash
pnpm --filter audio-client dist:mac
open "client/dist/mac-arm64/Cadence.app"
```

The packaging configuration reuses the Electron runtime already installed in
`client/node_modules`, so rebuilding the app does not require another runtime
download.

The packaged app is a client bundle only; start the Python websocket server
separately before connecting.

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
