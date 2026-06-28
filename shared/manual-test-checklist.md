# Manual Test Checklist (Phase 5)

## Local (same machine)

- [ ] **Start server**
  - [ ] `cd /path/to/audio`
  - [ ] `uv sync --project server --extra dev`
  - [ ] `PYTHONPATH=. uv run --project server python -m server.main`
  - [ ] Confirm server logs show it is listening on `0.0.0.0:8765`

- [ ] **Start Electron client**
  - [ ] `pnpm install`
  - [ ] `pnpm --filter audio-client dev`

- [ ] **Select devices**
  - [ ] Choose a **Mic device**
  - [ ] Optionally choose a **System device** (virtual loopback input)
  - [ ] Click **Refresh devices** if labels are missing

- [ ] **Connect & Start**
  - [ ] Set WebSocket URL to `ws://localhost:8765`
  - [ ] Click **Connect & Start**
  - [ ] Confirm connection status shows **Connected**
  - [ ] Speak into mic and confirm transcript updates approximate spoken content
  - [ ] Stop speaking and confirm mic gate status changes to **noise** with no new mic sends
  - [ ] Click **Mute mic** and confirm mic sends stop
  - [ ] Click **Unmute mic** and confirm mic sends resume when speaking

- [ ] **Stop + Save transcript**
  - [ ] Click **Stop**
  - [ ] Save `transcript-<session-id>.json`
  - [ ] Open the saved JSON and verify:
    - [ ] `session_id` matches UI
    - [ ] `started_at` / `ended_at` present
    - [ ] `segments[]` present and each segment has `id`, `start_sec`, `end_sec`, `text`, `stream_tags`, `speaker_id`, `speaker_label`, `is_final`

## Tailscale (machine A server, machine B client)

Assumption: Tailscale is installed and authenticated on both machines. App-level auth is disabled; rely on Tailscale network security.

- [ ] **Machine A: start server**
  - [ ] Determine server Tailscale hostname/IP (`tailscale status`)
  - [ ] Start server listening on port 8765:
    - [ ] `uv sync --project server --extra dev`
    - [ ] `WS_PORT=8765 PYTHONPATH=. uv run --project server python -m server.main`

- [ ] **Machine B: start Electron client**
  - [ ] `pnpm --filter audio-client dev`
  - [ ] Set WebSocket URL to `ws://<server-tailnet-hostname>:8765`
  - [ ] Click **Connect & Start**
  - [ ] Confirm transcript updates arrive with acceptable latency

- [ ] **Stability**
  - [ ] Let it run for **30–60 minutes**
  - [ ] Confirm no disconnects/crashes

## Long session (1 hour)

- [ ] Run a continuous session for **≥ 1 hour**
- [ ] Confirm:
  - [ ] No crashes (server/client)
  - [ ] Memory usage is roughly stable (no obvious unbounded growth)
  - [ ] Transcript updates continue arriving throughout the session
