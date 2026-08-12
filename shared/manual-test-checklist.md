# Manual Test Checklist (Phase 5)

## Local (same machine)

- [ ] **Start server**
  - [ ] `cd /path/to/audio`
  - [ ] `make setup-diarization`
  - [ ] `PYTHONPATH=. uv run --project server --extra diarization python -m server.main`
  - [ ] Confirm server logs show it is listening on `0.0.0.0:8765`

- [ ] **Start Electron client**
  - [ ] `pnpm install`
  - [ ] `pnpm --filter audio-client dev`

- [ ] **Select devices**
  - [ ] Choose a **Mic device**
  - [ ] Optionally choose a **System device** (virtual loopback input)
  - [ ] Confirm likely loopback devices auto-select as **System device**
  - [ ] Click **Refresh devices** if labels are missing

- [ ] **Connect & Start**
  - [ ] Set WebSocket URL to `ws://localhost:8765`
  - [ ] Click **Connect & Start**
  - [ ] Confirm connection status shows **Connected**
  - [ ] Speak into mic and confirm transcript updates approximate spoken content
  - [ ] Keep **Raw** selected and confirm a short phrase appears shortly after input goes quiet
  - [ ] Switch to **Processed** and confirm the cleaned phrase appears a few seconds later
  - [ ] Confirm switching layers does not duplicate or erase entries in either layer
  - [ ] Scroll upward in **Raw**, wait for another Raw update, and confirm the reading position does not jump
  - [ ] Switch to **Processed**, scroll to a different position, then switch between layers and confirm each position is preserved
  - [ ] Click **Latest** and confirm the active layer returns to the newest entry and resumes following updates
  - [ ] Click **Copy clean** while Raw is selected and confirm the clipboard still contains only the Processed transcript
  - [ ] Confirm the copied text contains `Name: dialogue` turns with adjacent fragments from one speaker joined
  - [ ] Confirm copied text contains no timestamps, IDs, Raw text, or JSON fields
  - [ ] Confirm **Mic level** moves while speaking
  - [ ] Play meeting/system audio and confirm **System level** moves
  - [ ] Play audio containing at least two alternating voices
  - [ ] Confirm sustained turns receive different anonymous speaker labels
  - [ ] Confirm several adjacent ASR fragments from one speaker join into a single phrase
  - [ ] Confirm a speaker change inside one ASR window creates separate transcript entries
  - [ ] With both mic and system sources active, let the mic hear the meeting audio and confirm each line still appears only once
  - [ ] Confirm a genuinely repeated phrase later in the meeting remains as a separate entry
  - [ ] Stop speaking and confirm mic gate status changes to **noise** with no new mic sends
  - [ ] Click **Mute mic** and confirm mic sends stop
  - [ ] Click **Unmute mic** and confirm mic sends resume when speaking
- [ ] **Window and control behavior**
  - [ ] Resize the Electron window to its minimum size (900 × 640)
  - [ ] Confirm the app frame itself does not scroll or drift
  - [ ] Scroll the audio-source rail and confirm the session controls remain visible
  - [ ] Expand **Advanced** and confirm source controls remain reachable in the rail
  - [ ] Confirm the transcript, layer switch, view switch, and **Latest** control remain usable

- [ ] **AI transcript Q&A**
  - [ ] Confirm an installed Codex and/or Claude CLI shows as ready in **Ask AI**
  - [ ] Create a thread and confirm the first question sends the current clean Processed transcript
  - [ ] Ask a follow-up and confirm the same provider thread resumes without a duplicate transcript snapshot
  - [ ] Let more Processed dialogue arrive and confirm **Update ready** and **Sync transcript** appear
  - [ ] Create a second provider tab, switch tabs, and confirm each conversation is independent
  - [ ] Reload the Electron renderer and confirm tabs and messages return
  - [ ] Cancel an in-flight request and confirm the CLI process exits and the tab remains usable
  - [ ] Close a tab and confirm it disappears without affecting other tabs

- [ ] **Speaker memory persistence**
  - [ ] Review a voice clip, enter a name, and click **Save**
  - [ ] Quit and reopen Cadence
  - [ ] Confirm the saved voice count returns
  - [ ] Start another session and confirm the saved profile is sent to the server

- [ ] **Stop + Save transcript**
  - [ ] Click **Stop**
  - [ ] Confirm status changes to **Finalizing…**, then **Finalized**
  - [ ] Confirm the last short phrase becomes final and appears before save
  - [ ] Confirm no duplicate partial remains beside its final replacement
  - [ ] Save `transcript-<session-id>.json`
  - [ ] Open the saved JSON and verify:
    - [ ] `session_id` matches UI
    - [ ] `started_at` / `ended_at` present
    - [ ] `segments[]` present and each segment has `id`, `start_sec`, `end_sec`, `text`, `stream_tags`, `speaker_id`, `speaker_label`, `is_final`
    - [ ] `segments[]` contains the processed/canonical transcript
    - [ ] `raw_segments[]` contains the immediate ASR snapshots
    - [ ] every saved segment has `is_final: true`

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
