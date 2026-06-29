# Local Zoom Mode

Use this when you cannot create or authorize a Zoom RTMS app.

## Shape

```txt
Zoom speaker output -> virtual loopback input -> system stream
Your microphone -> mic stream
Manual current speaker label -> speaker_activity
Optional pyannote diarization -> Speaker 1 / Speaker 2 labels
Whisper -> phrase transcript
```

## macOS Setup

1. Install a virtual loopback device such as BlackHole or Loopback.
2. Open Audio MIDI Setup.
3. Create a Multi-Output Device with:
   - your normal speakers or headphones
   - the virtual loopback device
4. In Zoom, set speaker output to that Multi-Output Device.
5. In this app:
   - Mic device: your real mic
   - System device: the loopback device
   - System fallback label: `Zoom audio`
6. Click Connect & Start.
7. Confirm System level moves when someone in Zoom talks.

## Windows Setup

1. Install a virtual cable such as VB-CABLE.
2. Route Zoom speaker output to the cable.
3. Monitor the cable back to your headphones if you also need to hear it.
4. In this app, select the cable output as System device.

## Speaker Labels

Without RTMS, Zoom does not send participant metadata to this app. Use Current
Zoom speaker when you want labels:

1. Type the visible speaker name.
2. Click Set active speaker, or press Enter.
3. System transcript phrases after that point use that label until changed.

Optional pyannote diarization can split mixed system audio into anonymous speaker labels:

```bash
uv sync --project server --extra dev --extra diarization

DIARIZATION_BACKEND=pyannote \
DIARIZATION_HF_TOKEN=<hugging-face-token> \
DIARIZATION_STREAMS=system \
PYTHONPATH=. uv run --project server --extra diarization python -m server.main
```

Accept Hugging Face access terms for `pyannote/speaker-diarization-community-1`
before running this.

Manual labels beat diarization labels. Use diarization for automatic separation,
then set Current Zoom speaker when you know the real name.

## Checks

- Mic level moves only when you speak.
- System level moves when Zoom participants talk.
- Sent stats show `sys` increasing during Zoom audio.
- Transcript lines with `[system]` use the active speaker label when set.
- With diarization enabled, unlabeled `[system]` lines use `Speaker 1`,
  `Speaker 2`, etc.
- Pause between phrases creates separate final transcript lines.

## Limits

- Mixed system audio cannot be perfectly split by Zoom participant without RTMS.
- Manual speaker labels are best-effort timing hints.
- Diarization still needs manual name mapping because Zoom names are not
  available locally.
