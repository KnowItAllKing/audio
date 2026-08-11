# Public two-speaker conversation validation

Validated on 2026-08-11 against the Chromium WebRTC conversational speech
dataset at revision `798e45187873d45b9251e56b9ff3345909af958e`:

- [Dataset description and reference script](https://webrtc.googlesource.com/src/+/798e45187873d45b9251e56b9ff3345909af958e/resources/audio_processing/conversational_speech/README.md)
- English Script 1, alternating a female `sp1` recording for speaker A and a
  male `sp1` recording for speaker B
- Four source WAV turns concatenated in A1, B1, A2, B2 order with 650 ms of
  silence between turns; 9.791 seconds total, mono 48 kHz PCM16

The combined recording was streamed through Cadence's real WebSocket input as
`system` audio, then finalized through the real Processed path using local
Whisper `base.en`, Sherpa-ONNX diarization constrained to two speakers,
word-timestamp speaker splitting, and the utterance assembler. No Hugging Face
token or `.env` file was used.

## Result

| Check | Result |
| --- | --- |
| Reference | `How is your day going? Quite busy. I'm preparing for my presentation on our marketing strategy. You must feel stressed out now. That's an understatement.` |
| Cadence ASR | `How is your decoying? Quite busy. I'm preparing for my presentation on our marketing strategy. You must feel stressed out now. That's an understatement.` |
| Word error rate | 7.41% (one short coarticulation error in the first turn; all later words correct) |
| Speaker-turn accuracy | 4/4 expected turns assigned to the correct recurring cluster pattern |
| Predicted pattern | `Speaker 1, Speaker 2, Speaker 1, Speaker 2` |
| Processed records | 5 unique final records; no duplicate IDs or repeated echo record |
| Clean joined turns | 4 turns; the two adjacent Speaker 2 sentence records join into one clean exported turn |

Clean Processed output (the value used by **Copy clean** and AI transcript
sync):

```text
Speaker 1: How is your decoying?

Speaker 2: Quite busy. I'm preparing for my presentation on our marketing strategy.

Speaker 1: You must feel stressed out now.

Speaker 2: That's an understatement.
```

This benchmark confirms speaker separation, stable speaker identity after a
speaker returns, word-timestamp splitting, finalization, adjacent-fragment
joining in the clean formatter, and absence of duplicate Processed output. The
remaining mismatch is acoustic-model recognition quality, not a diarization or
joining failure.
