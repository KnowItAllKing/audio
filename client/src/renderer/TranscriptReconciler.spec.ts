import assert from "node:assert/strict";

import { reconcileCrossStreamDuplicates, type ReconciliableTranscriptSegment } from "./TranscriptReconciler";

function segment(
  id: string,
  stream: "mic" | "system",
  text: string,
  startSec: number,
  endSec: number,
  speakerSource = "stream"
): ReconciliableTranscriptSegment {
  return {
    id,
    start_sec: startSec,
    end_sec: endSec,
    text,
    stream_tags: [stream],
    speaker_source: speakerSource,
    speaker_confidence: 0.9,
    is_final: true,
    full_context_available: speakerSource === "diarization"
  };
}

{
  const reconciled = reconcileCrossStreamDuplicates([
    segment("mic-1", "mic", "Welcome to the weekly design review.", 0, 3),
    segment("system-1", "system", "Welcome to the weekly design review.", 0.15, 3.1, "diarization")
  ]);
  assert.equal(reconciled.length, 1);
  assert.equal(reconciled[0].id, "system-1");
}

{
  const reconciled = reconcileCrossStreamDuplicates([
    segment("mic-1", "mic", "We should review the new navigation", 10, 13),
    segment("system-1", "system", "We should review new navigation.", 10.2, 13.1)
  ]);
  assert.equal(reconciled.length, 1);
}

{
  const reconciled = reconcileCrossStreamDuplicates([
    segment("system-1", "system", "Thank you.", 0, 1),
    segment("system-2", "system", "Thank you.", 0.1, 1.1)
  ]);
  assert.equal(reconciled.length, 2, "same-stream repetition must remain visible");
}

{
  const reconciled = reconcileCrossStreamDuplicates([
    segment("mic-1", "mic", "That sounds good.", 0, 1),
    segment("system-1", "system", "That sounds good.", 8, 9)
  ]);
  assert.equal(reconciled.length, 2, "separate turns must remain visible");
}

{
  const reconciled = reconcileCrossStreamDuplicates([
    segment("mic-1", "mic", "I agree with that approach.", 0, 2),
    segment("system-1", "system", "I disagree with that approach.", 0.1, 2.1)
  ]);
  assert.equal(reconciled.length, 2, "different simultaneous dialogue must remain visible");
}
