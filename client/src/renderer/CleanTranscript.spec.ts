import assert from "node:assert/strict";

import { formatCleanTranscript, type CleanTranscriptSegment } from "./CleanTranscript";

function segment(
  id: string,
  startSec: number,
  speakerId: string,
  speakerLabel: string,
  text: string
): CleanTranscriptSegment {
  return {
    id,
    start_sec: startSec,
    text,
    speaker_id: speakerId,
    speaker_label: speakerLabel
  };
}

assert.equal(formatCleanTranscript([]), "");

assert.equal(
  formatCleanTranscript([
    segment("3", 3, "alice", "Alice", "Great."),
    segment("1", 1, "alice", "Alice", "Hello there."),
    segment("2", 2, "alice", "Alice", "How are you?"),
    segment("4", 4, "bob", "Bob", "  Fine.  "),
    segment("5", 5, "alice", "Alice", "Thanks.")
  ]),
  "Alice: Hello there. How are you? Great.\n\nBob: Fine.\n\nAlice: Thanks."
);

assert.equal(
  formatCleanTranscript([
    segment("1", 1, "speaker-1", "", "First line"),
    segment("2", 2, "speaker-1", "", "continues")
  ]),
  "speaker-1: First line continues"
);

assert.equal(
  formatCleanTranscript([
    segment("1", 1, "alice", "Alice", "Hello"),
    segment("2", 2, "alice", "Alice", ", everyone.")
  ]),
  "Alice: Hello, everyone."
);
