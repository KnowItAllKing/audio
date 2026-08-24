import assert from "node:assert/strict";

import { ArchiveJotSession, selectJotWindowSegments } from "./ArchiveJotSession";

const now = 1_000_000;
const segments = [
  { id: "a", start_sec: now - 300, is_final: true, text: "too old" },
  { id: "b", start_sec: now - 100, is_final: true, text: "in window" },
  { id: "c", start_sec: now - 50, is_final: false, text: "partial" },
  { id: "d", start_sec: now - 10, is_final: true, text: "fresh" }
];
assert.deepEqual(
  selectJotWindowSegments(segments, now, 150).map((s) => s.id),
  ["b", "d"]
);

const session = new ArchiveJotSession("cadence-2026-08-21-standup", "session-1");

// Nothing final yet: never extract, even with text.
assert.equal(session.shouldExtract("Alice: hello"), false);

session.markDirty();
assert.equal(session.shouldExtract(""), false);
assert.equal(session.shouldExtract("Alice: hello"), true);

session.beginExtraction("Alice: hello");
// Same window, no new finals: skip.
assert.equal(session.shouldExtract("Alice: hello"), false);

// New finals but identical window text (fingerprint unchanged): still skip.
session.markDirty();
assert.equal(session.shouldExtract("Alice: hello"), false);
assert.equal(session.shouldExtract("Alice: hello\n\nBob: hi"), true);

session.recordJotted(["Alice: ship Friday."]);
assert.equal(session.jottedCount, 1);
assert.deepEqual(session.jottedTexts(), ["Alice: ship Friday."]);

// Failure re-arms the same window for the next tick.
session.beginExtraction("Alice: hello\n\nBob: hi");
assert.equal(session.shouldExtract("Alice: hello\n\nBob: hi"), false);
session.recordFailure();
assert.equal(session.shouldExtract("Alice: hello\n\nBob: hi"), true);
