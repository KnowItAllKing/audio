import assert from "node:assert/strict";

import { buildJotArgs, validateArchiveExtractRequest } from "./ArchiveJotExtractor";
import type { ArchiveExtractRequest } from "../shared/ArchiveJot";

function request(overrides: Partial<ArchiveExtractRequest> = {}): ArchiveExtractRequest {
  return {
    requestId: "extract-1",
    sessionId: "session-1",
    batch: "cadence-2026-08-21-standup",
    provider: "codex",
    model: "",
    transcript: "Alice: Ship Friday.",
    jottedTexts: [],
    ...overrides
  };
}

validateArchiveExtractRequest(request());
assert.throws(() => validateArchiveExtractRequest(request({ batch: "Not Kebab" })));
assert.throws(() => validateArchiveExtractRequest(request({ batch: "-leading-dash" })));
assert.throws(() => validateArchiveExtractRequest(request({ transcript: "   " })));
assert.throws(() => validateArchiveExtractRequest(request({ sessionId: "bad session id!" })));
assert.throws(() =>
  validateArchiveExtractRequest(request({ provider: "other" as unknown as "codex" }))
);
assert.throws(() =>
  validateArchiveExtractRequest(request({ jottedTexts: [42] as unknown as string[] }))
);

assert.deepEqual(buildJotArgs("Alice: ship Friday.", "cadence-2026-08-21-standup", "session-1"), [
  "jot",
  "--batch",
  "cadence-2026-08-21-standup",
  "--source",
  "cadence:session-1",
  "--",
  "Alice: ship Friday."
]);
