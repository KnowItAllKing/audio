import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { runArchiveExtraction } from "./ArchiveJotExtractor";
import type { ArchiveExtractRequest } from "../shared/ArchiveJot";

const aiExecutable = fileURLToPath(new URL("./test-fixtures/fake-ai-cli.mjs", import.meta.url));
const archiveExecutable = fileURLToPath(new URL("./test-fixtures/fake-archive-cli.mjs", import.meta.url));
const cwd = fileURLToPath(new URL("./test-fixtures", import.meta.url));

const workDir = await mkdtemp(join(tmpdir(), "cadence-archive-spec-"));
const jotLog = join(workDir, "jots.jsonl");

const fakeItems = JSON.stringify([
  { text: "Alice: ship the beta Friday.", kind: "decision", speakers: ["Alice"] },
  { text: "Bob: write the release notes.", kind: "action-item" }
]);

function request(overrides: Partial<ArchiveExtractRequest> = {}): ArchiveExtractRequest {
  return {
    requestId: "extract-integration",
    sessionId: "session-1",
    batch: "cadence-2026-08-21-standup",
    provider: "codex",
    model: "",
    transcript: "Alice: We will ship the beta on Friday.\n\nBob: I'll write the release notes.",
    jottedTexts: [],
    ...overrides
  };
}

async function loggedJotArgs(): Promise<string[][]> {
  const raw = await readFile(jotLog, "utf8").catch(() => "");
  return raw
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as string[]);
}

try {
  const env = { CADENCE_FAKE_AI_RESPONSE: fakeItems, CADENCE_FAKE_ARCHIVE_LOG: jotLog };
  const alreadyJotted = new Set<string>();
  const first = await runArchiveExtraction(request(), {
    aiExecutable,
    archiveExecutable,
    cwd,
    alreadyJotted,
    env
  });
  assert.equal(first.ok, true);
  assert.equal(first.jotted.length, 2);
  assert.equal(first.skipped, 0);

  const argsAfterFirst = await loggedJotArgs();
  assert.equal(argsAfterFirst.length, 2);
  assert.deepEqual(argsAfterFirst[0], [
    "jot",
    "--batch",
    "cadence-2026-08-21-standup",
    "--source",
    "cadence:session-1",
    "--",
    "Alice: ship the beta Friday."
  ]);

  // Overlapping window returns the same items: the main-process memory dedupes.
  const second = await runArchiveExtraction(request({ requestId: "extract-repeat" }), {
    aiExecutable,
    archiveExecutable,
    cwd,
    alreadyJotted,
    env
  });
  assert.equal(second.ok, true);
  assert.equal(second.jotted.length, 0);
  assert.equal(second.skipped, 2);
  assert.equal((await loggedJotArgs()).length, 2);

  // The renderer-provided jotted list dedupes on its own, without main memory.
  const third = await runArchiveExtraction(
    request({
      requestId: "extract-prompt-dedupe",
      jottedTexts: ["Alice: ship the beta Friday.", "Bob: write the release notes."]
    }),
    { aiExecutable, archiveExecutable, cwd, env }
  );
  assert.equal(third.jotted.length, 0);
  assert.equal(third.skipped, 2);

  // A model reply with no JSON payload produces no jots.
  const empty = await runArchiveExtraction(request({ requestId: "extract-empty" }), {
    aiExecutable,
    archiveExecutable,
    cwd,
    env: { ...env, CADENCE_FAKE_AI_RESPONSE: "Nothing worth keeping here." }
  });
  assert.equal(empty.ok, true);
  assert.equal(empty.jotted.length, 0);
  assert.equal((await loggedJotArgs()).length, 2);

  // A broken archive CLI surfaces an error and pushes nothing further.
  const failed = await runArchiveExtraction(request({ requestId: "extract-fail" }), {
    aiExecutable,
    archiveExecutable,
    cwd,
    env: { ...env, CADENCE_FAKE_ARCHIVE_FAIL: "1" }
  });
  assert.equal(failed.ok, false);
  assert.ok(failed.error?.includes("fake archive failure"));
  assert.equal(failed.jotted.length, 0);
} finally {
  await rm(workDir, { recursive: true, force: true });
}
