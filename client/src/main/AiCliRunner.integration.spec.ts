import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

import { runAiCliTurn } from "./AiCliRunner";
import type { AiRunRequest } from "../shared/AiTypes";

const executable = fileURLToPath(new URL("./test-fixtures/fake-ai-cli.mjs", import.meta.url));
const cwd = fileURLToPath(new URL("./test-fixtures", import.meta.url));

function request(overrides: Partial<AiRunRequest> = {}): AiRunRequest {
  return {
    requestId: "request-integration",
    uiThreadId: "ui-integration",
    provider: "codex",
    model: "",
    transcript: "Alice: Ship Friday.",
    question: "When?",
    syncOnly: false,
    ...overrides
  };
}

const codex = await runAiCliTurn(request(), { executable, cwd });
assert.equal(codex.ok, true);
assert.equal(codex.remoteThreadId, "fake-codex-thread");
assert.ok(codex.response?.includes("transcript included"));

const codexFollowUp = await runAiCliTurn(
  request({ requestId: "request-followup", remoteThreadId: codex.remoteThreadId, transcript: "" }),
  { executable, cwd }
);
assert.equal(codexFollowUp.ok, true);
assert.ok(codexFollowUp.response?.includes("transcript unchanged"));

const claude = await runAiCliTurn(request({ provider: "claude", requestId: "request-claude" }), {
  executable,
  cwd
});
assert.equal(claude.ok, true);
assert.match(claude.remoteThreadId || "", /^[0-9a-f-]{36}$/);
assert.ok(claude.response?.includes("Fake Claude answer"));

process.env.CADENCE_FAKE_AI_DELAY_MS = "1000";
const controller = new AbortController();
const cancelledPromise = runAiCliTurn(request({ requestId: "request-cancel" }), {
  executable,
  cwd,
  signal: controller.signal,
  timeoutMs: 5_000
});
setTimeout(() => controller.abort(), 25);
const cancelled = await cancelledPromise;
delete process.env.CADENCE_FAKE_AI_DELAY_MS;
assert.equal(cancelled.ok, false);
assert.equal(cancelled.cancelled, true);
