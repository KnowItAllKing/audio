#!/usr/bin/env node

const args = process.argv.slice(2);

if (args.includes("--version")) {
  process.stdout.write("cadence-fake-cli 1.0\n");
  process.exit(0);
}

let prompt = "";
for await (const chunk of process.stdin) prompt += chunk.toString("utf8");

const delayMs = Number.parseInt(process.env.CADENCE_FAKE_AI_DELAY_MS || "0", 10) || 0;
if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));

const isCodex = args[0] === "exec";
const transcriptIncluded = prompt.includes("BEGIN PROCESSED TRANSCRIPT");
const responseOverride = process.env.CADENCE_FAKE_AI_RESPONSE;
if (isCodex) {
  process.stdout.write(`${JSON.stringify({ type: "thread.started", thread_id: "fake-codex-thread" })}\n`);
  process.stdout.write(
    `${JSON.stringify({
      type: "item.completed",
      item: {
        type: "agent_message",
        text: responseOverride ?? `Fake Codex answer · transcript ${transcriptIncluded ? "included" : "unchanged"}`
      }
    })}\n`
  );
} else {
  const idFlag = args.includes("--resume") ? "--resume" : "--session-id";
  const idIndex = args.indexOf(idFlag);
  const sessionId = idIndex >= 0 ? args[idIndex + 1] : "fake-claude-thread";
  process.stdout.write(
    JSON.stringify({
      session_id: sessionId,
      result: responseOverride ?? `Fake Claude answer · transcript ${transcriptIncluded ? "included" : "unchanged"}`,
      is_error: false
    })
  );
}
