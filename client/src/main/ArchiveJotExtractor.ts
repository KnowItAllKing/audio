import { access, constants } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

import { AiCliRunError, resolveExecutable, runAiCliTurn, runProcess } from "./AiCliRunner";
import {
  buildExtractionQuestion,
  normalizeJotText,
  parseExtractedItems,
  type ArchiveCliStatus,
  type ArchiveExtractedItem,
  type ArchiveExtractRequest,
  type ArchiveExtractResult
} from "../shared/ArchiveJot";
import type { AiRunRequest } from "../shared/AiTypes";

const BATCH_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const ID_PATTERN = /^[a-zA-Z0-9._:-]{1,160}$/;
const MAX_BATCH_CHARS = 100;
const MAX_WINDOW_CHARS = 100_000;
const MAX_JOTTED_TEXTS = 500;
const DEFAULT_JOT_TIMEOUT_MS = 20_000;
const MAX_JOTTED_MEMORY = 500;

export type ArchiveExtractionOptions = {
  aiExecutable: string;
  archiveExecutable: string;
  cwd: string;
  signal?: AbortSignal;
  timeoutMs?: number;
  jotTimeoutMs?: number;
  /** Main-process dedupe memory for this batch; pushed texts are added to it. */
  alreadyJotted?: Set<string>;
  /** Extra child-process environment (used by tests to steer the fixtures). */
  env?: NodeJS.ProcessEnv;
};

export function validateArchiveExtractRequest(request: ArchiveExtractRequest): void {
  if (!request || (request.provider !== "codex" && request.provider !== "claude")) {
    throw new AiCliRunError("Unsupported extraction provider.");
  }
  if (!ID_PATTERN.test(request.requestId)) throw new AiCliRunError("Invalid extraction request identifier.");
  if (!ID_PATTERN.test(request.sessionId)) throw new AiCliRunError("Invalid session identifier.");
  if (request.batch.length > MAX_BATCH_CHARS || !BATCH_PATTERN.test(request.batch)) {
    throw new AiCliRunError("Invalid batch name (kebab-case required).");
  }
  if (!request.transcript.trim()) throw new AiCliRunError("The extraction window is empty.");
  if (request.transcript.length > MAX_WINDOW_CHARS) {
    throw new AiCliRunError("The extraction window is too large to send.");
  }
  if (
    !Array.isArray(request.jottedTexts) ||
    request.jottedTexts.length > MAX_JOTTED_TEXTS ||
    request.jottedTexts.some((text) => typeof text !== "string")
  ) {
    throw new AiCliRunError("Invalid jotted-text list.");
  }
}

export function buildJotArgs(text: string, batch: string, sessionId: string): string[] {
  // "--" keeps a text starting with "-" from being parsed as a flag by the Go CLI.
  return ["jot", "--batch", batch, "--source", `cadence:${sessionId}`, "--", text];
}

export async function resolveArchiveExecutable(
  env: NodeJS.ProcessEnv = process.env
): Promise<string | undefined> {
  const resolved = await resolveExecutable("archive", env.CADENCE_ARCHIVE_PATH, env);
  if (resolved) return resolved;
  // `go install` targets ~/go/bin, which a GUI-launched Electron app won't have on PATH.
  const goBinary = join(homedir(), "go", "bin", "archive");
  try {
    await access(goBinary, constants.X_OK);
    return goBinary;
  } catch {
    return undefined;
  }
}

export async function getArchiveCliStatus(): Promise<ArchiveCliStatus> {
  const path = await resolveArchiveExecutable();
  if (!path) return { available: false, error: "archive CLI not found" };
  try {
    // The archive CLI has a `version` subcommand, not a `--version` flag.
    const output = await runProcess(path, ["version"], "", { cwd: homedir(), timeoutMs: 5_000 });
    const version = (output.stdout || output.stderr).trim().split(/\r?\n/)[0];
    return {
      available: output.exitCode === 0,
      path,
      version: version || undefined,
      error: output.exitCode === 0 ? undefined : `version check exited ${output.exitCode}`
    };
  } catch (error) {
    return { available: false, path, error: error instanceof Error ? error.message : String(error) };
  }
}

function rememberJotted(memory: Set<string> | undefined, key: string): void {
  if (!memory) return;
  memory.add(key);
  if (memory.size > MAX_JOTTED_MEMORY) {
    const oldest = memory.values().next();
    if (!oldest.done) memory.delete(oldest.value);
  }
}

export async function runArchiveExtraction(
  request: ArchiveExtractRequest,
  options: ArchiveExtractionOptions
): Promise<ArchiveExtractResult> {
  const startedAt = Date.now();
  const requestId = typeof request?.requestId === "string" ? request.requestId : "invalid";
  const jotted: ArchiveExtractedItem[] = [];
  let skipped = 0;
  try {
    validateArchiveExtractRequest(request);

    const aiRequest: AiRunRequest = {
      requestId: request.requestId,
      uiThreadId: "archive-extractor",
      provider: request.provider,
      model: request.model,
      transcript: request.transcript,
      question: buildExtractionQuestion(request.jottedTexts),
      syncOnly: false
    };
    const aiResult = await runAiCliTurn(aiRequest, {
      executable: options.aiExecutable,
      cwd: options.cwd,
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      env: options.env
    });
    if (!aiResult.ok || !aiResult.response) {
      return {
        ok: false,
        requestId,
        jotted,
        skipped,
        error: aiResult.error || "The extraction model returned no output.",
        cancelled: aiResult.cancelled,
        durationMs: Date.now() - startedAt
      };
    }

    const seenKeys = new Set<string>(request.jottedTexts.map(normalizeJotText));
    for (const key of options.alreadyJotted ?? []) seenKeys.add(key);

    let error: string | undefined;
    for (const item of parseExtractedItems(aiResult.response)) {
      const key = normalizeJotText(item.text);
      if (seenKeys.has(key)) {
        skipped += 1;
        continue;
      }
      if (options.signal?.aborted) {
        return { ok: false, requestId, jotted, skipped, error: "Extraction cancelled.", cancelled: true, durationMs: Date.now() - startedAt };
      }
      const output = await runProcess(
        options.archiveExecutable,
        buildJotArgs(item.text, request.batch, request.sessionId),
        "",
        {
          cwd: options.cwd,
          signal: options.signal,
          timeoutMs: options.jotTimeoutMs ?? DEFAULT_JOT_TIMEOUT_MS,
          env: options.env
        }
      );
      if (output.exitCode !== 0) {
        // The archive CLI is broken or misconfigured; don't hammer it with the rest.
        error = (output.stderr.trim() || `archive exited ${output.exitCode}`).slice(0, 500);
        break;
      }
      seenKeys.add(key);
      rememberJotted(options.alreadyJotted, key);
      jotted.push(item);
    }

    return { ok: !error, requestId, jotted, skipped, error, durationMs: Date.now() - startedAt };
  } catch (caught) {
    const runError = caught instanceof AiCliRunError ? caught : new AiCliRunError(String(caught));
    return {
      ok: false,
      requestId,
      jotted,
      skipped,
      error: runError.message,
      cancelled: runError.cancelled,
      durationMs: Date.now() - startedAt
    };
  }
}
