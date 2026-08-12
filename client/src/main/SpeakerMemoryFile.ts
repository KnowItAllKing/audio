import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export const SPEAKER_MEMORY_FILE_NAME = "speaker-memory-profiles.json";

const MAX_SPEAKER_MEMORY_BYTES = 5 * 1024 * 1024;

export type SpeakerMemoryFileContents = {
  found: boolean;
  value?: unknown;
};

export async function readSpeakerMemoryFile(filePath: string): Promise<SpeakerMemoryFileContents> {
  try {
    const text = await readFile(filePath, "utf8");
    return { found: true, value: JSON.parse(text) };
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return { found: false };
    throw error;
  }
}

export async function writeSpeakerMemoryFile(filePath: string, value: unknown): Promise<void> {
  const serialized = JSON.stringify(value, null, 2);
  if (typeof serialized !== "string") throw new Error("Speaker memory is not serializable.");
  const text = `${serialized}\n`;
  if (Buffer.byteLength(text, "utf8") > MAX_SPEAKER_MEMORY_BYTES) {
    throw new Error("Speaker memory exceeds the 5 MB storage limit.");
  }

  await mkdir(dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, text, { encoding: "utf8", mode: 0o600 });
    await rename(temporaryPath, filePath);
  } catch (error) {
    await rm(temporaryPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
