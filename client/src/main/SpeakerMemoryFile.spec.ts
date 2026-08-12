import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { readSpeakerMemoryFile, writeSpeakerMemoryFile } from "./SpeakerMemoryFile";

const directory = await mkdtemp(join(tmpdir(), "cadence-speaker-memory-"));
const filePath = join(directory, "profiles.json");

try {
  assert.deepEqual(await readSpeakerMemoryFile(filePath), { found: false });

  const profiles = {
    version: 1,
    profiles: [{ profile_id: "alice", name: "Alice", fingerprint: [0, 1] }]
  };
  await writeSpeakerMemoryFile(filePath, profiles);
  assert.deepEqual(await readSpeakerMemoryFile(filePath), { found: true, value: profiles });

  const replacement = { version: 1, profiles: [] };
  await writeSpeakerMemoryFile(filePath, replacement);
  assert.deepEqual(await readSpeakerMemoryFile(filePath), { found: true, value: replacement });

  await writeFile(filePath, "{bad json", "utf8");
  await assert.rejects(readSpeakerMemoryFile(filePath), SyntaxError);
} finally {
  await rm(directory, { recursive: true, force: true });
}
