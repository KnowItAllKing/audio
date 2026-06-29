import assert from "node:assert/strict";

import {
  SPEAKER_MEMORY_STORAGE_KEY,
  loadSpeakerMemoryProfiles,
  normalizeSpeakerMemoryProfiles,
  saveSpeakerMemoryProfiles
} from "./SpeakerMemoryStore";

class MemoryStorage {
  values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

const storage = new MemoryStorage();

const saved = saveSpeakerMemoryProfiles(
  [
    {
      profile_id: "bad",
      name: "Bad",
      fingerprint: []
    },
    {
      profile_id: "alice",
      name: " Alice ",
      fingerprint: [0, 1, Number.NaN, 2],
      sample_count: 2,
      created_at: 10,
      updated_at: 20
    }
  ],
  storage
);

assert.equal(saved.length, 1);
assert.deepEqual(saved[0].fingerprint, [0, 1, 2]);
assert.equal(saved[0].name, "Alice");

const raw = storage.getItem(SPEAKER_MEMORY_STORAGE_KEY);
assert.ok(raw);
assert.deepEqual(loadSpeakerMemoryProfiles(storage), saved);

storage.setItem(SPEAKER_MEMORY_STORAGE_KEY, "{not json");
assert.deepEqual(loadSpeakerMemoryProfiles(storage), []);

assert.deepEqual(
  normalizeSpeakerMemoryProfiles({ profiles: [{ profile_id: "bob", name: "Bob", fingerprint: [1] }] })[0].name,
  "Bob"
);
