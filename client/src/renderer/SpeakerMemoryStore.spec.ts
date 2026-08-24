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

// Exemplar banks and auto-learned profiles survive normalization.
const banked = normalizeSpeakerMemoryProfiles([
  {
    profile_id: "john",
    name: "John",
    fingerprint: [1, 0],
    fingerprints: [
      [1, 0],
      [0.7, 0.7],
      ["bad"],
      []
    ],
    kind: "named",
    sample_count: 3
  },
  {
    profile_id: "auto-1",
    name: "Speaker 1",
    fingerprint: [0, 1],
    kind: "auto",
    updated_at: 5
  }
]);
assert.equal(banked.length, 2);
const john = banked.find((profile) => profile.profile_id === "john");
assert.ok(john);
assert.deepEqual(john.fingerprints, [
  [1, 0],
  [0.7, 0.7]
]);
assert.equal(john.kind, "named");
const autoProfile = banked.find((profile) => profile.profile_id === "auto-1");
assert.ok(autoProfile);
assert.equal(autoProfile.kind, "auto");
assert.equal(autoProfile.fingerprints, undefined);

// Auto profiles are capped to the most recently heard voices; named survive.
const flood = normalizeSpeakerMemoryProfiles([
  { profile_id: "named-1", name: "Zoe", fingerprint: [1], kind: "named", updated_at: 0 },
  ...Array.from({ length: 40 }, (_, index) => ({
    profile_id: `auto-${index}`,
    name: `Speaker ${index}`,
    fingerprint: [1],
    kind: "auto",
    updated_at: index
  }))
]);
assert.equal(flood.filter((profile) => profile.kind === "auto").length, 32);
assert.ok(flood.some((profile) => profile.profile_id === "named-1"));
assert.ok(flood.some((profile) => profile.profile_id === "auto-39"));
assert.ok(!flood.some((profile) => profile.profile_id === "auto-0"));
