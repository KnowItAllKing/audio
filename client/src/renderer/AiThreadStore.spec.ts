import assert from "node:assert/strict";

import {
  AI_THREAD_STORAGE_KEY,
  loadAiThreads,
  normalizeAiThreads,
  saveAiThreads,
  titleFromQuestion,
  transcriptFingerprint,
  type AiThread
} from "./AiThreadStore";

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

const thread: AiThread = {
  id: "ui-1",
  provider: "codex",
  model: "",
  remoteThreadId: "remote-1",
  title: "Release date",
  createdAt: 10,
  updatedAt: 20,
  lastTranscriptFingerprint: "abc",
  lastTranscriptChars: 42,
  messages: [{ id: "m-1", role: "user", text: "When?", createdAt: 20 }]
};

const storage = new MemoryStorage();
assert.deepEqual(saveAiThreads([thread], storage), [thread]);
assert.deepEqual(loadAiThreads(storage), [thread]);
assert.ok(storage.getItem(AI_THREAD_STORAGE_KEY)?.includes("remote-1"));
assert.equal(storage.getItem(AI_THREAD_STORAGE_KEY)?.includes("secret transcript"), false);

storage.setItem(AI_THREAD_STORAGE_KEY, "not json");
assert.deepEqual(loadAiThreads(storage), []);
assert.deepEqual(normalizeAiThreads([{ id: "bad", provider: "other" }]), []);

assert.equal(transcriptFingerprint("Alice: hello"), transcriptFingerprint("Alice: hello"));
assert.notEqual(transcriptFingerprint("Alice: hello"), transcriptFingerprint("Bob: hello"));
assert.equal(titleFromQuestion("  What   did Alice decide?  "), "What did Alice decide?");
assert.equal(titleFromQuestion("x".repeat(60)).length, 46);
