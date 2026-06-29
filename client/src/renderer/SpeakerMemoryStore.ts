export type SpeakerMemoryProfile = {
  profile_id: string;
  name: string;
  fingerprint: number[];
  sample_count: number;
  created_at: number;
  updated_at: number;
};

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export const SPEAKER_MEMORY_STORAGE_KEY = "cadence.speakerMemoryProfiles.v1";

export function loadSpeakerMemoryProfiles(storage: StorageLike = globalThis.localStorage): SpeakerMemoryProfile[] {
  try {
    const raw = storage.getItem(SPEAKER_MEMORY_STORAGE_KEY);
    if (!raw) return [];
    return normalizeSpeakerMemoryProfiles(JSON.parse(raw));
  } catch {
    return [];
  }
}

export function saveSpeakerMemoryProfiles(
  profiles: unknown,
  storage: StorageLike = globalThis.localStorage
): SpeakerMemoryProfile[] {
  const normalized = normalizeSpeakerMemoryProfiles(profiles);
  try {
    if (normalized.length === 0) {
      storage.removeItem(SPEAKER_MEMORY_STORAGE_KEY);
    } else {
      storage.setItem(
        SPEAKER_MEMORY_STORAGE_KEY,
        JSON.stringify({ version: 1, profiles: normalized })
      );
    }
  } catch {
    // Storage may be unavailable in unusual browser contexts.
  }
  return normalized;
}

export function normalizeSpeakerMemoryProfiles(value: unknown): SpeakerMemoryProfile[] {
  const items = Array.isArray(value)
    ? value
    : isRecord(value) && Array.isArray(value.profiles)
      ? value.profiles
      : [];
  const profiles: SpeakerMemoryProfile[] = [];

  for (const item of items) {
    if (!isRecord(item)) continue;
    const profileId = readString(item.profile_id);
    const name = readString(item.name);
    const fingerprint = Array.isArray(item.fingerprint)
      ? item.fingerprint.filter(isFiniteNumber)
      : [];
    if (!profileId || !name || fingerprint.length === 0) continue;

    profiles.push({
      profile_id: profileId,
      name,
      fingerprint,
      sample_count: Math.max(1, readNumber(item.sample_count, 1)),
      created_at: readNumber(item.created_at, 0),
      updated_at: readNumber(item.updated_at, 0)
    });
  }

  return profiles.sort((a, b) => a.name.localeCompare(b.name));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function readNumber(value: unknown, fallback: number): number {
  return isFiniteNumber(value) ? value : fallback;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
