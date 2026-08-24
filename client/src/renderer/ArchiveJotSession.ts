import { transcriptFingerprint } from "./AiThreadStore";

const MAX_TRACKED_JOTS = 200;

/** Only final processed segments inside the trailing window feed the extractor. */
export function selectJotWindowSegments<T extends { start_sec: number; is_final: boolean }>(
  segments: Iterable<T>,
  nowSec: number,
  windowSec: number
): T[] {
  const cutoff = nowSec - windowSec;
  return [...segments].filter((segment) => segment.is_final && segment.start_sec >= cutoff);
}

/**
 * Per-meeting extraction state: dirty marking from final processed segments,
 * fingerprint-based skip across unchanged windows, and the session-local list
 * of jotted texts used to dedupe overlapping windows in the prompt.
 */
export class ArchiveJotSession {
  readonly batch: string;
  readonly sessionId: string;
  jottedCount = 0;

  private dirty = false;
  private lastFingerprint = "";
  private jotted: string[] = [];

  constructor(batch: string, sessionId: string) {
    this.batch = batch;
    this.sessionId = sessionId;
  }

  markDirty(): void {
    this.dirty = true;
  }

  jottedTexts(): string[] {
    return [...this.jotted];
  }

  shouldExtract(windowText: string): boolean {
    if (!this.dirty || !windowText.trim()) return false;
    return transcriptFingerprint(windowText) !== this.lastFingerprint;
  }

  beginExtraction(windowText: string): void {
    this.dirty = false;
    this.lastFingerprint = transcriptFingerprint(windowText);
  }

  recordJotted(texts: string[]): void {
    this.jotted.push(...texts);
    if (this.jotted.length > MAX_TRACKED_JOTS) {
      this.jotted.splice(0, this.jotted.length - MAX_TRACKED_JOTS);
    }
    this.jottedCount += texts.length;
  }

  /** Re-arm the session so an unchanged window is retried on the next tick. */
  recordFailure(): void {
    this.dirty = true;
    this.lastFingerprint = "";
  }
}
