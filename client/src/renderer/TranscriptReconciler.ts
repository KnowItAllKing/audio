export type ReconciliableTranscriptSegment = {
  id: string;
  start_sec: number;
  end_sec: number;
  text: string;
  stream_tags: readonly string[];
  speaker_source?: string;
  speaker_confidence?: number;
  is_final?: boolean;
  full_context_available?: boolean;
};

const SOURCE_QUALITY: Record<string, number> = {
  unknown: 0,
  stream: 1,
  zoom: 4,
  diarization: 5,
  manual: 6,
  memory: 7
};

const ASR_FILLER_WORDS = new Set(["a", "an", "the", "uh", "um", "erm", "hmm"]);

function normalizedWords(text: string): string[] {
  return text
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}']+/gu, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function textMatches(left: string, right: string): boolean {
  const leftWords = normalizedWords(left);
  const rightWords = normalizedWords(right);
  if (!leftWords.length || !rightWords.length) return false;
  if (leftWords.join(" ") === rightWords.join(" ")) return true;
  return (
    leftWords.filter((word) => !ASR_FILLER_WORDS.has(word)).join(" ") ===
    rightWords.filter((word) => !ASR_FILLER_WORDS.has(word)).join(" ")
  );
}

function sourceKind(segment: ReconciliableTranscriptSegment): "mic" | "system" | "mixed" | "other" {
  const mic = segment.stream_tags.includes("mic");
  const system = segment.stream_tags.includes("system");
  if (mic && system) return "mixed";
  if (mic) return "mic";
  if (system) return "system";
  return "other";
}

function isCrossSourcePair(
  left: ReconciliableTranscriptSegment,
  right: ReconciliableTranscriptSegment
): boolean {
  const leftKind = sourceKind(left);
  const rightKind = sourceKind(right);
  return (leftKind === "mic" && rightKind === "system") || (leftKind === "system" && rightKind === "mic");
}

function timingMatches(
  left: ReconciliableTranscriptSegment,
  right: ReconciliableTranscriptSegment
): boolean {
  const leftStart = Math.min(left.start_sec, left.end_sec);
  const leftEnd = Math.max(left.start_sec, left.end_sec);
  const rightStart = Math.min(right.start_sec, right.end_sec);
  const rightEnd = Math.max(right.start_sec, right.end_sec);
  if (leftStart > rightEnd + 0.75 || rightStart > leftEnd + 0.75) return false;

  const leftDuration = leftEnd - leftStart;
  const rightDuration = rightEnd - rightStart;
  const midpointDistance = Math.abs((leftStart + leftEnd - rightStart - rightEnd) / 2);
  const midpointTolerance = Math.max(1, Math.min(2.5, Math.max(leftDuration, rightDuration) * 0.5 + 0.5));
  return midpointDistance <= midpointTolerance;
}

function preferenceScore(segment: ReconciliableTranscriptSegment): number {
  const sourceScore = SOURCE_QUALITY[segment.speaker_source ?? "unknown"] ?? 0;
  const systemScore = sourceKind(segment) === "system" ? 2 : 0;
  const finalScore = segment.is_final ? 1 : 0;
  const contextScore = segment.full_context_available ? 0.5 : 0;
  const confidenceScore = Math.max(0, Math.min(1, Number(segment.speaker_confidence ?? 0)));
  return sourceScore * 10 + systemScore + finalScore + contextScore + confidenceScore;
}

function isDuplicate(
  left: ReconciliableTranscriptSegment,
  right: ReconciliableTranscriptSegment
): boolean {
  return isCrossSourcePair(left, right) && timingMatches(left, right) && textMatches(left.text, right.text);
}

/**
 * Remove acoustic echo captured by both the microphone and meeting-loopback
 * streams. Same-stream repetition is deliberately preserved.
 */
export function reconcileCrossStreamDuplicates<T extends ReconciliableTranscriptSegment>(
  segments: Iterable<T>
): T[] {
  const ordered = [...segments].sort((left, right) => {
    if (left.start_sec !== right.start_sec) return left.start_sec - right.start_sec;
    return left.id.localeCompare(right.id);
  });
  const reconciled: T[] = [];

  for (const candidate of ordered) {
    const duplicateIndex = reconciled.findIndex((existing) => isDuplicate(existing, candidate));
    if (duplicateIndex < 0) {
      reconciled.push(candidate);
      continue;
    }

    const existing = reconciled[duplicateIndex];
    if (preferenceScore(candidate) > preferenceScore(existing)) {
      reconciled[duplicateIndex] = candidate;
    }
  }

  return reconciled.sort((left, right) => {
    if (left.start_sec !== right.start_sec) return left.start_sec - right.start_sec;
    return left.id.localeCompare(right.id);
  });
}
