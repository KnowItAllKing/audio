export type CleanTranscriptSegment = {
  id: string;
  start_sec: number;
  text: string;
  speaker_id?: string;
  speaker_label?: string;
};

function normalizeText(text: string): string {
  return String(text).trim().replace(/\s+/g, " ");
}

function joinText(left: string, right: string): string {
  if (!left) return right;
  if (!right) return left;
  if (/^[,.;!?:)\]}]/.test(right)) return `${left.trimEnd()}${right}`;
  return `${left.trimEnd()} ${right.trimStart()}`;
}

/** Format the canonical transcript as readable speaker turns with no metadata. */
export function formatCleanTranscript(segments: Iterable<CleanTranscriptSegment>): string {
  const ordered = [...segments].sort((left, right) => {
    if (left.start_sec !== right.start_sec) return left.start_sec - right.start_sec;
    return left.id.localeCompare(right.id);
  });
  const turns: Array<{ speakerKey: string; speakerLabel: string; text: string }> = [];

  for (const segment of ordered) {
    const text = normalizeText(segment.text);
    if (!text) continue;
    const speakerLabel = normalizeText(segment.speaker_label || segment.speaker_id || "Unknown");
    const speakerKey = normalizeText(segment.speaker_id || speakerLabel).toLocaleLowerCase();
    const previous = turns.at(-1);
    if (previous && previous.speakerKey === speakerKey) {
      previous.text = joinText(previous.text, text);
    } else {
      turns.push({ speakerKey, speakerLabel, text });
    }
  }

  return turns.map((turn) => `${turn.speakerLabel}: ${turn.text}`).join("\n\n");
}
