/**
 * Zoom-oriented seams (Phase 5)
 *
 * This file intentionally contains NO Zoom SDK imports or calls.
 * It defines the conceptual interfaces and events we'd expect from a future
 * Zoom integration so the rest of the app has a stable place to plug it in.
 *
 * Mapping idea:
 * - Send participant lists as `control:metadata`.
 * - Send active-speaker intervals as `control:speaker_activity`.
 * - The server maps those timestamps onto transcript utterances and emits
 *   `speaker_id`, `speaker_label`, `speaker_source`, and `speaker_confidence`.
 */

export interface ZoomParticipant {
  id: string;
  name: string;
}

export interface ZoomSpeakerEvent {
  participantId: string;
  participantName?: string;
  /** epoch ms or monotonic ms (implementation-defined) */
  startTime: number;
  /** if absent, speaker is currently active */
  endTime?: number;
}

export function zoomSpeakerEventToControlPayload(event: ZoomSpeakerEvent): Record<string, unknown> {
  return {
    participant_id: event.participantId,
    participant_name: event.participantName,
    stream_id: "system",
    start_timestamp_ms: event.startTime,
    end_timestamp_ms: event.endTime
  };
}
