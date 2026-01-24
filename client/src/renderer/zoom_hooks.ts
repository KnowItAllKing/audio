/**
 * Zoom-oriented seams (Phase 5)
 *
 * This file intentionally contains NO Zoom SDK imports or calls.
 * It defines the conceptual interfaces and events we'd expect from a future
 * Zoom integration so the rest of the app has a stable place to plug it in.
 *
 * Mapping idea (future):
 * - `ZoomSpeakerEvent.participantId` would map to a speaker label for transcript segments.
 * - Today we only have `TranscriptSegment.stream_tags` in the wire protocol.
 * - In a future protocol revision we can add explicit speaker fields like:
 *     speaker_id / speaker_name / diarization segments
 *   and then merge speaker events with ASR timestamps.
 */

export interface ZoomParticipant {
  id: string;
  name: string;
}

export interface ZoomSpeakerEvent {
  participantId: string;
  /** epoch ms or monotonic ms (implementation-defined) */
  startTime: number;
  /** if absent, speaker is currently active */
  endTime?: number;
}

