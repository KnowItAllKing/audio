#!/usr/bin/env node

import { WebSocketServer } from "ws";

const server = new WebSocketServer({ port: 8765 });

function segment(id, offset, speakerId, speakerLabel, text, streamTags = ["system"]) {
  const now = Date.now() / 1000;
  return {
    id,
    start_sec: now + offset,
    end_sec: now + offset + 0.8,
    text,
    stream_tags: streamTags,
    speaker_id: speakerId,
    speaker_label: speakerLabel,
    speaker_source: speakerId === "local:mic" ? "manual" : "diarization",
    speaker_confidence: 0.94,
    is_final: true,
    full_context_available: true,
    final_reason: "qa_fixture"
  };
}

server.on("connection", (socket) => {
  const initial = [
    segment("processed-1", 0, "speaker:alice", "Alice", "The release is Friday."),
    segment("processed-2", 1, "speaker:bob", "Bob", "I will send the launch brief."),
    segment("processed-3", 2, "speaker:alice", "Alice", "Please finish it by Thursday.")
  ];
  socket.send(JSON.stringify({ type: "transcript_update", session_id: "qa", layer: "processed", segments: initial }));
  socket.send(JSON.stringify({ type: "transcript_update", session_id: "qa", layer: "processed", segments: initial }));
  socket.send(
    JSON.stringify({
      type: "transcript_update",
      session_id: "qa",
      layer: "raw",
      segments: [
        segment("raw-mic", 0, "local:mic", "You", "Can everyone hear me?", ["mic"]),
        segment("raw-system-echo", 0.04, "system:unknown", "Meeting audio", "Can everyone hear me?", ["system"])
      ]
    })
  );

  const updateTimer = setTimeout(() => {
    if (socket.readyState === socket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "transcript_update",
          session_id: "qa",
          layer: "processed",
          segments: [segment("processed-4", 3, "speaker:bob", "Bob", "The owner is Morgan.")]
        })
      );
    }
  }, 12_000);

  socket.on("message", (data) => {
    try {
      const message = JSON.parse(data.toString("utf8"));
      if (message.type === "control" && message.command === "stop") {
        socket.send(JSON.stringify({ type: "control", session_id: message.session_id, command: "stopped" }));
      }
    } catch {
      // This fixture ignores malformed client input.
    }
  });
  socket.on("close", () => clearTimeout(updateTimer));
});

process.stdout.write("fake transcript server listening on ws://localhost:8765\n");
