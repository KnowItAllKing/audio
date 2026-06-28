import assert from "node:assert/strict";

import { MicInputFilter, getAudioStats } from "./MicInputFilter";

const quiet = new Float32Array([0.001, -0.001, 0.001, -0.001]);
const loud = new Float32Array([0.08, -0.08, 0.06, -0.06]);

{
  const stats = getAudioStats(loud);
  assert.equal(Number(stats.peak.toFixed(2)), 0.08);
  assert.ok(stats.rms > 0.06);
}

{
  const filter = new MicInputFilter();
  const decision = filter.decide(quiet, 0, {
    muted: true,
    enabled: true,
    thresholdRms: 0.01
  });
  assert.equal(decision.shouldSend, false);
  assert.equal(decision.reason, "muted");
}

{
  const filter = new MicInputFilter();
  const decision = filter.decide(quiet, 0, {
    muted: false,
    enabled: false,
    thresholdRms: 0.01
  });
  assert.equal(decision.shouldSend, true);
  assert.equal(decision.reason, "disabled");
}

{
  const filter = new MicInputFilter();
  const decision = filter.decide(quiet, 0, {
    muted: false,
    enabled: true,
    thresholdRms: 0.01
  });
  assert.equal(decision.shouldSend, false);
  assert.equal(decision.reason, "noise");
}

{
  const filter = new MicInputFilter();
  const speech = filter.decide(loud, 100, {
    muted: false,
    enabled: true,
    thresholdRms: 0.01,
    hangoverMs: 500
  });
  const hangover = filter.decide(quiet, 400, {
    muted: false,
    enabled: true,
    thresholdRms: 0.01,
    hangoverMs: 500
  });
  const expired = filter.decide(quiet, 700, {
    muted: false,
    enabled: true,
    thresholdRms: 0.01,
    hangoverMs: 500
  });

  assert.equal(speech.shouldSend, true);
  assert.equal(speech.reason, "speech");
  assert.equal(hangover.shouldSend, true);
  assert.equal(hangover.reason, "hangover");
  assert.equal(expired.shouldSend, false);
  assert.equal(expired.reason, "noise");
}
