import assert from "node:assert/strict";

import {
  chooseLikelyLoopbackDevice,
  chooseLikelyMicDevice,
  isLikelyLoopbackDevice
} from "./LoopbackDevice";

const devices = [
  { deviceId: "default", label: "Default - MacBook Pro Microphone" },
  { deviceId: "blackhole", label: "BlackHole 2ch" },
  { deviceId: "usb", label: "USB Audio Microphone" }
];

assert.equal(isLikelyLoopbackDevice(devices[0]), false);
assert.equal(isLikelyLoopbackDevice(devices[1]), true);
assert.equal(isLikelyLoopbackDevice({ deviceId: "cable", label: "CABLE Output (VB-Audio Virtual Cable)" }), true);
assert.equal(isLikelyLoopbackDevice({ deviceId: "empty", label: "" }), false);

assert.equal(chooseLikelyLoopbackDevice(devices)?.deviceId, "blackhole");
assert.equal(chooseLikelyMicDevice(devices, "blackhole")?.deviceId, "default");
assert.equal(chooseLikelyMicDevice([devices[1]], "blackhole"), undefined);
