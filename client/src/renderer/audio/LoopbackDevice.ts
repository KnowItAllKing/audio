export type AudioInputDeviceLike = {
  deviceId: string;
  label: string;
};

const LOOPBACK_LABEL_PARTS = [
  "blackhole",
  "loopback",
  "soundflower",
  "vb-audio",
  "vb cable",
  "vb-cable",
  "cable output",
  "stereo mix",
  "what u hear",
  "monitor of"
];

export function isLikelyLoopbackDevice(device: AudioInputDeviceLike): boolean {
  const label = normalizeLabel(device.label);
  if (!label) return false;
  return LOOPBACK_LABEL_PARTS.some((part) => label.includes(part));
}

export function chooseLikelyLoopbackDevice<T extends AudioInputDeviceLike>(devices: T[]): T | undefined {
  return devices.find(isLikelyLoopbackDevice);
}

export function chooseLikelyMicDevice<T extends AudioInputDeviceLike>(
  devices: T[],
  excludeDeviceId?: string
): T | undefined {
  return devices.find((device) => {
    if (excludeDeviceId && device.deviceId === excludeDeviceId) return false;
    return !isLikelyLoopbackDevice(device);
  });
}

function normalizeLabel(label: string): string {
  return label.trim().toLowerCase();
}
