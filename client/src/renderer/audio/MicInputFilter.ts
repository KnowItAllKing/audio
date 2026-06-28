export type MicInputFilterConfig = {
  muted: boolean;
  enabled: boolean;
  thresholdRms: number;
  hangoverMs?: number;
};

export type MicInputFilterDecision = {
  shouldSend: boolean;
  rms: number;
  peak: number;
  reason: "muted" | "disabled" | "speech" | "hangover" | "noise";
};

const DEFAULT_HANGOVER_MS = 650;

export class MicInputFilter {
  private speechUntilMs = 0;

  reset(): void {
    this.speechUntilMs = 0;
  }

  decide(input: Float32Array, nowMs: number, config: MicInputFilterConfig): MicInputFilterDecision {
    const stats = getAudioStats(input);
    if (config.muted) {
      this.speechUntilMs = 0;
      return { ...stats, shouldSend: false, reason: "muted" };
    }

    if (!config.enabled) {
      return { ...stats, shouldSend: true, reason: "disabled" };
    }

    const thresholdRms = Math.max(0, config.thresholdRms);
    const hasSpeech = stats.rms >= thresholdRms || stats.peak >= thresholdRms * 4;
    if (hasSpeech) {
      this.speechUntilMs = nowMs + (config.hangoverMs ?? DEFAULT_HANGOVER_MS);
      return { ...stats, shouldSend: true, reason: "speech" };
    }

    if (this.speechUntilMs > 0 && nowMs <= this.speechUntilMs) {
      return { ...stats, shouldSend: true, reason: "hangover" };
    }

    return { ...stats, shouldSend: false, reason: "noise" };
  }
}

export function getAudioStats(input: Float32Array): { rms: number; peak: number } {
  if (input.length === 0) return { rms: 0, peak: 0 };

  let sumSquares = 0;
  let peak = 0;
  for (let i = 0; i < input.length; i++) {
    const abs = Math.abs(input[i]);
    peak = Math.max(peak, abs);
    sumSquares += input[i] * input[i];
  }

  return {
    rms: Math.sqrt(sumSquares / input.length),
    peak
  };
}
