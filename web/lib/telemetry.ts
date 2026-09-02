/**
 * Fixed-capacity ring of recent samples, with percentiles.
 *
 * p50 and p95 rather than a mean: inference latency is long-tailed (GC pauses,
 * shader recompiles, thermal throttling) and a mean hides exactly the stutter a
 * user notices. A rolling window rather than a lifetime average so the numbers
 * respond when the model or resolution changes.
 */
export class RollingStats {
  private readonly buffer: Float64Array;
  private index = 0;
  private filled = 0;

  constructor(readonly capacity = 60) {
    this.buffer = new Float64Array(capacity);
  }

  push(value: number): void {
    this.buffer[this.index] = value;
    this.index = (this.index + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;
  }

  get count(): number {
    return this.filled;
  }

  reset(): void {
    this.index = 0;
    this.filled = 0;
  }

  percentile(p: number): number {
    if (this.filled === 0) return 0;
    const sorted = Array.from(this.buffer.subarray(0, this.filled)).sort((a, b) => a - b);
    // Nearest-rank; with a 60-sample window the interpolation nuance is noise.
    const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
    return sorted[rank];
  }

  get p50(): number {
    return this.percentile(50);
  }

  get p95(): number {
    return this.percentile(95);
  }

  get mean(): number {
    if (this.filled === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.filled; i++) sum += this.buffer[i];
    return sum / this.filled;
  }
}

/** Frames per second over a sliding wall-clock window, independent of the stats ring. */
export class FpsCounter {
  private timestamps: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  tick(now = performance.now()): void {
    this.timestamps.push(now);
    const cutoff = now - this.windowMs;
    while (this.timestamps.length && this.timestamps[0] < cutoff) this.timestamps.shift();
  }

  get fps(): number {
    if (this.timestamps.length < 2) return 0;
    const span = this.timestamps[this.timestamps.length - 1] - this.timestamps[0];
    return span > 0 ? ((this.timestamps.length - 1) * 1000) / span : 0;
  }

  reset(): void {
    this.timestamps = [];
  }
}
