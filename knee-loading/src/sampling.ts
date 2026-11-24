import { DEFAULT_PARAMETERS, PARAMETERS, PARAM_STRIDE } from './mechanics.ts';

/** Deterministic stratification of each independent parameter range. */
export function sampleParameters(count: number, seed = 0xf8ea9e83): Float32Array {
  if (!Number.isSafeInteger(count) || count < 2) throw new RangeError('Specify at least two parameter sets.');
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let random = Math.imul(state ^ (state >>> 15), state | 1);
    random ^= random + Math.imul(random ^ (random >>> 7), random | 61);
    return ((random ^ (random >>> 14)) >>> 0) / 4_294_967_296;
  };

  const samples = new Float32Array(PARAM_STRIDE * count);
  samples.set(DEFAULT_PARAMETERS);
  const strata = Array.from({ length: count - 1 }, (_, index) => index);
  for (const [feature, { min, max }] of PARAMETERS.entries()) {
    for (let index = strata.length - 1; index > 0; index--) {
      const swap = Math.floor(next() * (index + 1));
      [strata[index], strata[swap]] = [strata[swap], strata[index]];
    }
    for (const [index, stratum] of strata.entries()) {
      samples[(index + 1) * PARAM_STRIDE + feature] = min + (max - min) * (stratum + next()) / strata.length;
    }
  }
  return samples;
}

/** Return the exact best candidates, with index as the tie-breaker. */
export function rankCandidates(scores: Float32Array, retain: number): Uint32Array {
  if (!Number.isSafeInteger(retain) || retain < 1) throw new RangeError('Retain at least one candidate.');
  const size = Math.min(retain, scores.length);
  const heap = Uint32Array.from({ length: size }, (_, index) => index);
  const order = (left: number, right: number) => scores[left] - scores[right] || left - right;
  if (size === scores.length) return heap.sort(order);

  // The worst retained candidate sits at the root. Candidates outside the
  // ensemble need only one comparison, without a full sort or index array.
  function siftDown(parent: number) {
    const previous = heap[parent];
    for (let child = 2 * parent + 1; child < size; child = 2 * parent + 1) {
      if (child + 1 < size && order(heap[child + 1], heap[child]) > 0) child++;
      if (order(previous, heap[child]) >= 0) break;
      heap[parent] = heap[child];
      parent = child;
    }
    heap[parent] = previous;
  }

  for (let index = Math.floor(size / 2) - 1; index >= 0; index--) siftDown(index);
  for (let index = size; index < scores.length; index++) {
    if (order(index, heap[0]) < 0) {
      heap[0] = index;
      siftDown(0);
    }
  }
  return heap.sort(order);
}

export function quantile(sorted: readonly number[], probability: number): number | null {
  if (!sorted.length) return null;
  const index = probability * (sorted.length - 1);
  const before = Math.floor(index);
  return sorted[before] + (sorted[Math.ceil(index)] - sorted[before]) * (index - before);
}

export function mean(values: readonly (number | null)[]): number | null {
  let count = 0;
  let sum = 0;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    sum += value;
    count++;
  }
  return count ? sum / count : null;
}
