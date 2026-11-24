import type { Samples, TrialView, WaveformView } from './view-types.ts';

interface ForceScale {
  min: number;
  max: number;
  step: number;
}

const plotPadding = { left: 40, right: 15, top: 13, bottom: 29 };

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function forceScale(waves: (WaveformView | undefined)[]): ForceScale {
  let min = 0;
  let max = 0;
  for (const wave of waves) {
    if (!wave) continue;
    for (const series of [wave.measured, wave.modeled, wave.low, wave.high, wave.proxy ?? []]) {
      for (const value of series) {
        if (!isNumber(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }
  }
  if (max === min) max = min + 1;

  const minStep = (max - min) / 4;
  const magnitude = 10 ** Math.floor(Math.log10(minStep));
  const increment = [1, 2, 2.5, 5, 10].find((factor) => factor * magnitude >= minStep) ?? 10;
  const step = magnitude * increment;
  return { min: Math.floor(min / step) * step, max: Math.ceil(max / step) * step, step };
}

export function phaseAtPointer(canvas: HTMLCanvasElement, event: MouseEvent): number {
  const rect = canvas.getBoundingClientRect();
  const plotWidth = rect.width - plotPadding.left - plotPadding.right;
  if (plotWidth <= 0) return 0;
  return Math.round(Math.max(0, Math.min(100, (event.clientX - rect.left - plotPadding.left) * 100 / plotWidth)));
}

export function sampleAt(phases: number[], values: Samples, phase: number): number | null {
  const hi = phases.findIndex((sample) => sample >= phase);
  if (hi < 0) return values.at(-1) ?? null;
  const upper = values[hi];
  if (hi === 0 || phases[hi] === phase) return upper ?? null;

  const lower = values[hi - 1];
  if (!isNumber(lower) || !isNumber(upper)) return null;
  const fraction = (phase - phases[hi - 1]) / (phases[hi] - phases[hi - 1]);
  return lower + fraction * (upper - lower);
}

export function drawCurve(
  canvas: HTMLCanvasElement,
  trial: TrialView,
  wave: WaveformView | undefined,
  scale: ForceScale,
  cursor: number,
): void {
  const context = canvas.getContext('2d');
  if (!context) return;

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
  if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const style = getComputedStyle(canvas);
  const measured = style.getPropertyValue('--chart-measured').trim();
  const modeled = style.getPropertyValue('--chart-model').trim();
  const proxy = style.getPropertyValue('--chart-proxy').trim();
  const left = plotPadding.left;
  const right = width - plotPadding.right;
  const top = plotPadding.top;
  const bottom = height - plotPadding.bottom;
  if (right <= left || bottom <= top) return;

  const x = (phase: number) => left + phase * (right - left) / 100;
  const y = (force: number) => bottom - (force - scale.min) * (bottom - top) / (scale.max - scale.min);
  const phases = trial.phase;

  context.font = `11px ${style.getPropertyValue('--mono').trim()}`;
  context.fillStyle = style.getPropertyValue('--muted').trim();
  context.textBaseline = 'middle';
  context.textAlign = 'right';
  for (let index = 0; index <= Math.round((scale.max - scale.min) / scale.step); index++) {
    const force = scale.min + index * scale.step;
    const row = y(force);
    context.strokeStyle = index ? '#e6e9e3' : '#acb9af';
    context.beginPath();
    context.moveTo(left, row);
    context.lineTo(right, row);
    context.stroke();
    context.fillText(Number(force.toPrecision(4)).toString(), left - 7, row);
  }
  context.textAlign = 'center';
  for (const value of [0, 25, 50, 75, 100]) {
    context.fillText(`${value}%`, x(value), height - 10);
  }

  if (!wave) {
    context.fillText('Waveform unavailable', (left + right) / 2, (top + bottom) / 2);
    return;
  }

  const strokeSeries = (values: Samples, color: string, dash: number[], size: number) => {
    context.strokeStyle = color;
    context.lineWidth = size;
    context.setLineDash(dash);
    context.beginPath();
    let connected = false;
    for (let index = 0; index < phases.length; index++) {
      const value = values[index];
      if (!isNumber(value)) {
        connected = false;
        continue;
      }
      if (connected) context.lineTo(x(phases[index]), y(value));
      else context.moveTo(x(phases[index]), y(value));
      connected = true;
    }
    context.stroke();
    context.setLineDash([]);
  };

  context.save();
  context.beginPath();
  context.rect(left - 2, top - 3, right - left + 4, bottom - top + 6);
  context.clip();
  context.fillStyle = style.getPropertyValue('--chart-band').trim();
  let end = 0;
  for (let start = 0; start < phases.length; start = end + 1) {
    end = start;
    while (end < phases.length && isNumber(wave.high[end]) && isNumber(wave.low[end])) end++;
    if (end - start < 2) continue;

    context.beginPath();
    context.moveTo(x(phases[start]), y(wave.high[start] as number));
    for (let index = start + 1; index < end; index++) context.lineTo(x(phases[index]), y(wave.high[index] as number));
    for (let index = end - 1; index >= start; index--) context.lineTo(x(phases[index]), y(wave.low[index] as number));
    context.closePath();
    context.fill();
  }
  if (wave.proxy) strokeSeries(wave.proxy, proxy, [2, 3], 1.75);
  strokeSeries(wave.modeled, modeled, [6, 4], 2);
  strokeSeries(wave.measured, measured, [], 2.4);

  const cursorX = x(cursor);
  context.beginPath();
  context.strokeStyle = '#728677';
  context.lineWidth = 1;
  context.setLineDash([2, 3]);
  context.moveTo(cursorX, top);
  context.lineTo(cursorX, bottom);
  context.stroke();
  context.setLineDash([]);
  const markers: [Samples, string][] = [[wave.measured, measured], [wave.modeled, modeled]];
  for (const [series, color] of markers) {
    const value = sampleAt(phases, series, cursor);
    if (!isNumber(value)) continue;
    context.fillStyle = color;
    context.strokeStyle = '#fff';
    context.beginPath();
    context.arc(cursorX, y(value), 3.4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
  context.restore();
}
