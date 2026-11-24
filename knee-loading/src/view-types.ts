import type { BenchmarkResult, BenchmarkRow } from './backend.ts';
import type { KneeSide } from './dataset.ts';

export type Compartment = 'medial' | 'lateral' | 'total';
export type Samples = (number | null)[];

export interface WaveformView {
  measured: Samples;
  modeled: Samples;
  low: Samples;
  high: Samples;
  proxy?: Samples;
}

export interface TrialView {
  label: string;
  phase: number[];
  kneeFlexionDeg?: Samples;
  compartments: Partial<Record<Compartment, WaveformView>>;
  trialCount?: number;
}

export interface MetricView {
  label: string;
  unit: string;
  measured: number | null;
  model: number | null;
  proxy: number | null;
  modelLow?: number | null;
  modelHigh?: number | null;
  compartment?: Compartment;
  delta?: boolean;
  offloadThresholdBw?: number;
  offloadShare?: number | null;
}

export interface ComparisonView {
  id: string;
  subject: string;
  groupId?: string;
  side?: KneeSide;
  label: string;
  kamEarlyChangePercent?: number | null;
  reference: TrialView;
  intervention: TrialView;
  metrics: MetricView[];
  context?: string;
  metricsTitle?: string;
  metricsIntro?: string;
}

export interface StudyBenchmarkWorkload extends BenchmarkResult {
  rows: (BenchmarkRow & { topCandidateAgreement?: { size: number; shared: number } })[];
}

export interface StudyBenchmark extends StudyBenchmarkWorkload {
  calibration?: StudyBenchmarkWorkload;
  runtime?: string;
  trial?: { id: string; subject: string; condition: string };
}

export interface PatellaView {
  summary: string;
  flexionStartDeg: number;
  flexionEndDeg: number;
  measuredRotationDeg: number;
  referenceRotationDeg: number;
  viewerRotationDeg: number;
  source: { label: string; url: string };
}

export interface StudyView {
  dataset: string;
  summary: string;
  sources: { label: string; url: string }[];
  methods: string[];
  comparisons: ComparisonView[];
  highlightedComparisonId?: string;
  benchmark?: StudyBenchmark;
  benchmarkAvailable?: boolean;
  intervalLabel?: string;
  notice?: string;
  patella?: PatellaView;
  downloads?: { label: string; url: string; filename: string }[];
}
