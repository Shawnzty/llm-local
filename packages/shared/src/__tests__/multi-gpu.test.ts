import { describe, it, expect } from 'vitest';
import { checkMachineCompatibility, getMachineCompatibilityMatrix } from '../estimation/multi-gpu';
import { MODEL_FAMILIES } from '../data/models';
import { GPU_PROFILES } from '../data/gpus';
import type { Machine, ModelVariant } from '../types';

const testVariant7B: ModelVariant = {
  id: 'test-7b',
  familyId: 'test',
  sizeLabel: '7B',
  parameterCount: 7,
  layers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 128,
  maxContext: 8192,
  typeBadges: [],
  isMoE: false,
};

const testVariant70B: ModelVariant = {
  id: 'test-70b',
  familyId: 'test',
  sizeLabel: '70B',
  parameterCount: 70,
  layers: 80,
  hiddenSize: 8192,
  numAttentionHeads: 64,
  numKVHeads: 8,
  headDim: 128,
  maxContext: 8192,
  typeBadges: [],
  isMoE: false,
};

const singleGpuMachine: Machine = {
  id: 'single-4090',
  slug: 'single-4090',
  name: { ja: 'Test', en: 'Test', zh: 'Test' },
  tagline: { ja: '', en: '', zh: '' },
  description: { ja: '', en: '', zh: '' },
  gpus: [{ gpuId: 'rtx-4090-24gb', count: 1 }],
  cpu: 'Test',
  ramGB: 64,
  storage: '1TB',
  priceDisplay: { ja: '', en: '', zh: '' },
  images: [],
  featured: false,
  available: true,
  specs: {},
};

const dualGpuMachine: Machine = {
  id: 'dual-4090',
  slug: 'dual-4090',
  name: { ja: 'Test Dual', en: 'Test Dual', zh: 'Test Dual' },
  tagline: { ja: '', en: '', zh: '' },
  description: { ja: '', en: '', zh: '' },
  gpus: [{ gpuId: 'rtx-4090-24gb', count: 2 }],
  cpu: 'Test',
  ramGB: 128,
  storage: '2TB',
  priceDisplay: { ja: '', en: '', zh: '' },
  images: [],
  featured: false,
  available: true,
  specs: {},
};

describe('checkMachineCompatibility', () => {
  it('7B on single 4090 should be yes', () => {
    const result = checkMachineCompatibility(singleGpuMachine, testVariant7B, GPU_PROFILES);
    expect(result.verdict).toBe('yes');
    expect(result.gpuCount).toBe(1);
    expect(result.totalVramAvailableGB).toBe(24);
    expect(result.estimatedVramGB).toBeGreaterThan(0);
  });

  it('70B on single 4090 should be no', () => {
    const result = checkMachineCompatibility(singleGpuMachine, testVariant70B, GPU_PROFILES);
    expect(result.verdict).toBe('no');
  });

  it('70B on dual 4090 should improve (maybe or yes)', () => {
    const result = checkMachineCompatibility(dualGpuMachine, testVariant70B, GPU_PROFILES);
    expect(result.gpuCount).toBe(2);
    expect(result.totalVramAvailableGB).toBe(48);
    expect(['yes', 'maybe']).toContain(result.verdict);
  });

  it('multi-GPU has higher per-GPU overhead than single', () => {
    const single = checkMachineCompatibility(singleGpuMachine, testVariant7B, GPU_PROFILES);
    const dual = checkMachineCompatibility(dualGpuMachine, testVariant7B, GPU_PROFILES);
    // Per-GPU usage on dual should be less than single total (parallelism wins)
    expect(dual.perGpuUsageGB).toBeLessThan(single.estimatedVramGB);
  });
});

describe('getMachineCompatibilityMatrix', () => {
  it('returns results for all model variants', () => {
    const totalVariants = MODEL_FAMILIES.reduce((n, f) => n + f.variants.length, 0);
    const matrix = getMachineCompatibilityMatrix(singleGpuMachine, MODEL_FAMILIES, GPU_PROFILES);
    expect(matrix.length).toBe(totalVariants);
  });

  it('results are sorted: yes first, then maybe, then no', () => {
    const matrix = getMachineCompatibilityMatrix(singleGpuMachine, MODEL_FAMILIES, GPU_PROFILES);
    const verdictOrder = { yes: 0, maybe: 1, no: 2 };
    for (let i = 1; i < matrix.length; i++) {
      expect(verdictOrder[matrix[i].verdict]).toBeGreaterThanOrEqual(
        verdictOrder[matrix[i - 1].verdict],
      );
    }
  });
});
