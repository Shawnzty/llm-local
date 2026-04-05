import { describe, it, expect } from 'vitest';
import { estimateVram, checkCompatibility } from '@/lib/estimation/engine';
import type { ModelVariant, GpuProfile } from '@/lib/types';

// Test model: Llama-3-8B architecture from the research report
// At INT4 (4.1875 bpw): weights ≈ 3.90 GiB, KV at 8K ≈ 1.00 GiB
const llama3_8b: ModelVariant = {
  id: 'llama-3-8b',
  familyId: 'llama-3',
  sizeLabel: '8B',
  parameterCount: 8.03,
  layers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 128,
  maxContext: 8192,
  typeBadges: ['tools'],
  intelligenceScore: 33,
  isMoE: false,
};

// Qwen2.5-32B from report: at INT4 weights ≈ 15.62 GiB, KV at 8K ≈ 1.00 GiB
const qwen25_32b: ModelVariant = {
  id: 'qwen-2.5-32b',
  familyId: 'qwen-2.5',
  sizeLabel: '32B',
  parameterCount: 32.76,
  layers: 64,
  hiddenSize: 5120,
  numAttentionHeads: 40,
  numKVHeads: 8,
  headDim: 128,
  maxContext: 131072,
  typeBadges: ['tools'],
  isMoE: false,
};

// Mixtral 8x7B from report: total 46.7B params
const mixtral_8x7b: ModelVariant = {
  id: 'mixtral-8x7b',
  familyId: 'mixtral',
  sizeLabel: '8x7B',
  parameterCount: 46.7,
  layers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 128,
  maxContext: 32768,
  typeBadges: [],
  isMoE: true,
  activeParameterCount: 12.9,
};

const rtx4090: GpuProfile = {
  id: 'rtx-4090-24gb',
  name: 'RTX 4090 24GB',
  vendor: 'NVIDIA',
  vramGB: 24,
  tier: 'consumer',
};

const rtx4060: GpuProfile = {
  id: 'rtx-4060-8gb',
  name: 'RTX 4060 8GB',
  vendor: 'NVIDIA',
  vramGB: 8,
  tier: 'consumer',
};

const a100_80gb: GpuProfile = {
  id: 'a100-80gb',
  name: 'A100 80GB',
  vendor: 'NVIDIA',
  vramGB: 80,
  tier: 'datacenter',
};

describe('estimateVram', () => {
  it('estimates Llama-3-8B weight memory close to report value (~3.9 GiB)', () => {
    const result = estimateVram(llama3_8b);
    // Report says INT4 weights for 8B ≈ 3.90 GiB
    expect(result.weightMemoryGB).toBeGreaterThan(3.5);
    expect(result.weightMemoryGB).toBeLessThan(4.5);
  });

  it('estimates Llama-3-8B KV cache at 8K close to report value (~1.0 GiB)', () => {
    const result = estimateVram(llama3_8b);
    // Report: KV at 8K for Llama-3-8B = 1.00 GiB
    expect(result.kvCacheMemoryGB).toBeGreaterThan(0.4);
    expect(result.kvCacheMemoryGB).toBeLessThan(1.5);
  });

  it('includes runtime overhead in total', () => {
    const result = estimateVram(llama3_8b);
    expect(result.runtimeOverheadGB).toBeGreaterThan(1.0);
    expect(result.totalEstimatedGB).toBeGreaterThan(
      result.weightMemoryGB + result.kvCacheMemoryGB,
    );
  });

  it('estimates Qwen2.5-32B weight memory close to report value (~15.6 GiB)', () => {
    const result = estimateVram(qwen25_32b);
    expect(result.weightMemoryGB).toBeGreaterThan(14);
    expect(result.weightMemoryGB).toBeLessThan(18);
  });

  it('uses total parameter count for MoE models (not active)', () => {
    const result = estimateVram(mixtral_8x7b);
    // 46.7B at 4.1875 bpw should be ~22.7 GiB weights
    expect(result.weightMemoryGB).toBeGreaterThan(20);
    expect(result.weightMemoryGB).toBeLessThan(25);
  });

  it('returns rounded values to 2 decimal places', () => {
    const result = estimateVram(llama3_8b);
    const decimals = (n: number) => {
      const s = n.toString();
      const dot = s.indexOf('.');
      return dot === -1 ? 0 : s.length - dot - 1;
    };
    expect(decimals(result.totalEstimatedGB)).toBeLessThanOrEqual(2);
  });
});

describe('checkCompatibility', () => {
  it('returns "yes" when model fits comfortably', () => {
    // Llama-3-8B (~6 GB total) on RTX 4090 (24 GB) = clear yes
    const result = checkCompatibility(llama3_8b, rtx4090);
    expect(result.verdict).toBe('yes');
    expect(result.availableVramGB).toBe(24);
  });

  it('returns "no" when model exceeds GPU VRAM', () => {
    // Qwen2.5-32B (~19 GB total) on RTX 4060 (8 GB) = no
    const result = checkCompatibility(qwen25_32b, rtx4060);
    expect(result.verdict).toBe('no');
  });

  it('returns "yes" for large GPU with large model', () => {
    // Mixtral 8x7B (~26 GB total) on A100 80GB = yes
    const result = checkCompatibility(mixtral_8x7b, a100_80gb);
    expect(result.verdict).toBe('yes');
  });

  it('returns correct estimatedVramGB matching estimateVram', () => {
    const estimation = estimateVram(llama3_8b);
    const compat = checkCompatibility(llama3_8b, rtx4090);
    expect(compat.estimatedVramGB).toBe(estimation.totalEstimatedGB);
  });
});
