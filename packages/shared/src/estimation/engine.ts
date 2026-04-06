import type { ModelVariant, GpuProfile, EstimationResult, CompatibilityResult } from '../types';
import {
  EFFECTIVE_BPW,
  KV_BYTES_PER_ELEMENT,
  DEFAULT_CONTEXT_TOKENS,
  DEFAULT_BATCH_SIZE,
  FIXED_OVERHEAD_GB,
  HEADROOM_FACTOR,
  COMPAT_YES_THRESHOLD,
  COMPAT_MAYBE_THRESHOLD,
  BYTES_PER_GIB,
} from './constants';

/**
 * Compute weight memory in GB.
 *
 * Formula: M_weights = P * b_eff / 8  (bytes)
 * where P is parameter count and b_eff is effective bits per weight.
 *
 * For MoE models, all expert weights must reside in memory even though
 * only a subset are active per token — so we use total parameter count.
 */
function computeWeightMemoryGB(parameterCountBillions: number): number {
  const parameterCount = parameterCountBillions * 1e9;
  const bytes = parameterCount * EFFECTIVE_BPW / 8;
  return bytes / BYTES_PER_GIB;
}

/**
 * Compute KV cache memory in GB.
 *
 * Formula: M_KV = 2 * B * S * L * H_kv * d_h * b_KV  (bytes)
 *
 * Factor of 2 accounts for both keys and values stored per layer.
 */
function computeKVCacheMemoryGB(
  layers: number,
  numKVHeads: number,
  headDim: number,
  contextTokens: number = DEFAULT_CONTEXT_TOKENS,
  batchSize: number = DEFAULT_BATCH_SIZE,
): number {
  const bytes = 2 * batchSize * contextTokens * layers * numKVHeads * headDim * KV_BYTES_PER_ELEMENT;
  return bytes / BYTES_PER_GIB;
}

/**
 * Estimate total VRAM needed for a model under V1 default assumptions:
 * 4-bit quantization, 8K context, single-user, NVIDIA GPU, FP16 KV cache.
 *
 * Uses the "workable" planning envelope from the research report:
 *   M_total = M_fixed + (M_weights + M_KV) * (1 + alpha)
 */
export function estimateVram(variant: ModelVariant): EstimationResult {
  const weightMemoryGB = computeWeightMemoryGB(variant.parameterCount);
  const kvCacheMemoryGB = computeKVCacheMemoryGB(
    variant.layers,
    variant.numKVHeads,
    variant.headDim,
  );
  const tensorTotal = weightMemoryGB + kvCacheMemoryGB;
  const runtimeOverheadGB = FIXED_OVERHEAD_GB + tensorTotal * HEADROOM_FACTOR;
  const totalEstimatedGB = tensorTotal + runtimeOverheadGB;

  return {
    weightMemoryGB: round2(weightMemoryGB),
    kvCacheMemoryGB: round2(kvCacheMemoryGB),
    runtimeOverheadGB: round2(runtimeOverheadGB),
    totalEstimatedGB: round2(totalEstimatedGB),
  };
}

/**
 * Check whether a model can run on a given GPU.
 *
 * Thresholds (configurable in constants.ts):
 *   yes:   estimated VRAM <= 80% of available (clear headroom)
 *   maybe: estimated VRAM <= 100% of available (tight but may work)
 *   no:    estimated VRAM > available (insufficient)
 */
export function checkCompatibility(
  variant: ModelVariant,
  gpu: GpuProfile,
): CompatibilityResult {
  const estimation = estimateVram(variant);
  const ratio = estimation.totalEstimatedGB / gpu.vramGB;

  let verdict: 'yes' | 'maybe' | 'no';
  if (ratio <= COMPAT_YES_THRESHOLD) {
    verdict = 'yes';
  } else if (ratio <= COMPAT_MAYBE_THRESHOLD) {
    verdict = 'maybe';
  } else {
    verdict = 'no';
  }

  return {
    estimatedVramGB: estimation.totalEstimatedGB,
    availableVramGB: gpu.vramGB,
    verdict,
  };
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
