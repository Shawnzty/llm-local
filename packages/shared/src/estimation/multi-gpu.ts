import type { Machine, MachineCompatibility, ModelVariant, ModelFamily } from '../types';
import type { GpuProfile } from '../types';
import { estimateVram } from './engine';
import {
  FIXED_OVERHEAD_GB,
  HEADROOM_FACTOR,
  MULTI_GPU_OVERHEAD,
  COMPAT_YES_THRESHOLD,
  COMPAT_MAYBE_THRESHOLD,
} from './constants';

function getGpuById(gpus: GpuProfile[], id: string): GpuProfile | undefined {
  return gpus.find((g) => g.id === id);
}

export function checkMachineCompatibility(
  machine: Machine,
  variant: ModelVariant,
  gpus: GpuProfile[],
): MachineCompatibility {
  const estimation = estimateVram(variant);

  let totalVram = 0;
  let perGpuVram = Infinity;
  let gpuCount = 0;
  for (const mg of machine.gpus) {
    const gpu = getGpuById(gpus, mg.gpuId);
    if (!gpu) continue;
    totalVram += gpu.vramGB * mg.count;
    perGpuVram = gpu.vramGB;
    gpuCount += mg.count;
  }

  if (gpuCount === 0) {
    return {
      variant,
      totalVramAvailableGB: 0,
      estimatedVramGB: estimation.totalEstimatedGB,
      perGpuUsageGB: estimation.totalEstimatedGB,
      perGpuAvailableGB: 0,
      gpuCount: 0,
      verdict: 'no',
    };
  }

  if (gpuCount === 1) {
    const ratio = estimation.totalEstimatedGB / perGpuVram;
    return {
      variant,
      totalVramAvailableGB: totalVram,
      estimatedVramGB: estimation.totalEstimatedGB,
      perGpuUsageGB: estimation.totalEstimatedGB,
      perGpuAvailableGB: perGpuVram,
      gpuCount: 1,
      verdict: ratio <= COMPAT_YES_THRESHOLD ? 'yes' : ratio <= COMPAT_MAYBE_THRESHOLD ? 'maybe' : 'no',
    };
  }

  // Multi-GPU: weights and KV cache split across GPUs, each pays fixed overhead
  const commOverhead = MULTI_GPU_OVERHEAD[gpuCount] ?? 0.20;
  const modelMemory = estimation.weightMemoryGB + estimation.kvCacheMemoryGB;
  const perGpuModel = modelMemory / gpuCount;
  const perGpuUsage = perGpuModel * (1 + commOverhead) * (1 + HEADROOM_FACTOR) + FIXED_OVERHEAD_GB;

  const ratio = perGpuUsage / perGpuVram;

  return {
    variant,
    totalVramAvailableGB: totalVram,
    estimatedVramGB: round2(perGpuUsage * gpuCount),
    perGpuUsageGB: round2(perGpuUsage),
    perGpuAvailableGB: perGpuVram,
    gpuCount,
    verdict: ratio <= COMPAT_YES_THRESHOLD ? 'yes' : ratio <= COMPAT_MAYBE_THRESHOLD ? 'maybe' : 'no',
  };
}

export function getMachineCompatibilityMatrix(
  machine: Machine,
  families: ModelFamily[],
  gpus: GpuProfile[],
): MachineCompatibility[] {
  const results: MachineCompatibility[] = [];
  for (const family of families) {
    for (const variant of family.variants) {
      results.push(checkMachineCompatibility(machine, variant, gpus));
    }
  }
  const verdictOrder = { yes: 0, maybe: 1, no: 2 };
  results.sort((a, b) => {
    const vd = verdictOrder[a.verdict] - verdictOrder[b.verdict];
    if (vd !== 0) return vd;
    return a.variant.parameterCount - b.variant.parameterCount;
  });
  return results;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
