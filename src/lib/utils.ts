import { MODEL_FAMILIES } from './data/models';
import { GPU_PROFILES } from './data/gpus';
import type { ModelFamily, ModelVariant, GpuProfile } from './types';

export function getAllFamilies(): ModelFamily[] {
  return MODEL_FAMILIES;
}

export function getFamilyById(familyId: string): ModelFamily | undefined {
  return MODEL_FAMILIES.find((f) => f.id === familyId);
}

export function getVariantsForFamily(familyId: string): ModelVariant[] {
  return getFamilyById(familyId)?.variants ?? [];
}

export function getVariantById(variantId: string): ModelVariant | undefined {
  for (const family of MODEL_FAMILIES) {
    const variant = family.variants.find((v) => v.id === variantId);
    if (variant) return variant;
  }
  return undefined;
}

export function getAllGpus(): GpuProfile[] {
  return GPU_PROFILES;
}

export function getGpuById(gpuId: string): GpuProfile | undefined {
  return GPU_PROFILES.find((g) => g.id === gpuId);
}

export function formatGB(gb: number): string {
  if (gb >= 100) return `${Math.round(gb)} GB`;
  if (gb >= 10) return `${gb.toFixed(1)} GB`;
  return `${gb.toFixed(1)} GB`;
}

export function formatContext(tokens: number): string {
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}K`;
  return `${tokens}`;
}
