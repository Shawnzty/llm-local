import { describe, it, expect } from 'vitest';
import {
  getAllFamilies,
  getFamilyById,
  getVariantsForFamily,
  getVariantById,
  getAllGpus,
  getGpuById,
  formatGB,
  formatContext,
} from '@/lib/utils';

describe('data lookups', () => {
  it('returns all model families', () => {
    const families = getAllFamilies();
    expect(families.length).toBeGreaterThan(5);
    expect(families.every((f) => f.id && f.name && f.variants.length > 0)).toBe(true);
  });

  it('finds a family by id', () => {
    const family = getFamilyById('llama-3.1');
    expect(family).toBeDefined();
    expect(family!.name).toBe('Llama 3.1');
  });

  it('returns undefined for unknown family', () => {
    expect(getFamilyById('nonexistent')).toBeUndefined();
  });

  it('returns variants for a family', () => {
    const variants = getVariantsForFamily('qwen-2.5');
    expect(variants.length).toBeGreaterThan(2);
    expect(variants.every((v) => v.familyId === 'qwen-2.5')).toBe(true);
  });

  it('returns empty array for unknown family variants', () => {
    expect(getVariantsForFamily('nonexistent')).toEqual([]);
  });

  it('finds a variant by id across all families', () => {
    const variant = getVariantById('mixtral-8x7b');
    expect(variant).toBeDefined();
    expect(variant!.isMoE).toBe(true);
    expect(variant!.parameterCount).toBe(46.7);
  });

  it('returns all GPUs', () => {
    const gpus = getAllGpus();
    expect(gpus.length).toBeGreaterThan(10);
    expect(gpus.every((g) => g.vendor === 'NVIDIA')).toBe(true);
  });

  it('finds a GPU by id', () => {
    const gpu = getGpuById('rtx-4090-24gb');
    expect(gpu).toBeDefined();
    expect(gpu!.vramGB).toBe(24);
  });
});

describe('formatGB', () => {
  it('formats small values with one decimal', () => {
    expect(formatGB(3.9)).toBe('3.9 GB');
    expect(formatGB(7.5)).toBe('7.5 GB');
  });

  it('formats medium values with one decimal', () => {
    expect(formatGB(15.62)).toBe('15.6 GB');
  });

  it('formats large values as integers', () => {
    expect(formatGB(130.39)).toBe('130 GB');
  });
});

describe('formatContext', () => {
  it('formats thousands as K', () => {
    expect(formatContext(8192)).toBe('8K');
    expect(formatContext(131072)).toBe('128K');
    expect(formatContext(32768)).toBe('32K');
  });
});
