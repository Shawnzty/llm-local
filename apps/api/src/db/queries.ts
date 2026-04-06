import type { ModelFamily, ModelVariant, GpuProfile, ModelTypeBadge } from '@llm-local/shared';
import { getDb } from './client.js';
import { modelFamilies, modelVariants, gpuProfiles } from './schema.js';

export async function readFamiliesFromDb(): Promise<ModelFamily[] | null> {
  if (!process.env.DATABASE_URL) return null;
  const db = getDb();
  const families = await db.select().from(modelFamilies);
  const variants = await db.select().from(modelVariants);

  const byFamily = new Map<string, ModelVariant[]>();
  for (const v of variants) {
    const mv: ModelVariant = {
      id: v.id,
      familyId: v.familyId,
      sizeLabel: v.sizeLabel,
      parameterCount: v.parameterCount,
      layers: v.layers,
      hiddenSize: v.hiddenSize,
      numAttentionHeads: v.numAttentionHeads,
      numKVHeads: v.numKVHeads,
      headDim: v.headDim,
      maxContext: v.maxContext,
      typeBadges: (v.typeBadges ?? []) as ModelTypeBadge[],
      intelligenceScore: v.intelligenceScore ?? undefined,
      isMoE: v.isMoE,
      activeParameterCount: v.activeParameterCount ?? undefined,
    };
    const list = byFamily.get(v.familyId) ?? [];
    list.push(mv);
    byFamily.set(v.familyId, list);
  }

  return families.map((f) => ({
    id: f.id,
    name: f.name,
    variants: byFamily.get(f.id) ?? [],
  }));
}

export async function readGpusFromDb(): Promise<GpuProfile[] | null> {
  if (!process.env.DATABASE_URL) return null;
  const db = getDb();
  const rows = await db.select().from(gpuProfiles);
  return rows.map((g) => ({
    id: g.id,
    name: g.name,
    vendor: g.vendor as 'NVIDIA',
    vramGB: g.vramGB,
    tier: g.tier as GpuProfile['tier'],
  }));
}

