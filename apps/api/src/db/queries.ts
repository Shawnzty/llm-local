import type { ModelFamily, ModelVariant, GpuProfile, ModelTypeBadge } from '@tadzuna/shared';
import { getDb } from './client.js';
import { modelFamilies, modelVariants, gpuProfiles, inquiries } from './schema.js';
import type { InquiryInput } from '../lib/email.js';

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

/**
 * Persist an inquiry. Best-effort: returns false (no-op) when DATABASE_URL is
 * unset so the caller can still deliver the lead by email without a database.
 */
export async function saveInquiry(input: InquiryInput): Promise<boolean> {
  if (!process.env.DATABASE_URL) return false;
  const db = getDb();
  await db.insert(inquiries).values({
    name: input.name,
    email: input.email,
    phone: input.phone ?? null,
    company: input.company ?? null,
    machineId: input.machineId ?? null,
    message: input.message,
    locale: input.locale ?? null,
  });
  return true;
}

