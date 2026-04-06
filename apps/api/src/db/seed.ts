import { MODEL_FAMILIES, GPU_PROFILES } from '@llm-local/shared';
import { getDb, closeDb } from './client.js';
import { modelFamilies, modelVariants, gpuProfiles } from './schema.js';

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('DATABASE_URL not set — cannot seed.');
    process.exit(1);
  }
  const db = getDb();
  const now = new Date();

  console.log('[seed] upserting model families…');
  for (const family of MODEL_FAMILIES) {
    await db
      .insert(modelFamilies)
      .values({ id: family.id, name: family.name, createdAt: now, updatedAt: now })
      .onConflictDoUpdate({
        target: modelFamilies.id,
        set: { name: family.name, updatedAt: now },
      });

    for (const v of family.variants) {
      await db
        .insert(modelVariants)
        .values({
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
          typeBadges: v.typeBadges,
          intelligenceScore: v.intelligenceScore ?? null,
          isMoE: v.isMoE,
          activeParameterCount: v.activeParameterCount ?? null,
          source: 'seed',
          lastUpdated: now,
        })
        .onConflictDoUpdate({
          target: modelVariants.id,
          set: {
            sizeLabel: v.sizeLabel,
            parameterCount: v.parameterCount,
            layers: v.layers,
            hiddenSize: v.hiddenSize,
            numAttentionHeads: v.numAttentionHeads,
            numKVHeads: v.numKVHeads,
            headDim: v.headDim,
            maxContext: v.maxContext,
            typeBadges: v.typeBadges,
            intelligenceScore: v.intelligenceScore ?? null,
            isMoE: v.isMoE,
            activeParameterCount: v.activeParameterCount ?? null,
            source: 'seed',
            lastUpdated: now,
          },
        });
    }
  }

  console.log('[seed] upserting GPU profiles…');
  for (const gpu of GPU_PROFILES) {
    await db
      .insert(gpuProfiles)
      .values({
        id: gpu.id,
        name: gpu.name,
        vendor: gpu.vendor,
        vramGB: gpu.vramGB,
        tier: gpu.tier,
        source: 'seed',
        lastUpdated: now,
      })
      .onConflictDoUpdate({
        target: gpuProfiles.id,
        set: {
          name: gpu.name,
          vendor: gpu.vendor,
          vramGB: gpu.vramGB,
          tier: gpu.tier,
          source: 'seed',
          lastUpdated: now,
        },
      });
  }

  console.log(
    `[seed] done. ${MODEL_FAMILIES.length} families, ${MODEL_FAMILIES.reduce(
      (n: number, f) => n + f.variants.length,
      0,
    )} variants, ${GPU_PROFILES.length} GPUs.`,
  );

  await closeDb();
}

main().catch(async (err) => {
  console.error('[seed] failed:', err);
  await closeDb();
  process.exit(1);
});
