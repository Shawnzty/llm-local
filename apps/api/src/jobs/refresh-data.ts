/**
 * V1.5 data refresh job.
 *
 * For V1.5 this is a placeholder that re-runs the seed against the current
 * bundled shared data. Neither Ollama nor Artificial Analysis publish clean
 * public APIs at the time of writing, so real fetching logic is deferred.
 *
 * This job exists to prove the cron infrastructure works end-to-end on
 * Railway; swapping the body for real scrapers later is a small change.
 *
 * See ASSUMPTIONS.md for details.
 */
import { MODEL_FAMILIES, GPU_PROFILES } from '@llm-local/shared';
import { getDb, closeDb } from '../db/client.js';
import { modelFamilies, modelVariants, gpuProfiles } from '../db/schema.js';

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('[refresh] DATABASE_URL not set');
    process.exit(1);
  }
  console.log('[refresh] starting data refresh job…');
  const db = getDb();
  const now = new Date();

  // TODO: Replace with real upstream fetches once sources have stable APIs.
  // For now: re-upsert bundled data, updating lastUpdated timestamps.
  let families = 0;
  let variants = 0;
  for (const family of MODEL_FAMILIES) {
    await db
      .insert(modelFamilies)
      .values({ id: family.id, name: family.name, createdAt: now, updatedAt: now })
      .onConflictDoUpdate({
        target: modelFamilies.id,
        set: { name: family.name, updatedAt: now },
      });
    families++;
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
          source: 'refresh-job',
          lastUpdated: now,
        })
        .onConflictDoUpdate({
          target: modelVariants.id,
          set: { lastUpdated: now, source: 'refresh-job' },
        });
      variants++;
    }
  }

  let gpus = 0;
  for (const gpu of GPU_PROFILES) {
    await db
      .insert(gpuProfiles)
      .values({
        id: gpu.id,
        name: gpu.name,
        vendor: gpu.vendor,
        vramGB: gpu.vramGB,
        tier: gpu.tier,
        source: 'refresh-job',
        lastUpdated: now,
      })
      .onConflictDoUpdate({
        target: gpuProfiles.id,
        set: { lastUpdated: now, source: 'refresh-job' },
      });
    gpus++;
  }

  console.log(`[refresh] done. families=${families} variants=${variants} gpus=${gpus}`);
  await closeDb();
}

main().catch(async (err) => {
  console.error('[refresh] failed:', err);
  await closeDb();
  process.exit(1);
});
