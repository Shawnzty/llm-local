import { Hono } from 'hono';
import { GPU_PROFILES, getGpuById } from '@llm-local/shared';
import { readGpusFromDb } from '../db/queries.js';

export const gpus = new Hono();

gpus.get('/', async (c) => {
  try {
    const fromDb = await readGpusFromDb();
    if (fromDb && fromDb.length > 0) return c.json(fromDb);
  } catch (err) {
    console.warn('[gpus] DB read failed, using bundled data:', err);
  }
  return c.json(GPU_PROFILES);
});

gpus.get('/:gpuId', async (c) => {
  const gpuId = c.req.param('gpuId');
  const gpu = getGpuById(gpuId);
  if (!gpu) return c.json({ error: 'GPU not found' }, 404);
  return c.json(gpu);
});
