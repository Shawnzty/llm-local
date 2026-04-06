import { Hono } from 'hono';
import { MODEL_FAMILIES, getFamilyById } from '@llm-local/shared';
import { readFamiliesFromDb } from '../db/queries.js';

export const models = new Hono();

models.get('/', async (c) => {
  // Try DB first; fall back to bundled seed data if DB is unavailable.
  try {
    const fromDb = await readFamiliesFromDb();
    if (fromDb && fromDb.length > 0) return c.json(fromDb);
  } catch (err) {
    console.warn('[models] DB read failed, using bundled data:', err);
  }
  return c.json(MODEL_FAMILIES);
});

models.get('/:familyId', async (c) => {
  const familyId = c.req.param('familyId');
  const family = getFamilyById(familyId);
  if (!family) return c.json({ error: 'Family not found' }, 404);
  return c.json(family);
});
