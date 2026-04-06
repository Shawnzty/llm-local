import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema.js';

const databaseUrl = process.env.DATABASE_URL;

// Lazy singleton so importing this module doesn't crash when DATABASE_URL
// is absent (e.g. in local dev without Postgres, routes fall back to bundled data).
let _db: ReturnType<typeof drizzle<typeof schema>> | null = null;
let _client: ReturnType<typeof postgres> | null = null;

export function getDb() {
  if (_db) return _db;
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is not set');
  }
  _client = postgres(databaseUrl, { max: 5 });
  _db = drizzle(_client, { schema });
  return _db;
}

export async function closeDb() {
  if (_client) {
    await _client.end();
    _client = null;
    _db = null;
  }
}

export { schema };
