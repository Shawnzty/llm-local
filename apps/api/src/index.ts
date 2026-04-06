import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { health } from './routes/health.js';
import { models } from './routes/models.js';
import { gpus } from './routes/gpus.js';

const app = new Hono();

const allowedOrigins = (process.env.CORS_ORIGIN ?? 'http://localhost:3000')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

app.use('*', logger());
app.use(
  '*',
  cors({
    origin: (origin) => {
      if (!origin) return null;
      if (allowedOrigins.includes(origin)) return origin;
      // Allow Vercel preview deployments
      if (/^https:\/\/[a-z0-9-]+\.vercel\.app$/.test(origin)) return origin;
      return null;
    },
    allowMethods: ['GET', 'OPTIONS'],
  }),
);

app.route('/health', health);
app.route('/models', models);
app.route('/gpus', gpus);

app.notFound((c) => c.json({ error: 'Not found' }, 404));
app.onError((err, c) => {
  console.error(err);
  return c.json({ error: 'Internal server error' }, 500);
});

const port = Number(process.env.PORT ?? 4000);

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`[api] listening on http://localhost:${info.port}`);
});

export default app;
