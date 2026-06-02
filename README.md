# LLM Local

Sales-oriented bilingual website for pre-built local LLM inference machines (NVIDIA Tesla V100). Each machine page shows vendor-measured performance and which models it can run.

## Current scope (V2)

- **Machine catalog**: 9 pre-built Tesla V100 machines (entry / mid / flagship tiers, single- and multi-GPU)
- **Per-machine sales data**: Vendor-measured performance (tokens/sec), "what it can run", value props, badges, warranty/PSE
- **Competitor comparison**: Market-anchor table (RTX 5060 Ti / 5090 / Applied / Mac Studio) positioning the lineup at 32GB+ VRAM
- **Bilingual**: Japanese (default, ¥JPY), English ($USD)
- **VRAM tools**: Estimator and GPU compatibility checker (SEO / free traffic, decoupled from the catalog)
- **Inquiry form**: Quote request flow (no payment processing)

Market: Japan-first, expanding overseas. Commerce: inquiry-based. Catalog is maintained as TypeScript in `packages/shared/src/data/`.

## Tech stack

- **Monorepo**: pnpm workspaces
- **Frontend (`apps/web`)**: Next.js 16 (App Router, Server Components + ISR), TypeScript, Tailwind CSS v4
- **Backend (`apps/api`)**: Hono (Node), Drizzle ORM, Postgres, tsx runtime
- **Shared (`packages/shared`)**: Estimation engine, types, seed data — consumed by both apps
- **Tests**: Vitest

## Data sources

- **VRAM estimation logic**: Derived from a deep research report analyzing inference memory requirements (weight memory, KV cache, runtime overhead).
- **Model catalog**: Seed data modeled after Ollama-style catalogs (Llama 3.1/3.2, Qwen 2.5, Gemma 2, DeepSeek, Mistral, Mixtral, Phi 3). Architecture parameters sourced from published model configs.
- **Intelligence scores**: Based on external leaderboard data (Artificial Analysis style).
- **GPU profiles**: Common NVIDIA consumer (RTX 30/40/50 series), professional, and datacenter cards.

All data is stored locally as TypeScript seed files. No live scraping or external API calls at runtime.

## Estimation assumptions

See [ASSUMPTIONS.md](./ASSUMPTIONS.md) for detailed documentation.

Summary:
- 4-bit weight-only quantization (effective 4.1875 bits per weight)
- 8K context window
- Single user (batch size 1)
- FP16 KV cache
- 1.5 GiB fixed runtime overhead + 10% headroom

## Limitations

- V1 covers NVIDIA discrete GPUs only (no Apple Silicon, AMD, or CPU-only estimation)
- No inference speed / latency / throughput estimation
- No advanced mode (custom quantization, context, concurrency)
- Seed data is a practical subset, not exhaustive
- MoE models (DeepSeek V3, Mixtral) use total parameter count for weight memory, which is correct for single-GPU but doesn't account for expert parallelism
- Intelligence scores are approximate external benchmarks, not verified

## Running locally

Requires Node 20+ and pnpm 10+.

```bash
pnpm install

# frontend only (uses bundled seed data when NEXT_PUBLIC_API_URL is unset)
pnpm dev:web

# backend only (requires DATABASE_URL for DB-backed reads; falls back to bundled data otherwise)
pnpm dev:api
```

Open [http://localhost:3000](http://localhost:3000). The API runs on `http://localhost:4000`.

Environment:
- `apps/web/.env.local`: `NEXT_PUBLIC_API_URL=http://localhost:4000`
- `apps/api/.env`: `DATABASE_URL=postgres://…`, `CORS_ORIGIN=http://localhost:3000`

## Tests and build

```bash
pnpm test           # all packages
pnpm build          # all packages
pnpm build:web      # Next.js production build
pnpm build:api      # typecheck the API
```

## Database (backend)

```bash
pnpm --filter @llm-local/api db:generate   # create migration from schema
pnpm --filter @llm-local/api db:migrate    # apply migrations
pnpm --filter @llm-local/api db:seed       # seed from bundled data
pnpm --filter @llm-local/api job:refresh   # run the data refresh job
```

## Project structure

```
apps/
  web/                    # Next.js frontend (Server Components + ISR)
    src/app/              # Pages: /, /estimate, /compatibility
    src/components/       # UI components + client islands
    src/lib/api.ts        # Server-side fetch w/ fallback to bundled data
    src/lib/utils.ts      # Formatters
  api/                    # Hono backend
    src/index.ts          # HTTP entrypoint
    src/routes/           # /health, /models, /gpus
    src/db/               # Drizzle schema, client, queries, seed
    src/jobs/             # refresh-data.ts cron job
packages/
  shared/                 # @llm-local/shared
    src/types.ts          # Domain types
    src/estimation/       # VRAM engine + constants (heart of product)
    src/data/             # Bundled seed data (models, GPUs)
    src/lookups.ts        # Helper lookups on seed arrays
```

## Deployment

- Frontend → Vercel (`apps/web`); set `NEXT_PUBLIC_API_URL` to the Railway API URL
- Backend → Railway (`apps/api`); attach Postgres, run `db:migrate` + `db:seed` on first deploy, schedule `job:refresh` via Railway cron
- DNS → Cloudflare (DNS-only records, no proxy): `llm-local.com` → Vercel, `api.llm-local.com` → Railway
