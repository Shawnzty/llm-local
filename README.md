# LLM Local

VRAM estimation tool for local LLM deployment. Helps you figure out how much GPU memory you need and whether a specific model will run on your hardware.

## Current scope (V1)

Two modes:

1. **VRAM Estimator** — Select a model family and size, get the estimated VRAM requirement.
2. **Compatibility Check** — Select a model and GPU, see whether it fits (yes / maybe / no).

Estimation uses a conservative 4-bit quantization baseline with 8K context and single-user assumptions. NVIDIA discrete GPUs only for V1.

## Tech stack

- Next.js 16 (App Router)
- TypeScript
- Tailwind CSS v4
- Vitest for tests
- No database — local seed data files

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

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Tests

```bash
npx vitest run
```

## Project structure

```
src/
  app/                    # Next.js pages (home, /estimate, /compatibility)
  components/             # UI components
  lib/
    estimation/           # VRAM estimation engine and constants
    data/                 # Seed data (models, GPUs)
    types.ts              # TypeScript interfaces
    utils.ts              # Lookup and formatting helpers
  __tests__/              # Vitest tests
```
