# VRAM Estimation Assumptions — V1

## How VRAM estimation works

Total estimated VRAM is the sum of three components:

```
Total = Weight Memory + KV Cache Memory + Runtime Overhead
```

### 1. Weight memory

```
M_weights = P × b_eff / 8  (bytes)
```

- **P**: Total parameter count (billions × 10^9)
- **b_eff**: Effective bits per weight = 4.1875

The effective bpw accounts for 4-bit integer weights plus group quantization metadata overhead:
- Group size: 128
- Scale: FP16 (16 bits per group)
- Zero-point: INT8 (8 bits per group)
- Overhead: (16 + 8) / 128 = 0.1875 additional bpw

For MoE models, total parameter count is used (not active parameters per token), because all expert weights must reside in VRAM on a single GPU.

### 2. KV cache memory

```
M_KV = 2 × B × S × L × H_kv × d_h × b_KV  (bytes)
```

- **B**: Concurrent sequences = 1 (single-user)
- **S**: Context tokens = 8,192 (8K)
- **L**: Number of transformer layers
- **H_kv**: Number of KV heads (reduced for GQA/MQA models)
- **d_h**: Head dimension (typically hidden_size / num_attention_heads)
- **b_KV**: Bytes per KV element = 2 (FP16)
- Factor of 2 accounts for both keys and values

### 3. Runtime overhead

```
M_overhead = M_fixed + (M_weights + M_KV) × alpha
```

- **M_fixed**: 1.5 GiB — CUDA initialization + allocator reservation
- **alpha**: 0.10 (10%) — headroom for fragmentation and allocator behavior

This follows the "workable" planning envelope from the research report, which provides enough room for stable interactive use without over-provisioning.

## Default assumptions for Mode 1

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | 4-bit (INT4 weight-only) | Most common practical setting for local inference |
| Context window | 8K tokens | Reasonable default for single-user chat |
| Concurrency | 1 (single-user) | Local deployment typical case |
| KV cache precision | FP16 | Standard default in most inference frameworks |
| GPU platform | NVIDIA discrete | V1 scope |
| Fixed overhead | 1.5 GiB | CUDA kernel loading + allocator pools |
| Headroom | 10% | Fragmentation + allocator reservation |

## Compatibility verdicts (Mode 2)

The compatibility check compares estimated VRAM against available GPU VRAM:

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| **Yes** | Estimated ≤ 80% of available | Clear headroom for stable operation |
| **Maybe** | Estimated ≤ 100% of available | Tight fit; may work with careful configuration |
| **No** | Estimated > 100% of available | Insufficient VRAM |

These thresholds are configurable in `packages/shared/src/estimation/constants.ts`.

The 80% "yes" threshold provides a practical safety margin for:
- Allocator caching behavior (PyTorch/CUDA reserve memory pools)
- Temporary workspace buffers during inference
- Minor estimation inaccuracies

## Known simplifications

1. **No KV cache quantization** — V1 assumes FP16 KV. Real deployments may use FP8 or INT8 KV cache, which would reduce KV memory by 2x.
2. **No multimodal token inflation** — Vision models inject additional tokens that increase effective context length. V1 uses the text-only context assumption.
3. **No MLA compression modeling** — DeepSeek models use Multi-head Latent Attention which compresses KV state. V1 uses standard KV head count, which may overestimate.
4. **Single-GPU only** — No tensor parallelism or expert parallelism across multiple GPUs.
5. **No activation memory modeling** — Temporary activations during prefill are not explicitly modeled; they're covered by the headroom factor.

## Source

Estimation logic is derived from a comprehensive research report analyzing inference memory requirements across model architectures, quantization formats, and runtime stacks. The report is included in the repository as `deep-research-report.md`.

## Data refresh job

`apps/api/src/jobs/refresh-data.ts` is the scheduled cron that keeps model / GPU metadata up to date in Postgres.

For V1.5, this job re-upserts the bundled seed data from `@llm-local/shared` and bumps `last_updated` timestamps. It exists primarily to prove the Railway cron infrastructure end-to-end. Real upstream fetchers (Ollama catalog, Artificial Analysis leaderboard, NVIDIA spec pages) will replace the placeholder body in a later version — the surrounding schema, fallback behavior, and deployment wiring will not change.
