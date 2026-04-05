# Estimating Local LLM Inference Memory Requirements: Foundations and Model Factors

## Executive summary

This report builds a deployment-grade mental and mathematical model for estimating **inference-time memory** when running a local transformer LLM, emphasizing why the common shortcut “parameter count × bytes” is systematically insufficient. That shortcut only estimates (a portion of) **static weight storage**, but real inference memory is dominated—often decisively—by (a) **Key–Value (KV) cache growth with context length and concurrency**, (b) **temporary workspaces and kernel scratch buffers** that depend on the attention/linear algebra implementation, and (c) **runtime allocator behavior** (reservation, fragmentation, and shape-dependent preallocation). citeturn0search0turn9view1turn12search0turn2search13

A second thesis is architectural: model families that keep the same “headline parameter count” can have radically different memory curves because KV cache scales with **layers and attention state dimensionality**; therefore design choices like **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** explicitly exist to reduce KV cache capacity/bandwidth costs during decoding, while **Multi-head Latent Attention (MLA)** (prominently used in DeepSeek models) targets KV-cache reduction via low-rank compression of the KV state. citeturn4search0turn9view1turn11view0

A third thesis is that “parameters” and “compute per token” diverge in sparse architectures: **Mixture-of-Experts (MoE)** systems can activate only a fraction of parameters per token, improving compute economics, while still requiring large **total parameter storage** somewhere (on one device for single-GPU, or sharded across devices for multi-device). Therefore, memory planning must distinguish **total parameters** vs **active parameters per token**, and must also account for MoE-specific parallelism/communication modes (tensor vs expert parallelism). citeturn10search2turn9view2turn11view3

## Memory accounting foundations for inference

### A decomposed memory budget

For a single LLM process, a practical top-level accounting identity is:

\[
M_{\text{total}} \;\approx\; M_{\text{weights}} \;+\; M_{\text{KV}} \;+\; M_{\text{act}} \;+\; M_{\text{tmp}} \;+\; M_{\text{runtime}} \;+\; M_{\text{headroom}}
\]

The key failure mode of “parameters × bytes” is that it approximates only \(M_{\text{weights}}\), while \(M_{\text{KV}}\) and \(M_{\text{runtime}}\) can dominate depending on **context length, concurrency, and backend choices**. citeturn0search0turn12search0turn2search13

### Prefill vs decode as a memory-shaping primitive

LLM serving and interactive local inference naturally split into:

- **Prefill**: process the full prompt sequence in parallel and build the initial KV cache. Prefill is widely characterized as **compute-bound** in typical GPU regimes. citeturn12search0turn12search3  
- **Decode**: generate tokens autoregressively; each step attends to an ever-growing cached state. Decode is widely characterized as **memory-bandwidth-bound**, driven by repeated reads of weights and KV-cache data. citeturn4search0turn9view1turn12search0

This distinction matters because:
- **Peak \(M_{\text{act}}\) and \(M_{\text{tmp}}\)** are usually shaped by prefill kernels and sequence length.
- **Steady-state \(M_{\text{KV}}\)** is shaped by decode, context growth, and concurrency, and can persist for the lifetime of a session/request. citeturn12search0turn9view1

### Component-level breakdown

The table below defines the major memory components and their scaling behavior. (Symbols are defined in the next section.)

| Component | What it stores | Dominant scaling variables | A useful first-order scaling law | Why “param × bytes” misses it |
|---|---|---:|---|---|
| \(M_{\text{weights}}\) | Model parameters (including embeddings / LM head if untied) | total parameter count \(P\) | \(M_{\text{weights}} \approx P \cdot b_w\) | This is the *only* term “param × bytes” tries to represent |
| \(M_{\text{KV}}\) | Cached attention state (keys & values) for prior tokens, per layer | layers \(L\), cached tokens \(S\), batch/concurrency \(B\), KV heads \(H_{kv}\), head dim \(d_h\) | \(M_{\text{KV}} \propto B\cdot S\cdot L\cdot H_{kv}\cdot d_h\) | Can grow linearly with context and number of live sessions; often dominates long-context serving citeturn8search7turn11view0 |
| \(M_{\text{act}}\) | Intermediate activations (hidden states, Q/K/V projections, MLP intermediates) | \(B\), \(S\), \(d_{\text{model}}\), backend kernel strategy | backend-dependent; often \(\propto B\cdot S\cdot d_{\text{model}}\) times a constant factor | Peak can be governed by fused kernels / implementation even when weights fit citeturn3search0turn12search9 |
| \(M_{\text{tmp}}\) | Temporary workspaces: GEMM scratch, attention softmax scratch, compilation buffers | kernel selection, sequence shapes, backend | backend-dependent; not reliably inferable from model config alone | Often creates “mysterious” OOM even when \(M_{\text{weights}}+M_{\text{KV}}\) fits citeturn3search0turn12search17 |
| \(M_{\text{runtime}}\) | Framework overhead: allocator caches, graph capture buffers, metadata, paging structures | backend/runtime | varies; can include large reserved pools | Can be large because allocators intentionally reserve memory for performance citeturn2search13turn12search17 |
| \(M_{\text{headroom}}\) | Unallocated capacity required to avoid OOM from fragmentation and spikes | allocator + workload | not a model property; an operational requirement | Fragmentation and reserved-vs-allocated behavior can make “free memory” smaller than expected citeturn2search5turn2search13turn2search3 |

### Fragmentation, reservation, and why “free VRAM” is not a scalar

Even after you correctly estimate the *bytes of tensors*, you can still OOM due to allocator behavior:

- Many inference stacks (notably those built on PyTorch) use a **caching allocator** that retains freed blocks for speed; as a result, “reserved” memory can exceed “allocated” memory and appear “used” in external tools. citeturn2search13turn2search9  
- Allocators may **round allocation sizes** to reduce fragmentation; the PyTorch memory statistics explicitly discuss fragmentation and rounding tradeoffs. citeturn2search5  
- At the GPU runtime level, memory-pool mechanisms (e.g., stream-ordered allocators) exist to reduce overhead and fragmentation, underscoring that fragmentation is a first-class issue in real deployments. citeturn2search3turn2search11  
- KV-cache memory itself is prone to *waste/fragmentation* under variable-length requests; systems like vLLM propose **PagedAttention** explicitly to reduce KV-cache waste and fragmentation by paging the KV cache. citeturn0search0turn0search11

## KV cache and context window effects

### Notation and base KV cache formula

Define:

- \(B\): number of concurrent sequences (or “live requests”) whose KV state must be retained  
- \(S\): cached token count per sequence (prompt + generated so far, up to the effective context window)  
- \(L\): transformer layer count  
- \(H_q\): number of query heads  
- \(H_{kv}\): number of **distinct** KV heads stored (equals \(H_q\) for MHA; smaller for GQA/MQA)  
- \(d_h\): per-head dimension (often \(d_h = d_{\text{model}}/H_q\))  
- \(b_{kv}\): bytes per KV element (depends on dtype/quantization; handled in Part 2)

For standard transformer decoding with KV caching, the cache stores **both keys and values**, giving:

\[
M_{\text{KV}} \;=\; 2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}
\]

The necessity and cost driver here is explicit in the MQA and GQA literature: incremental decoding becomes bottlenecked by repeatedly loading large KV tensors; reducing KV cache size reduces both capacity and bandwidth pressure. citeturn4search0turn9view1

### Long-context scaling: why KV cache can dominate

A critical property of the KV cache is that it grows **linearly** with cached tokens \(S\) and concurrency \(B\). Memory planning errors therefore compound when a system evolves from “single-user chat” to “multiple concurrent sessions,” or from “4K context” to “128K context.”

Recent systems and analyses explicitly call out KV cache as a dominant bottleneck in long-context regimes, because each generated token requires access to the prefilled KV cache. citeturn8search7turn9view1turn12search0

A useful normalization is the **KV bytes per cached token per sequence**:

\[
m_{\text{KV/token}} \;=\; \frac{M_{\text{KV}}}{B\cdot S}
\;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}
\]

This quantity is purely architectural (plus dtype), and it is the number you multiply by **total cached tokens across all live sessions** to get steady-state KV occupancy. citeturn9view1turn11view0

### A concurrency-aware form for real serving

In practice, sequences have different cached lengths \(S_i\). Then:

\[
M_{\text{KV}} \;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}\;\cdot\;\sum_{i=1}^{B} S_i
\]

This “sum of lengths” form is operationally useful: memory pressure is driven by **aggregate cached tokens** across sessions, which is exactly why KV-cache management systems emphasize reducing waste and enabling flexible sharing/packing of KV blocks. citeturn0search0turn0search11

### Context window effects beyond KV cache

While KV cache is usually the largest *persistent* context-linked term, context length also increases:

- Prefill compute and intermediate storage needs, particularly in attention kernels that must traverse long sequences (the classic attention cost pressure that IO-aware kernels like FlashAttention target). citeturn3search0turn12search3  
- The attractiveness of paging/virtual-memory-inspired KV management (PagedAttention) to control fragmentation and “near-zero waste” KV allocation under variable-length contexts. citeturn0search0turn0search11  

## Model-side factors that drive memory

### Core architectural parameters and what they control

Inference memory is determined by a small number of architectural scalars (plus dtype/runtime). The table below maps LLM spec fields to the memory terms they influence.

| Model spec factor | Where it appears in config | Primary memory term(s) impacted | Key scaling relationship |
|---|---|---|---|
| Total parameters \(P\) | “#params”, often decomposable into embeddings + blocks | \(M_{\text{weights}}\) | \(M_{\text{weights}} \approx P \cdot b_w\) |
| Layers \(L\) | \(n\_\text{layers}\) | \(M_{\text{KV}}, M_{\text{act}}\) | \(M_{\text{KV}} \propto L\); many activation/temporary terms also scale \(\propto L\) citeturn4search0turn11view0 |
| Model width \(d_{\text{model}}\) | hidden size / embedding dim | \(M_{\text{act}},\; M_{\text{KV}}\) (via \(d_h\)) | if \(d_h=d_{\text{model}}/H_q\), then \(M_{\text{KV}} \propto H_{kv}\cdot d_{\text{model}}/H_q\) citeturn9view1turn11view0 |
| Attention head count \(H_q\) | num attention heads | KV size (through \(d_h\) and \(H_{kv}\)) + attention kernel shapes | \(d_h = d_{\text{model}}/H_q\) (typical) citeturn7search0turn11view0 |
| KV head count \(H_{kv}\) | explicit “num_kv_heads” or implied by attention type | \(M_{\text{KV}}\) | \(M_{\text{KV}} \propto H_{kv}\) (dominant in long context) citeturn4search0turn9view1 |
| Context length \(S_{\max}\) | max positions / rope scaling / context window | \(M_{\text{KV}}\) ceiling; prefill workspace pressure | \(M_{\text{KV}} \propto S\) up to \(S_{\max}\) citeturn8search7turn11view0 |
| Vocabulary size \(V\) and weight tying | tokenizer vocab / tied embeddings | \(M_{\text{weights}}\) | embeddings scale as \(V\cdot d_{\text{model}}\) (twice if input/output untied) citeturn7search0 |
| Multimodal tokenization (image/video/audio tokens injected) | VLM/MLLM design choices | \(M_{\text{KV}}\) through increased effective \(S\) | visual/audio tokens increase the sequence length the LLM must cache citeturn6search2turn6search3 |

### Multimodal extensions change memory through two levers

Multimodal LLMs increase inference memory by:

1. **Adding extra parameterized towers** (e.g., a vision encoder + projector) to \(M_{\text{weights}}\). LLaVA explicitly connects a vision encoder and an LLM via visual instruction tuning. citeturn6search2turn6search0  
2. **Inflating effective sequence length** \(S\) by injecting visual/audio tokens into the LLM stream, which increases \(M_{\text{KV}}\) linearly with the number of injected tokens. Qwen-VL similarly describes a “visual receptor” and interface that extends a language model with visual capacity. citeturn6search3turn6search7  

Audio encoders (e.g., encoder–decoder speech models) follow the same principle: additional encoder parameters and recurrent attention over encoded representations. citeturn6search1turn6search13

### Adapters and LoRA as an additive weight term

Low-rank adaptation methods insert additional trainable matrices into transformer layers. The LoRA formulation explicitly injects low-rank decomposition matrices while freezing base weights, changing the parameterization but keeping the backbone intact. citeturn2search0

For memory estimation, LoRA can be treated as an additive parameter term:

\[
M_{\text{weights}} \;\approx\; (P_{\text{base}} + P_{\text{adapter}})\cdot b_w
\]

with \(P_{\text{adapter}}\) depending on which projections are adapted and the chosen rank \(r\). Even when adapter parameters are small relative to the base model, they are not “free” in a strict memory budget. citeturn2search0

## Attention mechanisms and KV cache economics

KV cache is the most architecture-sensitive inference memory term, and attention design is the primary lever that changes its scaling.

### MHA vs MQA vs GQA: sharing KV heads

**Multi-Query Attention (MQA)** was proposed specifically to share keys/values across heads to reduce the KV tensors and memory bandwidth cost during decoding. citeturn4search0  
**Grouped-Query Attention (GQA)** generalizes this by using an intermediate number of KV heads (groups), trading quality for memory/bandwidth more smoothly. citeturn9view1

A convenient way to express KV reduction for models where \(d_h = d_{\text{model}}/H_q\) is:

\[
\frac{M_{\text{KV}}}{M_{\text{KV}}^{\text{MHA}}}
\;=\;
\frac{H_{kv}}{H_q}
\quad
\text{(same \(B,S,L,d_{\text{model}},b_{kv}\))}
\]

because MHA typically has \(H_{kv}=H_q\). This relationship is exactly what GQA formalizes as an interpolation between MHA and MQA. citeturn9view1turn11view0

### MLA: low-rank compression of KV state

The DeepSeek-V2 paper motivates MLA as a direct response to the inference-time KV-cache bottleneck, referencing MQA and GQA as prior KV-reduction approaches and proposing MLA to reduce KV cache while maintaining (or improving) capability. citeturn9view0turn11view0

The core mechanism is **low-rank joint compression** of keys and values into a latent vector \(c\) of dimension \(d_c\), caching \(c\) instead of full \(K/V\):

\[
c_t = W^D h_t,\quad
k_t = W^U_k c_t,\quad
v_t = W^U_v c_t
\]

and during inference “MLA only needs to cache \(c\)” (plus additional state needed for positional encoding, discussed below). citeturn9view0turn11view0

A further complication addressed in the same section is that RoPE-style positional mechanisms can interact with the low-rank factorization; DeepSeek-V2 proposes a **decoupled RoPE** strategy that introduces an additional shared key component that must also be cached, yielding a total cached state that includes both latent \(c\) and a decoupled key component. citeturn11view0

### Comparative KV cache formulas by attention type

The following table expresses **KV cache element-count scaling per token** (ignoring dtype), which is often the clearest way to see architectural differences. Let \(d_h\) be per-head dimension, \(H_q\) query heads, \(G\) GQA groups (\(H_{kv}=G\)), and MLA latent dimension \(d_c\). For MLA with decoupled RoPE, let \(d_R\) denote the per-token decoupled key dimensionality (DeepSeek-V2 introduces such a cached decoupled key). citeturn9view1turn11view0

| Attention mechanism | Distinct KV heads stored \(H_{kv}\) | KV cache elements per token across all layers | Implication for \(M_{\text{KV}}\) |
|---|---:|---:|---|
| MHA | \(H_{kv}=H_q\) | \(E_{\text{KV/token}} = 2\cdot L\cdot H_q\cdot d_h\) | baseline KV memory; strongest “no-sharing” form citeturn7search0turn11view0 |
| MQA | \(H_{kv}=1\) | \(E_{\text{KV/token}} = 2\cdot L\cdot d_h\) | KV reduction factor \(\approx H_q\) vs MHA citeturn4search0turn9view1 |
| GQA | \(H_{kv}=G\) | \(E_{\text{KV/token}} = 2\cdot L\cdot G\cdot d_h\) | KV reduction factor \(\approx H_q/G\) vs MHA citeturn9view1 |
| MLA (DeepSeek-style) | not head-indexed KV; latent cached | \(E_{\text{KV/token}} \approx L\cdot(d_c + d_R)\) | KV cache can be comparable to “very small-group” GQA while keeping strong capability; DeepSeek-V2 states its KV cache matches GQA with ~2.25 groups under their chosen hyperparameters citeturn11view0turn9view0 |

The practical engineering takeaway is that **KV cache must be treated as an explicit spec** when selecting a model for a fixed memory budget. If you only know “7B/13B/70B,” you are missing the critical \(H_{kv}\) / \(d_c\) dimension that drives long-context feasibility. citeturn9view1turn11view0turn8search7

## Dense vs MoE and parallelism overheads

### Total vs active parameters

MoE models are sparse: only some expert parameters are used per token, but the model still has a large total parameter set. This is explicit in widely deployed MoE families:

- Mixtral reports that each token has access to a large parameter pool but uses a smaller set of **active parameters** per token during inference. citeturn10search2turn10search3  
- DeepSeek-V2 similarly distinguishes total parameters from activated parameters per token and couples this with MLA to address KV cache costs on long context. citeturn0search10turn9view0  
- Switch Transformers motivates sparsely-activated models as enabling extremely large parameter counts with constant computational cost, while identifying communication/training complexity as a key barrier. citeturn10search0  

For memory estimation, define:

\[
P_{\text{total}} = P_{\text{shared}} + P_{\text{experts,total}},
\qquad
P_{\text{active}} = P_{\text{shared}} + P_{\text{experts,active}}
\]

Then:

- **Weight memory** scales with \(P_{\text{total}}\): \(M_{\text{weights}} \approx P_{\text{total}}\cdot b_w\).
- **Compute and (some) activation work** scale closer to \(P_{\text{active}}\), because only selected experts run for a token. citeturn10search2turn10search0

This decoupling is the primary reason MoE can be compute-economical yet still memory-demanding.

### MoE routing and communication-aware design affects memory envelopes

MoE also introduces memory-impacting system concerns:
- Routing requires moving token representations to the selected experts; this motivates MoE-specific communication strategies and can require temporary routing buffers.
- DeepSeek-V2 explicitly discusses **bounding MoE-related communication costs** under expert parallelism (device-limited routing). citeturn11view3

### Parallelism modes: tensor parallelism vs expert parallelism

Even though Part 2 will cover platform/runtime details, MoE memory estimation already depends on which parallelism style is used because it changes **where weights live** and what must be replicated.

The DeepSpeed-MoE paper provides a clean conceptual distinction:

- **Tensor-slicing / tensor parallelism**: splits individual operators across devices and requires **all-reduce** communication.
- **Expert parallelism**: places experts across devices without splitting individual expert operators, and requires **all-to-all** communication. citeturn9view2turn8search9

A practitioner-oriented comparison is:

| Parallelism type | What is partitioned | Primary comm pattern | First-order per-device weight memory effect | Why it matters for memory requirement |
|---|---|---|---|---|
| Tensor parallelism (TP) | linear algebra inside layers | all-reduce / all-gather around sharded matmuls | per-device \(M_{\text{weights}}\) decreases roughly \(\propto 1/T\) for sharded tensors, but some parameters may replicate | Enables fitting larger dense components, but adds comm buffers and synchronization surfaces citeturn8search0turn9view2 |
| Expert parallelism (EP) | experts are distributed across devices | all-to-all for routing | each device stores only its assigned experts, so expert weights scale \(\propto 1/E\) in ideal sharding | Critical for fitting huge expert pools; introduces routing buffers and communication overheads citeturn9view2turn8search9 |

A subtle but important inference-specific detail is also noted in the GQA literature: some sharding strategies can replicate KV heads across partitions; GQA is partly motivated by reducing such waste by using more than one KV head (but still fewer than MHA). citeturn9view1

### Where speculative decoding fits in the memory model

Speculative decoding methods accelerate decoding by using an additional “draft” model and then verifying with the target model, which implies that memory budgeting may need to consider:

\[
M_{\text{total}} \;\approx\; M_{\text{target}} \;+\; M_{\text{draft}} \;+\; \Delta M_{\text{spec}}
\]

where each model contributes its own weights (and potentially its own KV cache if both are run autoregressively). This follows directly from the algorithms’ structure: a smaller model proposes multiple tokens and the larger model validates them. citeturn1search7turn1search3

## Provide Part 2

If you want me to continue, please provide **Part 2** (platform-specific and runtime-specific factors: GPU/TPU/Apple/CPU memory behavior; quantization and precision; backend differences such as paged attention vs contiguous KV; and a step-by-step estimation framework with worked examples).


# Estimating Local LLM Inference Memory: Precision, Quantization, and KV Cache

## Precision and quantization taxonomy for inference memory

A deployment-grade memory estimate must separate **storage precision** (how tensors are stored in memory) from **compute precision** (how kernels operate internally) and **accumulation precision** (what dtype partial sums use). This is not just semantics: many “low-bit” approaches are **weight-only** (W\*A16-style), meaning weights shrink but activations and KV cache retain 16-bit footprint, shifting the memory bottleneck toward KV in long-context or multi-user serving. citeturn3search7turn11search0turn1search1

The canonical bytes-per-element facts (for frameworks like PyTorch) are: FP32 is 32-bit; FP16 and BF16 are 16-bit. citeturn11search0turn11search15

### Core dtypes requested and their “first-order” storage costs

| Storage dtype | Bits / element | Bytes / element | First-order weight memory if all weights stored in this dtype | Practically important caveats |
|---|---:|---:|---|---|
| FP32 | 32 | 4 | \(M_{\text{w}} \approx 4P\) bytes | Rare for inference due to size/bandwidth; matmuls may not run as FP32 end-to-end (e.g., TF32 on some NVIDIA paths) depending on backend/hardware. citeturn11search0turn11search24 |
| FP16 | 16 | 2 | \(M_{\text{w}} \approx 2P\) bytes | Smaller range than BF16; some stacks keep layernorm/residual ops in FP32 even when weights are FP16 (mixed precision). citeturn11search0turn11search14 |
| BF16 | 16 | 2 | \(M_{\text{w}} \approx 2P\) bytes | Same exponent bits as FP32 (range advantage); widely used for training/serving; still 2 bytes/weight. citeturn11search0turn11search15 |
| INT8 | 8 | 1 | \(M_{\text{w}} \approx 1P\) bytes (idealized) | Real systems typically need scaling metadata; “LLM.int8()” is mixed precision for outliers (some work done in FP16) and thus not pure 1 byte/weight in practice. citeturn3search0turn3search9turn3search1 |

Here \(P\) is the **total parameter count** stored (including embeddings / LM head if untied). citeturn6view0turn11search0

## Weight memory estimation under different precisions and why naive estimates fail

### Baseline formula: full-precision storage

If every parameter is stored at \(b\) bits (e.g., 16 for FP16/BF16), then:

\[
M_{\text{weights,ideal}} \;=\; P \cdot \frac{b}{8} \quad \text{bytes}
\]

\[
M_{\text{weights,GiB}} \;=\; \frac{M_{\text{weights,ideal}}}{2^{30}}
\]

The “parameter count × bytes” heuristic is exactly this formula—useful but incomplete, because modern inference almost never stores *only* the raw weight tensor values without additional overhead (quantization metadata, alignment, mixed-precision exceptions) and because weights are only one term in the full inference memory budget. citeturn6view0turn11search5turn10view0

### Group-wise quantization: effective bits-per-weight (bpw) matters more than nominal bit-width

Most practical PTQ schemes store **packed low-bit integers** plus **scale (and sometimes zero-point)** per group/channel. A common estimator is:

\[
b_{\text{eff}} \;=\; b_{\text{q}} \;+\; \frac{b_{\text{scale}} + b_{\text{zp}}}{g}
\]

\[
M_{\text{weights}} \;\approx\; P \cdot \frac{b_{\text{eff}}}{8}
\]

Where:
- \(b_{\text{q}}\) is the nominal quant bits (e.g., 4 for INT4-like weight-only formats),
- \(g\) is the group size (#weights sharing scale/zero),
- \(b_{\text{scale}}\) is typically FP16 scale storage if saved as FP16 (16 bits),
- \(b_{\text{zp}}\) is often 8 bits for an INT8 zero-point in asymmetric schemes.

A concrete worked overhead example from a recent quantization survey: if each group stores \(s\) in FP16 (16 bits) and \(z\) in INT8 (8 bits), then \(b_{\text{scale}} + b_{\text{zp}} = 24\) bits. For group size \(g=128\), overhead is:

\[
b_{\text{overhead}} \;=\; \frac{24}{128} \;=\; 0.1875 \;\text{bpw}
\]

So a “4-bit” weight-only model can effectively be \(\approx 4.1875\) bpw just from scale/zero metadata, before considering padding/alignment. citeturn6view0

### Why real weight memory exceeds naive theory

The gap between \(P \cdot b/8\) and observed memory comes from several backend- and format-dependent causes:

| Cause of deviation | What happens | Direction of error | Where it shows up in practice |
|---|---|---|---|
| Quantization metadata | Scales/zero-points (per-group/per-channel) add persistent bytes | Underestimation | Any GPTQ/AWQ-style group quant; GGUF block/super-block metadata; INT8 schemes with per-channel scales. citeturn6view0turn7view0turn1search3 |
| Mixed-precision exceptions | Some tensors kept higher precision (e.g., embeddings/LM head, outlier columns) | Underestimation | “LLM.int8()” uses a mixed decomposition for outliers; frameworks often skip quantizing specific modules for stability. citeturn3search0turn3search9turn3search1 |
| Packing + alignment | Weights stored with hardware-friendly packing/alignment (row/block padding) | Usually underestimation | GGUF explicitly supports optional alignment for efficient access; many kernels require aligned dimensions. citeturn1search14turn1search3 |
| Backend allocator behavior | The runtime reserves extra memory pools/caches, so “used” memory includes non-tensor blocks | Appears as underestimation when comparing to “nvidia-smi” | PyTorch uses a CUDA caching allocator; reserved memory can remain “used” after tensors are freed. citeturn11search5turn11search26turn11search9 |
| Temporary dequant / workspace buffers | Kernels allocate scratch/workspace; dequant may happen into tiles/buffers | Underestimation of peak memory | Backend-dependent (e.g., TensorRT-LLM engine workspace; fused kernels); can cause OOM at runtime even if static weights “fit.” citeturn3search7turn10view1 |

The operational implication: you should estimate **persistent tensor bytes** (weights + KV) and then add a backend-specific **runtime overhead + headroom factor**, rather than trying to “predict” exact peak bytes from model metadata alone. citeturn11search5turn10view0turn10view1

## Quantization formats and how backend choice changes effective memory

This section maps the requested formats to *what is actually stored*, *what remains high precision*, and *what it means for memory planning*.

### Modern weight formats requested

| Format / method | What it is | Typical quant target | Key memory property for estimation | Primary sources |
|---|---|---|---|---|
| GPTQ | One-shot, post-training, weight-only quant using approximate second-order info; widely used at 3–4 bits | W3–W4 (weight-only) | Memory is driven by \(b_{\text{eff}}\) = quant bits + (scale/zero)/group; kernels/backends decide whether extra buffers exist | citeturn0search2turn5view1turn3search11 |
| AWQ | Activation-aware weight-only quant: uses activation statistics to protect salient channels via scaling | Often W4A16 | Still weight-only: KV cache and activations remain (usually) FP16/BF16; memory savings mainly from weights | citeturn8search0turn3search3turn0search1 |
| EXL2 | A quantized model format used by ExLlamaV2 allowing mixed bitrates and variable group sizes | Mixed 2–8 bpw average | Estimation uses **average bpw** across layers (not uniform b); additional metadata depends on grouping; very flexible size/quality trade | citeturn8search1turn7view0turn6view0 |
| GGUF | File format for GGML-based inference (e.g., llama.cpp); stores tensors + metadata + quant types | Many (2–8 bit integer types, FP16/BF16/FP32 also possible) | You estimate by the **effective bpw of the chosen GGML quant type**, not by “GGUF” itself; format contains metadata and optional alignment | citeturn1search3turn1search14turn1search7 |
| bitsandbytes 8-bit / 4-bit (NF4/FP4) | Runtime quantization integrated with Transformers; NF4 introduced by QLoRA; compute dtype configurable | INT8; NF4/FP4 4-bit | Weights stored quantized + metadata; compute often BF16/FP16; some ops (outliers) may run higher precision | citeturn1search1turn1search0turn3search9turn3search1 |

### Ultra-low and “nonstandard” regimes requested

| Regime | Representative work | Key memory point for inference planning | Practical caveat |
|---|---|---|---|
| FP4 / NF4 | QLoRA introduces NF4 and “double quantization” of quantization constants | “4-bit” storage still has metadata; NF4 targets weight distributions and is used for quantized storage with higher-precision compute | QLoRA is primarily a finetuning method; inference pathways differ by backend and may not match training kernels. citeturn1search0turn1search1turn1search4 |
| 1.58-bit / ternary | BitNet b1.58 uses ternary weights \(\{-1,0,1\}\) (information \(\log_2 3\approx 1.58\) bits) | Weight storage can be dramatically smaller than 4-bit PTQ; if truly ternary-packed, \(M_{\text{w}}\approx P\cdot 1.58/8\) bytes + minimal metadata | BitNet is trained natively with specialized layers (BitLinear) and keeps other components at higher precision (e.g., 8-bit in experiments), so end-to-end inference memory is not “1.58-bit everywhere.” citeturn1search2turn1search13turn1search10 |

### Activation precision vs weight precision: why weight-only quant is a KV-dominated world

A recurring production pattern is:

- Weights are quantized (INT8/INT4/NF4/EXL2…), but
- Activations (hidden states) and KV cache are kept at FP16/BF16/FP8 depending on backend,
- Accumulations may use higher precision in matmul pipelines, depending on hardware and compiler choices. citeturn3search7turn3search14turn3search10

This matters because weight-only quant reduces \(M_{\text{weights}}\), but does not reduce the dominant long-context term \(M_{\text{KV}}\) unless you also use **KV cache quantization/compression** (covered below). citeturn2search3turn3search10turn10view0

### Backend effects that change “effective memory” even for the same model

The same “4-bit model” (same conceptual \(P\) and nominal bits) can have meaningfully different memory use depending on whether the backend:

1. **Stores pre-quantized weights (static PTQ)** vs **quantizes on load**. (Transformers + bitsandbytes can quantize during loading; GPTQ/AWQ often produce pre-quantized checkpoints.) citeturn1search1turn0search2turn8search0  
2. Uses **paged/block-based KV allocation** (reduces fragmentation and waste) vs **contiguous per-request KV preallocation** (can waste large fractions). vLLM’s PagedAttention is explicitly designed to eliminate KV waste/fragmentation by paging KV blocks. citeturn10view0turn10view1turn10view2  
3. Supports **KV cache quantization** (FP8/INT8/2-bit) vs not. vLLM documents FP8 KV cache quantization; TensorRT-LLM documents INT8 KV cache support; KIVI proposes tuning-free 2-bit KV quantization. citeturn3search10turn3search15turn2search3  
4. Has allocator behavior that **reserves memory pools** (e.g., PyTorch CUDA caching allocator), affecting the difference between “allocated” vs “reserved” memory. citeturn11search5turn11search26  

## KV cache sizing: exact formulas and scaling with architecture, precision, and concurrency

### What KV cache stores (practitioner view)

During autoregressive decoding, recomputing attention keys/values for all prior tokens would be prohibitively expensive, so inference stores per-layer \(K\) and \(V\) states for past tokens in a **KV cache**. vLLM’s paper highlights that KV cache memory is huge, grows dynamically, and can be heavily wasted by fragmentation without careful management. citeturn10view0turn10view1

### Exact KV cache memory formula (dense transformer with KV caching)

Define:

- \(B\): number of **concurrent live sequences** (users/sessions/requests that retain cache)  
- \(S\): cached sequence length (prompt + generated tokens kept in context)  
- \(L\): number of transformer layers  
- \(H_{kv}\): number of KV heads actually stored (equals \(H_q\) for MHA; smaller for GQA/MQA; MLA changes the structure but Part 1 covered the head-count effects) citeturn0search4turn10view1turn2search3  
- \(d_h\): head dimension (commonly \(d_h = d_{\text{model}}/H_q\))  
- \(b_{\text{KV}}\): **bytes per stored KV element** (dtype-dependent; FP16/BF16: 2 bytes; FP8: 1 byte; 2-bit: 0.25 bytes, ignoring metadata)

Then the **persistent** KV cache storage (keys + values) is:

\[
M_{\text{KV}} \;=\; 2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}
\]

The factor 2 is for storing both \(K\) and \(V\). The linear scaling in \(B\) and \(S\) is the most operationally important property: multi-user serving and long context expand KV memory linearly. citeturn10view0turn2search3turn3search10

A useful derived quantity is KV bytes per token per sequence:

\[
m_{\text{KV/token}} \;=\; \frac{M_{\text{KV}}}{B\cdot S}
\;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}
\]

This isolates architecture and dtype from workload. citeturn10view1turn2search3

### Variable-length multi-user serving: the aggregate-token form

In real serving, each live request \(i\) has cached length \(S_i\). KV cache becomes:

\[
M_{\text{KV}} \;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}} \cdot \sum_{i=1}^{B} S_i
\]

Operationally, **total cached tokens across live sessions** is the primary memory driver, which is exactly why vLLM focuses on memory management (paged blocks + scheduler) to support higher batch sizes without waste. citeturn10view0turn10view2

### Concurrency and multi-user impact

If you serve \(B\) users concurrently at roughly similar cached length \(S\), KV cache multiplies by \(B\). This is why weight-only quantization often fails to “unlock” high concurrency: once weights are compressed, the KV cache becomes the dominant term at long context. citeturn10view0turn2search3turn3search10

A compact “capacity planning” inequality (ignoring activations/temp buffers) is:

\[
M_{\text{weights}} + M_{\text{KV}} \;\le\; M_{\text{device,usable}}
\]

and because \(M_{\text{KV}}\) is linear in \(\sum S_i\), you can solve for a **token budget**:

\[
\sum_{i=1}^{B} S_i
\;\le\;
\frac{M_{\text{device,usable}} - M_{\text{weights}}}{2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}}
\]

In practice you must subtract runtime overhead/headroom (allocator reservation, fragmentation, workspace), but this inequality is the right backbone for reasoning. citeturn11search5turn10view0turn11search26

## KV cache management and compression in real systems

### Continuous batching: fixes compute padding waste, not KV growth

Orca introduced **iteration-level scheduling** (“continuous batching”) so new requests can be merged into ongoing decode iterations, improving utilization compared to static batching. citeturn2search37turn2search2

However, continuous batching does **not** change the fundamental KV scaling laws above; it primarily improves throughput/latency tradeoffs by reducing compute fragmentation. Systems still require KV cache memory proportional to total live tokens unless they also employ paging, eviction, or compression strategies. citeturn10view0turn2search37

### PagedAttention (vLLM): bounded waste via block-based KV allocation

vLLM’s PagedAttention partitions KV cache into fixed-size **KV blocks** (“pages”) so blocks need not be physically contiguous. This is explicitly designed to alleviate internal fragmentation (small blocks allocated on demand) and eliminate external fragmentation (uniform block sizes), enabling near-zero waste KV memory management. citeturn10view0turn10view1turn10view2

If block size is \(B_{\text{blk}}\) tokens, each sequence of length \(S\) needs:

\[
N_{\text{blk}} \;=\; \left\lceil\frac{S}{B_{\text{blk}}}\right\rceil
\]

and the **reserved** KV capacity per sequence is \(N_{\text{blk}} \cdot B_{\text{blk}}\) tokens (not necessarily equal to \(S\)). Wasted “padding tokens” per sequence is bounded by:

\[
W_{\text{tokens}} \;=\; N_{\text{blk}}\cdot B_{\text{blk}} - S
\quad\Rightarrow\quad
0 \le W_{\text{tokens}} < B_{\text{blk}}
\]

Thus, paged KV allocation makes per-sequence waste \(O(B_{\text{blk}})\) rather than \(O(S_{\max}-S)\) as in naive max-context preallocation. vLLM reports that existing systems can use only a minority of reserved KV memory for token states due to fragmentation/waste, motivating this design. citeturn10view0turn10view1

### Prefix / prompt caching: reducing multi-user KV footprint via KV reuse

Prefix caching reuses KV blocks for identical prefixes so later requests can skip recomputing the shared part and reuse its KV cache. vLLM documents **automatic prefix caching** as caching KV-cache blocks of processed requests and reusing them when a new request shares the same prefix; it is widely used in practice and does not change model outputs. citeturn2search5turn2search8

A simple memory model for shared system prompt:

- Let \(S_{\text{shared}}\) tokens be an identical prefix across \(B\) requests.
- Without prefix caching, KV memory includes \(B \cdot S_{\text{shared}}\).
- With perfect prefix reuse, KV memory includes only \(1 \cdot S_{\text{shared}}\) for that prefix plus per-request unique suffixes.

Approximate KV memory saved (if the shared prefix KV is stored once rather than \(B\) times):

\[
\Delta M_{\text{KV}} 
\;\approx\;
2 \cdot (B-1)\cdot S_{\text{shared}} \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}
\]

In block-based systems, reuse is typically in units of KV blocks, so the effective reusable shared length may be \(\lfloor S_{\text{shared}}/B_{\text{blk}}\rfloor \cdot B_{\text{blk}}\) tokens depending on implementation. vLLM’s documentation emphasizes reuse of KV-cache blocks and references a design page for details. citeturn2search5turn2search0turn10view1

### KV cache quantization: shrinking the dominant long-context term

KV cache quantization compresses \(b_{\text{KV}}\) in \(M_{\text{KV}}\), directly increasing the max storable cached tokens and/or concurrency for a fixed device memory.

Representative approaches:

| Approach | Stored KV precision | First-order KV memory reduction vs FP16/BF16 | Notes relevant to estimation |
|---|---:|---:|---|
| Baseline | FP16/BF16 (2 bytes) | \(1\times\) | Most common default. citeturn11search0turn10view0 |
| FP8 KV cache (vLLM) | FP8 E4M3 (1 byte) | \(\approx 2\times\) | vLLM documents that FP8 KV cache quantization reduces memory footprint and increases tokens storable in cache, improving throughput. citeturn3search10turn2search5 |
| INT8 / mixed KV (TensorRT-LLM) | INT8 KV cache (implementation-specific) | \(\approx 2\times\) (ideal) | TensorRT-LLM architecture docs list INT8 KV cache as a supported quantization capability (details depend on recipe). citeturn3search15turn3search7 |
| 2-bit KV cache (KIVI) | 2-bit (keys per-channel, values per-token) | \(\approx 8\times\) (idealized) | KIVI proposes tuning-free 2-bit KV cache quantization; reports reduced peak memory (including weights) and higher throughput from enabling larger batch sizes. citeturn2search3turn2search7 |

In estimation terms, KV cache quantization modifies \(b_{\text{KV}}\) (and may add metadata overhead). For instance, switching FP16 \((b_{\text{KV}}=2)\) to FP8 \((b_{\text{KV}}=1)\) halves the KV term in:

\[
M_{\text{KV}} \;=\; 2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}
\]

while keeping weights unchanged. citeturn3search10turn2search3turn10view1

## Concrete OOM example: fits at 4K context, OOM at 128K

This example is intentionally “calculator-ready”: you can change numbers to match your target model.

### Assumptions

- Device usable memory (after runtime overhead/headroom): \(M_{\text{device,usable}} = 24\) GiB (typical single-GPU budget class). (Headroom caveat: allocator reservation and fragmentation can reduce usable space; PyTorch explicitly notes reserved memory can appear as “used” in monitoring tools.) citeturn11search5turn11search26  
- Model (representative mid-size transformer):
  - \(L = 40\) layers  
  - \(d_{\text{model}} = 5120\)  
  - \(H_q = 40 \Rightarrow d_h = 128\)  
  - GQA with \(H_{kv} = 8\) (KV heads fewer than query heads) citeturn10view1turn2search3  
- KV dtype FP16/BF16: \(b_{\text{KV}} = 2\) bytes. citeturn11search0  
- Concurrency \(B = 1\) live sequence (single-user).  
- Weight storage: “4-bit weight-only” with group quantization overhead example:
  - effective \(b_{\text{eff}} \approx 4 + 0.1875 = 4.1875\) bpw (from the documented overhead example with FP16 scale + INT8 zero at group size 128). citeturn6view0  
  - parameter count \(P = 13 \times 10^{9}\) (13B-class).

### Step A: estimate weight memory

\[
M_{\text{weights}} \approx P \cdot \frac{b_{\text{eff}}}{8}
= 13\times 10^9 \cdot \frac{4.1875}{8}
\approx 6.80\times 10^9 \text{ bytes}
\approx 6.34 \text{ GiB}
\]

This is already larger than the naive “\(13\text{B}\times 4\text{ bits} = 6.05\text{ GiB}\)” estimate because metadata overhead pushes 4.0 bpw to 4.1875 bpw in this example. citeturn6view0turn7view0

### Step B: KV cache at 4K context

Let \(S = 4096\).

\[
M_{\text{KV}}(4K)
= 2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot b_{\text{KV}}
\]

\[
= 2 \cdot 1 \cdot 4096 \cdot 40 \cdot 8 \cdot 128 \cdot 2
= 671{,}088{,}640 \text{ bytes}
\approx 0.625 \text{ GiB}
\]

Total (weights + KV) \(\approx 6.34 + 0.63 \approx 6.97\) GiB, leaving substantial room for runtime and temp buffers on a 24 GiB card. vLLM explicitly notes that beyond weights, other data includes activations/ephemeral tensors, and KV cache management is critical for maximum batch. citeturn10view0turn11search5

### Step C: KV cache at 128K context

Let \(S = 131072\).

\[
M_{\text{KV}}(128K)
= 2 \cdot 1 \cdot 131072 \cdot 40 \cdot 8 \cdot 128 \cdot 2
= 21{,}474{,}836{,}480 \text{ bytes}
\approx 20.0 \text{ GiB}
\]

Now (weights + KV) \(\approx 6.34 + 20.0 \approx 26.34\) GiB, which exceeds 24 GiB **before** accounting for allocator reservation, runtime memory pools, and temporary workspaces—so OOM is expected even though the model “fit easily” at 4K. This illustrates the core KV reality: scaling context from 4K to 128K multiplies KV memory by \(128K/4K = 32\times\). citeturn10view0turn10view1turn11search26

### Step D: multi-user serving explodes the KV term linearly

If \(B=8\) concurrent sessions at 128K, KV becomes:

\[
M_{\text{KV}}(128K, B{=}8) \approx 8 \cdot 20.0 \text{ GiB} = 160 \text{ GiB}
\]

This is why production systems prioritize paging, reuse (prefix caching), and KV compression: concurrency and context are multiplicative drivers. citeturn10view0turn2search5turn2search3

## Provide Part 3

Please provide **Part 3** (platform-specific memory behaviors across NVIDIA/AMD/Apple/TPU/CPU, runtime/framework comparisons, and the complete pre-deployment estimation framework with “minimum/workable/recommended” sizing plus additional worked examples).

# Estimating Local LLM Inference Memory: Platform and Runtime Factors

## Cross-platform memory model and what changes by ecosystem

A correct pre-deployment memory estimate must be *platform-aware* because the same conceptual tensors (weights, KV cache, activations, workspaces) map onto materially different **memory hierarchies, allocation semantics, and “usable memory” ceilings** across discrete GPUs (separate VRAM), unified-memory SoCs (Apple silicon), CPU-only paging systems, and graph-compiled NPUs.

A practical way to formalize this is to separate:

- **Physical capacity** (advertised VRAM / installed RAM / unified memory)
- **Usable capacity** (capacity remaining after driver reservation, runtime pools, mandatory buffers, and safety headroom)
- **Effective capacity for low-latency inference** (capacity before paging / oversubscription / device↔host traffic dominates)

A cross-platform “usable memory” budget is:

\[
M_{\text{usable}} \;\approx\; M_{\text{physical}} \;-\; M_{\text{driver+OS}} \;-\; M_{\text{runtime\_reserve}} \;-\; M_{\text{static\_buffers}} \;-\; M_{\text{headroom}}
\]

Where each subtracted term is **platform-specific**:

- On **discrete GPUs**, driver/OS reservation (and sometimes ECC parity) reduce visible FB memory; frameworks also reserve pools; monitoring tools may show “used” memory even when buffers are returned to a pool. citeturn13view0turn16view2  
- On **Apple silicon**, the key limiter is not just “can allocate,” but “can allocate without affecting runtime performance,” an interface explicitly exposed by Metal as `recommendedMaxWorkingSetSize`. citeturn1search7  
- On **CPU-only**, the operating system’s page cache and swapping can create the illusion of “fit,” but with severe latency/throughput collapse once page faults dominate. Llama.cpp’s explicit guidance around `mmap`, `mlock`, and pageouts is a direct reflection of this. citeturn11view2turn4search0  
- On **NPUs/edge accelerators**, compilation and static planning often fix memory schedules based on graph shapes; first-run “compile/initialize” costs and static-shape constraints are common. citeturn14view1turn3search1  

The following table summarizes the key differences that matter for memory estimation in local LLM inference.

| Ecosystem | Memory pool(s) visible to the program | Dominant “hidden” memory consumers | What “fits” means | What “runs well” means |
|---|---|---|---|---|
| Discrete GPU (NVIDIA/AMD) | VRAM (FB) + optional pinned/managed host buffers | Driver reservation; runtime allocators/pools; preallocated KV pools; kernel/engine workspaces citeturn13view0turn16view2turn6view3 | \(M_{\text{weights}} + M_{\text{KV}} + M_{\text{act/tmp}} \le M_{\text{VRAM,usable}}\) | Avoid host paging/UM oversubscription; minimize interconnect traffic; keep KV in-device for decode unless explicitly trading latency for capacity citeturn0search15turn8view1 |
| Apple silicon | Unified memory shared by CPU+GPU | “Recommended” GPU working set limit; system memory pressure → compression/swap; GPU resource storage mode costs citeturn1search7turn7view2 | Allocation may succeed until memory pressure triggers paging; practical cap often near `recommendedMaxWorkingSetSize` citeturn1search7 | Stay below pressure thresholds (avoid swap); keep GPU working set under recommended limit to prevent stalls/instability citeturn7view2 |
| CPU-only | DRAM + page cache + swap | Page faults; file-system cache churn; NUMA locality | Can “fit” via mmap + swap, but may be unusably slow citeturn11view2turn3search3 | DRAM ≥ hot working set (weights accessed + KV + scratch); NUMA-aware placement for bandwidth-limited decode citeturn11view2 |
| NPUs & edge | Device SRAM/CMX + shared DRAM (varies) | Graph compile artifacts; static buffers; limited supported ops → fallback; static-shape requirements | Depends on compiler memory planning more than tensor arithmetic citeturn14view1turn3search1 | Stable shapes, supported op set, and staying within on-chip memory limits to avoid DRAM thrash |

## GPU ecosystems with discrete VRAM

### NVIDIA GPUs: VRAM vs usable VRAM, CUDA overhead, and “fits” vs “runs well”

#### VRAM vs usable VRAM in practice

On **entity["company","NVIDIA","gpu vendor"]** discrete GPUs, “Total VRAM” is not the same as **usable VRAM** for an LLM process:

- `nvidia-smi` documentation explicitly notes that reported total FB memory can be affected by **ECC parity** and that the **driver may reserve** memory even without active work. citeturn13view0  
- The same documentation also highlights that on some systems FB memory is managed by the OS and may not be released immediately (pages not released after process termination to enhance performance), which can cause reporting discrepancies and “sticky” usage. citeturn13view0  
- In the PyTorch/Accelerate “big model inference” guide, there is an explicit warning that the **first CUDA allocation** can load CUDA kernels and consume on the order of **1–2 GiB** depending on GPU, reducing practical headroom for model + KV. citeturn6view3  

A deployment-grade budgeting expression is:

\[
M_{\text{VRAM,usable}} \;\approx\; M_{\text{FB,total}} \;-\; M_{\text{ECC}} \;-\; M_{\text{driver\_reserve}} \;-\; M_{\text{CUDA\_init}} \;-\; M_{\text{allocator\_reserve}} \;-\; M_{\text{headroom}}
\]

Where \(M_{\text{CUDA\_init}}\) is often non-trivial in PyTorch-based stacks. citeturn6view3

#### “Fits” vs “runs well” on NVIDIA

A model can “fit” (weights + KV + buffers within VRAM), but still “run poorly” if achieving that fit relies on:

- **Host-device paging / oversubscription** mechanisms (e.g., Unified Memory oversubscription can “bridge” device and host address spaces but typically trades bandwidth/latency dramatically). The CUDA programming guide describes Unified Memory as enabling oversubscription and managed migration, which is a *capacity* tool, not a *latency* tool. citeturn0search15  
- **Heavy CPU↔GPU transfers** for offloaded layers or KV segments; decode is sensitive to bandwidth and repeated reads, so interconnect limitations become performance ceilings (even if capacity is technically sufficient). citeturn9view0  

A useful operational distinction is:

\[
\text{Fits} \equiv \max_t M_{\text{alloc}}(t) \le M_{\text{VRAM,usable}}
\]

\[
\text{Runs well} \equiv \text{Fits} \;\wedge\; \text{no paging/oversubscription} \;\wedge\; \text{interconnect traffic is not dominant}
\]

#### PCIe vs NVLink for multi-GPU sharding and offloading

When weights/KV are sharded or exchanged across GPUs (tensor parallel, pipeline parallel, KV exchange, etc.), interconnect bandwidth and topology matter. NVIDIA’s NVLink performance brief motivates NVLink specifically because PCIe can constrain multi-GPU applications; it describes NVLink as providing at least **80 GB/s** and “at least 5×” PCIe Gen3 x16 bandwidth in their assumptions, with discussion of multi-GPU peer-to-peer benefits. citeturn9view0  

A concise planning table:

| Interconnect | Typical role in local LLM inference | Memory-related implication | Practical takeaway |
|---|---|---|---|
| PCIe (host + peer-to-peer) | Baseline multi-GPU and CPU offload transport | Lower bandwidth increases the cost of sharding synchronization and CPU offload; encourages keeping KV local to each GPU | CPU offload over PCIe is often a last resort for latency-sensitive decode |
| NVLink | High-bandwidth GPU↔GPU fabric | Higher bandwidth reduces “tax” of model parallelism and KV movement; expands the regime where multi-GPU sharding is throughput-efficient citeturn9view0 | NVLink makes multi-GPU “fit-and-run-well” more attainable for large models |

### AMD GPUs: ROCm limitations and memory overhead semantics

On **entity["company","AMD","gpu vendor"]**, inference feasibility and memory behavior are strongly shaped by ROCm maturity and the supported OS/GPU matrix:

- ROCm’s compatibility matrix is explicitly **Linux-scoped** (“Applies to Linux”) and enumerates supported GPU targets and framework support—meaning “capacity planning” must include “is my GPU/OS/toolchain supported,” not just VRAM size. citeturn8view0  
- ROCm’s GPU memory documentation distinguishes host pageable, host pinned, and managed memory, emphasizing that host-resident pinned memory accessed in device kernels forces traversal over the host-device interconnect (e.g., PCIe) and is **not recommended for performance**; this is the same core constraint that makes CPU offload expensive. citeturn8view1  
- ROCm managed memory depends on HMM support and page-migration capability; lacking this, “managed” can degrade toward pinned-host behavior, again impacting the “runs well” boundary. citeturn8view1  

A ROCm-oriented budget mirrors CUDA, but with special attention to managed-memory capability and PCIe penalties:

\[
M_{\text{VRAM,usable}} \;\approx\; M_{\text{VRAM}} - M_{\text{driver/runtime}} - M_{\text{allocator\_reserve}} - M_{\text{headroom}}
\]

\[
\text{Runs well} \;\Rightarrow\; \text{avoid host-resident access paths during decode}
\]

Because ROCm documentation explicitly warns that traversing the host-device interconnect is much slower than on-device bandwidth. citeturn8view1

## Apple silicon: unified memory, working-set limits, memory pressure, and swap

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Apple silicon unified memory architecture diagram","Metal recommendedMaxWorkingSetSize concept diagram","Apple Activity Monitor memory pressure graph screenshot"],"num_per_query":1}

### Unified memory is capacity *and* contention

On **entity["company","Apple","apple silicon vendor"]** silicon, CPU and GPU access a shared physical pool. From the MLX documentation: “Apple silicon has a unified memory architecture. The CPU and GPU have direct access to the same memory pool,” and MLX is designed to take advantage of this. citeturn2search16turn2search12  

The most important planner’s consequence is: **GPU memory is not a separate silo**, so GPU allocations contend with CPU allocations, filesystem cache, and OS “wired” memory.

### “Can allocate” vs “can run fast”: `recommendedMaxWorkingSetSize`

Metal exposes `recommendedMaxWorkingSetSize` as “an approximation of how much memory, in bytes, this GPU device can allocate without affecting its runtime performance.” citeturn1search7  

This maps precisely to the practical distinction local LLM engineers observe:

- **Can allocate**: unified memory may allow allocations beyond the comfort zone.
- **Can run fast**: beyond the recommended working set size, the system is more likely to enter memory pressure regimes that induce paging/compression and GPU stalls.

Thus, for Apple silicon LLM sizing, treat:

\[
M_{\text{GPU,fast}} \;\approx\; \texttt{recommendedMaxWorkingSetSize}
\]

as the “runs well” memory ceiling, and treat total unified memory as the “absolute” ceiling (often unusable for low-latency inference once swap/compression becomes significant). citeturn1search7turn7view2  

### Memory pressure and swap as first-class inference constraints

Apple’s Activity Monitor documentation defines **Memory Pressure** as determined by free memory, swap rate, wired memory, and file cached memory; it explicitly surfaces swap usage and compression as part of overall memory health. citeturn7view2  

For local LLM inference, a unified-memory working set that pushes the system into heavy compression/swap will typically manifest as:

- Dramatically increased time-to-first-token (TTFT) as pages are faulted and decompressed.
- Decode “jitter” and throughput collapse due to repeated memory stalls.

Planning implication:

\[
\text{On Apple silicon:}\quad M_{\text{weights}} + M_{\text{KV}} + M_{\text{runtime}} \;\lesssim\; M_{\text{GPU,fast}}
\]

rather than merely \(\le\) physical unified memory. citeturn1search7turn7view2  

### MLX as a unified-memory-native stack

MLX emphasizes that arrays live in shared memory and can be operated on without explicit transfers; the GitHub README and official MLX docs both call out the unified memory model as a key differentiator. citeturn2search12turn2search16turn15search6  

From a memory accounting perspective, this means your estimator should not treat “CPU tensors” vs “GPU tensors” as disjoint pools; instead, budget a single pool with a **performance-sensitive working set limit** (`recommendedMaxWorkingSetSize`) and observe memory pressure aggressively during testing. citeturn1search7turn7view2  

## CPU-only systems: RAM sizing, mmap/paging, and when RAM compensates for low VRAM

### RAM estimation: “file size” is not the whole story

In CPU-only inference (or CPU-heavy offload), total resident memory is approximately:

\[
M_{\text{CPU,total}} \;\approx\; M_{\text{weights,resident}} \;+\; M_{\text{KV}} \;+\; M_{\text{tmp}} \;+\; M_{\text{runtime}} \;+\; M_{\text{OS\_headroom}}
\]

The key subtlety is \(M_{\text{weights,resident}}\): if weights are memory-mapped, physical residency depends on *which pages are touched* and OS prefetch behavior—not just the checkpoint file size.

### mmap behavior: load-on-demand is real, but page faults are the tax

The llama.cpp tooling documentation is explicit:

- By default, models may be memory-mapped; `--no-mmap` disables mapping.
- `--mlock` can lock mapped pages to prevent swapping.
- Disabling mmap slows load and can reduce pageouts if you are not using mlock; but if the model exceeds total RAM, turning off mmap can prevent loading. citeturn11view2turn4search0turn11view2  

Separately, general OS behavior for memory-mapped files is that pages are typically pulled in on-demand via page faults (with OS prefetch heuristics). citeturn3search3turn3search14  

Practical inference consequence:

- mmap can reduce *startup* and avoid reading the entire file upfront.
- But if inference touches most weights repeatedly (common), the “hot set” tends toward the whole model, and if RAM is insufficient the system will thrash (major page faults), destroying throughput.

### When RAM can compensate for low VRAM (and when it cannot)

RAM can “compensate” for low VRAM only in offload regimes where you accept:

- Lower throughput / higher latency from device↔host transfers (PCIe-limited on discrete GPUs), or
- CPU-only decode performance ceilings.

This is not theoretical—vendor docs explicitly warn about host-device interconnect penalties when device kernels access host memory (ROCm pinned memory guidance), which is the same constraint that makes CPU offload expensive. citeturn8view1turn9view0  

Therefore, for performance-critical local inference:

\[
\text{Use RAM for capacity (fit), not for speed (runs well), unless CPU-only is acceptable.}
\]

## NPUs and edge accelerators: graph compilers, static shapes, and reserved SRAM

Local LLM inference on NPUs/edge accelerators is typically constrained less by “raw tensor bytes” and more by:

- Static-shape compiler requirements
- Supported operator sets (fallback paths)
- Fixed memory arenas / on-chip SRAM limits
- Graph compilation and initialization costs

### Static-shape and compile/initialize constraints are common

In TensorFlow Lite GPU delegation, errors and warnings explicitly reference that some delegates only support **static-sized tensors**, and models with dynamic-sized tensors can fail delegation. citeturn3search1turn3search27  

In embedded/edge vendor documentation (NXP i.MX ML guide), there is an explicit note that the **first execution** using a delegate can take longer due to **computational graph compilation and initialization**, with subsequent iterations faster. citeturn14view1  

This aligns with a key estimator shift for NPUs:

- Memory usage is often determined at *compile time* based on fixed shapes and scheduling.
- “Maximum supported sequence length” or “maximum batch” may be hard-limited by compiled buffer schedules.

### Reserved SRAM / arena-style memory planning

While many vendor NPUs expose shared DRAM, they often rely on constrained on-chip memory (SRAM/CMX) for performance. Edge inference frameworks like TensorFlow Lite Micro explicitly target environments with “only a few kilobytes of memory” and severe resource constraints, underscoring that memory planning is a primary design axis. citeturn14view0  

Thus, for NPUs/edge, the pre-deployment procedure is less “compute KV bytes” and more:

1. Fix shapes (sequence length, batch) and export with static constraints.
2. Compile for target NPU; inspect compile reports for buffer sizes and fallback ops.
3. Validate whether attention/KV is supported or requires fallback (often the actual blocker for LLMs).

## Runtime stacks: how memory is allocated, accounted, and constrained

The same model and context can have radically different memory behavior depending on whether the runtime uses:

- **Contiguous KV** vs **paged/block KV**
- **Preallocation policies** (reserve for worst-case) vs **allocate-on-demand**
- **CPU/disk offload** for weights and/or KV
- **Device memory pooling** (reserved vs allocated)

### Comparative table of requested frameworks

| Runtime | Primary ecosystem | KV cache management | Preallocation behavior | CPU offload support | Prefix/prompt caching | Notes that materially affect memory planning |
|---|---|---|---|---|---|---|
| llama.cpp | CPU + CUDA/HIP + Metal/Vulkan | Configurable KV dtype (K/V can be f16/bf16/q8/q4/etc) and supports multi-GPU split modes where “layer” split includes KV across GPUs citeturn12view0turn12view1 | Preallocates buffers according to ctx/batch; operationally, ctx size directly drives KV buffer size (OOM at large ctx is a common failure mode) citeturn12view0turn12view1turn10search4 | Hybrid CPU+GPU inference for models larger than VRAM is explicitly described in backend docs citeturn6view2 | Server supports prompt caching/slot save-restore; request-level `cache_prompt` enables KV reuse for common prefix citeturn12view3turn15search13 | Memory mapping + mlock/no-mmap are first-class knobs affecting pageouts and load behavior citeturn11view2turn4search0 |
| vLLM | PyTorch + CUDA/ROCm | PagedAttention partitions KV into blocks to eliminate fragmentation; also supports prefix caching at KV-block granularity citeturn1search32turn15search0 | Pre-allocates GPU KV cache as a fraction of VRAM via `gpu_memory_utilization`; insufficient KV triggers preemption/recompute citeturn6view1turn1search1 | CPU offload and KV sharing appear via LMCache examples (disaggregated prefill / CPU offload / KV sharing) citeturn1search9 | Automatic Prefix Caching (APC) explicitly caches and reuses KV blocks for identical prefixes citeturn15search12turn15search0 | vLLM exposes “GPU KV cache size” and “maximum concurrency” planning outputs in docs citeturn5search9 |
| Hugging Face Transformers | PyTorch-centric | Typically contiguous KV (framework implementation dependent); KV is generally per-request and grows with context | Relies on PyTorch allocator behavior; big-model loading can use layer-wise dispatch across devices | CPU/disk offload via Accelerate `device_map` and `load_checkpoint_and_dispatch`; supports CPU-only offload and disk offload modes citeturn6view3turn1search22 | Prefix caching is not a default core feature in vanilla generate; requires additional serving layer | Accelerate documentation explicitly warns about first CUDA allocation consuming ~1–2GiB and recommends adjusting `max_memory` accordingly citeturn6view3 |
| MLX | Apple silicon-specific | Implementation-dependent; but arrays and model state reside in unified memory | Lazy computation and unified memory remove explicit transfer staging, but do not remove working-set limits | CPU/GPU not separated; unified pool | Higher-level caching depends on the app/framework built on MLX | MLX explicitly targets unified memory: arrays live in shared memory and can run on supported device types without transfer citeturn2search16turn15search6 |
| TensorRT-LLM | NVIDIA TensorRT engines | Uses paged KV cache pools; KV size and policy configurable via `KVCacheConfig`; KV is a major component of inference memory citeturn16view2turn16view0 | C++ runtime preallocates runtime/decoder buffers and preallocates paged KV pools; default KV may allocate ~90% of remaining free GPU memory unless constrained citeturn16view2turn16view0 | Primarily GPU-centric; “offload” means distributed/parallel rather than CPU KV spill in typical deployments | In-flight batching supported via scheduler/batch manager; memory must reserve enough KV pages for scheduled requests citeturn16view2turn15search15 | Activation memory is computed at build time based on max shapes and cannot be changed post-build; reducing build-time max shapes reduces activation memory citeturn6view0turn16view2 |
| exllamav2 | CUDA consumer GPU-focused | KV cache can be quantized; TP discussions indicate KV/cache handling and P2P exchange requirements in parallel setups citeturn5search1turn5search12 | Implementation tends to be VRAM-first; capacity is dominated by VRAM availability and cache configuration | Typically not positioned as CPU-offload-first | Application/server dependent | Treat as a “specialized high-throughput CUDA stack” whose memory behavior is tied to its quant format and kernel choices; validate per version on target GPU citeturn5search12turn5search1 |

### Important runtime-specific nuances that affect memory estimates

#### llama.cpp: explicit knobs that alter the memory equation

From the llama.cpp server documentation:

- KV cache dtype is configurable separately for K and V, with allowed values including `f16`, `bf16`, and multiple quant types (`q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`). This directly changes \(b_{\text{KV}}\) in the KV formula and thus the maximum context/concurrency that fits. citeturn12view0  
- `--n-gpu-layers` controls the maximum number of layers stored in VRAM; multi-GPU split modes include `layer` (default) which **splits layers and KV across GPUs**, and `row` which splits rows across GPUs (i.e., tensor slicing). citeturn12view1  
- Prompt caching is operationalized via endpoints for saving/restoring slot prompt caches; request-level `cache_prompt` enables KV reuse for identical prefixes, reducing repeated prefill and the multi-user memory footprint for shared system prompts. citeturn12view3turn15search13  

Additionally, llama.cpp’s “completion” documentation provides a highly actionable statement of mmap vs mlock behavior and explicitly calls out pageouts as the performance risk. citeturn11view2  

#### vLLM: memory planning is KV planning

vLLM documents that:

- It **pre-allocates GPU cache** based on `gpu_memory_utilization`; raising this increases KV cache capacity but reduces headroom for other allocations. citeturn6view1turn1search1  
- When there is insufficient KV cache space, vLLM can preempt/recompute (default in vLLM v1 is `RECOMPUTE`), explicitly tying performance stability to KV cache headroom and scheduling. citeturn6view1  
- Automatic Prefix Caching caches KV blocks of processed requests and reuses them for new requests with the same prefix; the design docs emphasize KV-block caching as “almost a free lunch” that avoids redundant prompt computation. citeturn15search0turn15search12  
- vLLM provides planner-friendly outputs (“GPU KV cache size” and “Maximum concurrency”) to translate KV capacity into request concurrency limits. citeturn5search9  

#### Transformers + Accelerate: weight placement is flexible; runtime overhead is real

Accelerate’s big model inference guide is unusually explicit about two issues that impact memory estimation:

- The first CUDA allocation can consume roughly **1–2 GiB** due to loading kernels, so usable memory is smaller than “VRAM size”; you should account for this in `max_memory` maps. citeturn6view3  
- Device maps should place parameters sequentially to avoid excessive cross-device transfers; this is effectively an *interconnect-pressure minimization* rule that impacts both performance and transient memory buffers. citeturn6view3  

#### TensorRT-LLM: build-time shapes fix activation memory; runtime preallocates KV pools

TensorRT-LLM’s memory documentation provides a production-grade decomposition:

- Inference-time GPU memory is dominated by **weights**, **internal activation tensors**, and **I/O tensors**, where KV cache is the major I/O footprint. citeturn6view0turn16view2  
- TensorRT precomputes the activation memory requirement at build time (based on max shapes across optimization profiles), which cannot be changed after engine build; hence, build-time max batch/seq/token choices directly determine activation memory. citeturn6view0turn16view2  
- The C++ runtime **pre-allocates paged KV cache pools** and allocates KV based on `KVCacheConfig`; if neither `maxTokens` nor `freeGpuMemoryFraction` is specified, KV cache may by default allocate **90%** of remaining free GPU memory—an aggressive preallocation policy that must be included in planning. citeturn16view0turn16view2  
- TensorRT-LLM also documents that `nvidia-smi` may show high memory occupation due to CUDA driver memory pools even after buffers are returned, and it describes use of the CUDA driver’s default memory pool (stream-ordered allocator) for buffer management. citeturn16view2  

## Request for the next phase

Please provide **Part 4**.

# Estimating Local LLM Inference Memory Requirements: Practical Pre-Deployment Toolkit

## Goals and operating assumptions

This phase synthesizes Parts 1–3 into a **pre-download, calculator-ready** method for predicting whether a local LLM deployment will (a) *fit*, (b) *run stably without OOM*, and (c) *run well* (avoid paging / pathological allocator behavior / KV-cache thrash). The core operational constraint remains that inference memory is the sum of **weights + KV cache + runtime/temporary buffers + safety headroom**, and the dominant term frequently shifts from weights to KV cache as **context length** and **concurrency** increase. citeturn7search1turn6search13turn5search0

Two platform/runtime facts heavily shape a “workable” memory budget:

- PyTorch-based CUDA stacks typically lose **~1–2 GiB** to CUDA kernel loading on first allocation (so “VRAM advertised” ≠ “VRAM usable”). citeturn6search0  
- Many high-throughput servers pre-allocate KV-cache pools as a fraction of VRAM (vLLM via `gpu_memory_utilization`; TensorRT-LLM via `freeGpuMemoryFraction` with a **90% default** if unconstrained). These policies can convert “fits on paper” into immediate OOM if you don’t budget for them. citeturn5search0turn5search2

## Inputs you can fetch before downloading model weights

You can compute high-fidelity memory estimates from **metadata** (typically a few KB) without downloading tens of GB of tensors.

### Minimal model metadata required

The following table is the smallest set of fields that makes KV-cache sizing exact (for standard KV caching) and weight sizing accurate to “bits-per-weight” assumptions.

| Category | Required fields | Where to obtain (typical) | Why it matters |
|---|---|---|---|
| Transformer shape | \(L\) (layers), \(d_{\text{model}}\), \(H_q\), \(H_{kv}\) (or `num_key_value_heads`), and/or \(d_h\) | `config.json` on model hub | KV cache scales as \(L \cdot H_{kv} \cdot d_h\). citeturn2view1turn2view0turn8search0 |
| Context capability | `max_position_embeddings` / max context | `config.json` and/or model card | Determines the *native* ceiling for \(S\) (though you can still allocate beyond it with rope scaling at quality risk). citeturn2view1turn2view0turn1search0 |
| Attention type | MHA vs GQA vs MQA (implied by \(H_{kv}\)) | `num_key_value_heads` conventions | KV memory reduction relative to MHA is \(\approx H_{kv}/H_q\). citeturn5search1turn2view1 |
| Total parameters | \(P\) (total parameter count) | Model card / release notes / hub listing | Weights memory is \(P \cdot b_{\text{eff}}/8\). |
| Multimodal towers | Vision encoder type, projector type, and whether vision encoder is frozen | Model/paper card | Adds additional weight memory and typically increases effective prompt token count. citeturn0search3turn4search17 |

### Example configs used in the Worked Examples section

All architecture fields for the examples below are taken directly from the published `config.json` files:

- Llama-3-8B Instruct (`hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=8`, `max_position_embeddings=8192`). citeturn2view1  
- Llama-3-70B Instruct (`hidden_size=8192`, `num_hidden_layers=80`, `num_attention_heads=64`, `num_key_value_heads=8`, `max_position_embeddings=8192`). citeturn2view2  
- Qwen2.5-32B (`hidden_size=5120`, `num_hidden_layers=64`, `num_attention_heads=40`, `num_key_value_heads=8`, `max_position_embeddings=131072`). citeturn2view0  
- Mixtral-8x7B (`hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=8`, `max_position_embeddings=32768`, `num_experts_per_tok=2`). citeturn8search0  
- LLaVA-1.5 composition (Vicuna-v1.5 + CLIP ViT-L/14 + MLP connector) is documented in multimodal literature; the LLM base for KV sizing here is Llama-2-7B (`hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=32`, `max_position_embeddings=4096`). citeturn4search17turn3search2  

Context-length variants matter: Meta states Llama 3 was trained on sequences of **8192 tokens** and uses GQA; the Llama 3.1 family expands context length to **128K**. citeturn1search0turn1search13

## Step-by-step estimation framework

### Step zero: define workload knobs precisely

Define the deployment target in workload variables:

- \(B\): **concurrent active sequences** (live sessions that retain KV)  
- \(S_i\): cached tokens per sequence \(i\); often \(S_i = S_{\text{prompt},i} + S_{\text{generated},i}\)  
- \(S_{\max}\): maximum allowed cached tokens per sequence (hard cap or service policy)  
- \(T = \sum_{i=1}^{B} S_i\): total cached tokens across the system  
- \(b_w\): effective **bits per weight** (depends on precision/quant format + metadata)  
- \(b_{KV}\): **bytes per KV element** (e.g., FP16/BF16: 2 bytes; FP8: 1 byte)  
- \(M_{\text{usable}}\): usable memory budget on the target device (after fixed overhead & reservation policies)

When using a server that preallocates memory pools, you must incorporate its policy as a first-class constraint:

- vLLM: “pre-allocates GPU cache using `gpu_memory_utilization` percent of memory.” citeturn5search0  
- TensorRT-LLM: default KV cache allocation can be **90% of remaining free GPU memory** if you don’t specify `maxTokens` or `freeGpuMemoryFraction`. citeturn5search2  

### Step one: compute weight memory

The baseline (idealized) weight-memory estimate is:

\[
M_{\text{weights,ideal}} \;=\; P \cdot \frac{b_w}{8}
\]

To account for group-wise metadata overhead, use an *effective bpw* model:

\[
b_w \;=\; b_q \;+\; \frac{b_{\text{scale}} + b_{\text{zp}}}{g}
\]

A concrete published example: if each group stores scale \(s\) in FP16 (16 bits) and zero-point \(z\) in INT8 (8 bits), then \(b_{\text{scale}}+b_{\text{zp}}=24\) bits. With group size \(g=128\), overhead is \(24/128=0.1875\) bpw. citeturn3search3turn3search6

Thus an “INT4” weight-only model often behaves like \(\approx 4.1875\) bpw under this specific grouping assumption (and alignment can add more). citeturn3search6

### Step two: compute KV cache memory exactly

For standard KV caching in a decoder-only transformer:

- \(L\): layers  
- \(H_{kv}\): KV heads (equals \(H_q\) for MHA, smaller for GQA/MQA)  
- \(d_h\): head dimension (often \(d_h = d_{\text{model}} / H_q\))  
- \(b_{KV}\): bytes per KV element (FP16/BF16: 2 bytes)

Per-sequence KV cache for one request of length \(S\):

\[
M_{\text{KV}}(B{=}1) \;=\; 2 \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV}
\]

For \(B\) concurrent sequences (different lengths):

\[
M_{\text{KV,total}} \;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV} \cdot \sum_{i=1}^{B} S_i
\]

These relationships are the core reason serving systems treat KV management as the bottleneck: KV is “huge,” grows dynamically, and can be wasted via fragmentation without careful allocation strategies. citeturn7search1turn5search1

A highly useful derived constant is KV bytes per token per sequence:

\[
m_{\text{KV/token}} \;=\; 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV}
\]

### Step three: incorporate paging / block allocation and prefix caching

**Paged KV allocation (vLLM / PagedAttention):** vLLM’s design goal is “near-zero waste in KV cache memory,” using block-level management inspired by virtual memory. citeturn7search1turn7search5  
In block-based KV systems, the reserved tokens per sequence become:

\[
S^{\text{reserved}}_i \;=\; \left\lceil \frac{S_i}{B_{\text{blk}}} \right\rceil \cdot B_{\text{blk}}
\]

So you plan against \(S^{\text{reserved}}_i\), not \(S_i\), if the runtime rounds to block boundaries.

**Prefix / prompt caching:** vLLM caches “kv-cache blocks of processed requests and reuses these blocks when a new request comes in with the same prefix.” citeturn5search1  
If \(S_{\text{shared}}\) tokens are identical across requests, and the system reuses the shared prefix KV once, the saved KV memory is approximately:

\[
\Delta M_{\text{KV}} \;\approx\; 2 \cdot (B-1)\cdot S_{\text{shared}} \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV}
\]

This is one of the few ways to reduce multi-user KV memory without reducing \(B\) or \(S\), and it is emphasized as widely used because it “won’t change model outputs.” citeturn5search1

### Step four: bound activation and temporary workspace memory

You cannot predict kernel workspaces perfectly pre-download, but you can bound the risk region:

- Long-context prefill can be expensive because “standard attention” is \(O(S^2)\) in time and memory (if it materializes the attention matrix). FlashAttention addresses this with IO-aware tiling and fused kernels, reducing HBM traffic and avoiding writing the full \(S\times S\) matrix to HBM in typical implementations. citeturn7search0turn7search4  

A practical pre-deployment bound for prefill activations uses a tunable constant \(\gamma\) (backend-dependent):

\[
M_{\text{act+tmp,prefill}} \;\approx\; \gamma \cdot B \cdot S_{\text{prefill}} \cdot d_{\text{model}} \cdot b_a
\]

where \(b_a\) is activation bytes (often 2 bytes under FP16/BF16 compute). The major point is not the exact constant, but that **prefill transient memory scales linearly with \(B\cdot S\cdot d_{\text{model}}\)** for memory-efficient attention kernels, rather than quadratically. citeturn7search0turn7search4

### Step five: include runtime overhead, allocator reservation, and fixed costs

For PyTorch/CUDA stacks, two documented facts matter for planning:

- PyTorch uses a **caching memory allocator**, so “unused memory managed by the allocator will still show as if used in `nvidia-smi`.” citeturn6search13  
- The caching allocator can round allocations to reduce fragmentation; this can add overhead, and PyTorch exposes stats (`memory_stats`) to diagnose it. citeturn6search33  

Additionally, Hugging Face Accelerate explicitly notes ~1–2 GiB consumed by CUDA kernels on first allocation. citeturn6search0

Therefore a robust *planning* envelope should be parameterized, not hard-coded:

\[
M_{\text{plan}} \;=\; M_{\text{fixed}} \;+\; \big(M_{\text{weights}} + M_{\text{KV}} + M_{\text{act+tmp}}\big)\cdot (1+\alpha)
\]

- \(M_{\text{fixed}}\): fixed startup/runtime pool cost (CUDA init; engine runtime pools; etc.)  
- \(\alpha\): safety headroom for fragmentation, allocator reservation, and burst allocations (heuristic; calibrate per runtime)

## Minimum, workable, recommended budgets and a matching decision tree

### A practical “minimum / workable / recommended” framework

The following is a **capacity planning convention** that makes the implicit tradeoffs explicit. Only the fixed-cost magnitude is directly documented; headroom percentages are engineering policy knobs justified by allocator behavior and memory pool reservation (and should be calibrated against your chosen runtime). citeturn6search0turn6search13turn6search33

| Budget level | Memory inequality (GPU case) | Meaning in practice | Common failure mode |
|---|---|---|---|
| Minimum | \(M_{\text{weights}} + M_{\text{KV}} + M_{\text{fixed}} \le M_{\text{usable}}\) | Might run if your workload is stable and kernels don’t need large workspaces | OOM from fragmentation / transient buffers / runtime preallocation (vLLM/TensorRT) citeturn5search0turn5search2turn6search13 |
| Workable | \(M_{\text{fixed}} + (M_{\text{weights}}+M_{\text{KV}})\cdot 1.10 \le M_{\text{usable}}\) | Stable for most interactive use; some room for allocator growth and moderate prefill spikes | Throughput collapse if you rely on CPU offload or memory pressure paging |
| Recommended | \(M_{\text{fixed}} + (M_{\text{weights}}+M_{\text{KV}})\cdot 1.25 \le M_{\text{usable}}\) | Stable for multi-turn chat and some concurrency, with fewer “mystery OOMs” | Capacity waste (over-provision) if your workload is always tiny |

### Runtime-aware decision tree for model ↔ hardware matching

This is a structured checklist rather than a literal diagram, but it functions as a deterministic decision tree.

| Decision point | Test | If “no” | If “yes” |
|---|---|---|---|
| Choose runtime pool policy | Does your runtime preallocate KV pools (vLLM, TensorRT-LLM)? citeturn5search0turn5search2 | Use tensor-based estimate + allocator headroom | Replace \(M_{\text{usable}}\) with the runtime’s “target fraction” budget (e.g., vLLM `gpu_memory_utilization`) citeturn5search0 |
| Weight fit | \(M_{\text{weights}} + M_{\text{fixed}} \le M_{\text{usable}}\) citeturn6search0 | Switch to lower bpw (INT8/INT4), multi-GPU, or CPU offload | Proceed |
| KV fit for your SLA | \(M_{\text{KV}}(T) \le M_{\text{usable}} - M_{\text{weights}} - M_{\text{fixed}}\) | Reduce \(S\), reduce \(B\), choose GQA/MQA/MLA model, or enable KV quantization | Proceed |
| Multi-user savings | Do many sessions share identical prefixes? | If no, prefix caching won’t help | If yes, prefix caching can reduce KV by \(\Delta M_{\text{KV}}\) and reduce prefill compute citeturn5search1 |
| Long-context transient risk | Are you guaranteed to have FlashAttention or equivalent? | Pre-fill may require larger temporary buffers | If yes, prefill memory is better behaved for long sequences citeturn7search0turn7search4 |
| Apple Silicon specialization | Are you on Apple unified memory (MLX/Metal)? | Use discrete GPU rules | Use `recommendedMaxWorkingSetSize` as “runs well” ceiling; unified allocation ≠ stable perf citeturn6search2turn6search3 |

### Separate estimation paths by deployment scenario

The mathematics is the same; what changes are \(B\), the distribution of \(S_i\), and whether “usable memory” is a single VRAM pool or a unified pool.

| Scenario | Primary objective | Set these knobs | Most likely bottleneck |
|---|---|---|---|
| Single-user chat | Low-latency interactive | \(B=1\); \(S\) = max chat context (e.g., 4k–8k) | Weights for FP16/BF16; KV only dominates at very long context |
| Long-context research | Maximize \(S\) (e.g., 128k) | \(B=1\); \(S\approx 128k\); consider KV quantization | KV cache dominates; GQA/MQA/MLA choice is decisive citeturn1search13turn7search1 |
| Multi-user serving | Maximize \(\sum S_i\) under SLA | Model \(\sum S_i\) distribution; apply prefix caching when applicable | KV cache and KV fragmentation; capacity depends on paging/prefix reuse citeturn7search1turn5search1 |
| CPU-offloaded setup | Fit larger weights than VRAM allows | Split layers/experts: GPU vs CPU; compute \(M_{\text{weights}}^{GPU}\), \(M_{\text{KV}}^{GPU}\) by layer placement | PCIe transfer + CPU compute dominate “runs well”; memory can fit but latency can explode |
| Apple Silicon | Stay below memory pressure | Use unified pool; treat `recommendedMaxWorkingSetSize` as performance limit | Memory pressure / swap + GPU working set ceiling citeturn6search2turn6search3 |

If you use Hugging Face Accelerate-style `device_map="auto"`, it “fills GPU(s) first, then CPU, and finally disk,” which is a memory-management mechanism, not a throughput-optimal parallelism strategy. citeturn7search14turn7search10

## Worked examples with computed tables

### Example model specs and KV-per-token constants

The table below derives \(m_{\text{KV/token}}\) at FP16/BF16 (\(b_{KV}=2\) bytes) directly from each model’s config. All computed numbers later use the exact KV formula and these constants. citeturn2view1turn2view2turn2view0turn8search0turn3search2

Assumptions for weight bpw in these examples:

- FP16/BF16 weights: \(b_w=16\).  
- “INT8 (w/scale)” planning example: \(b_w \approx 8.0625\) bpw (representative of 8-bit weights + small per-group scale overhead; exact overhead depends on format).  
- “INT4 (w/scale+zp)” uses the published group-quant overhead example \(4.1875\) bpw. citeturn3search6  

| Example model | \(L\) | \(d_{\text{model}}\) | \(H_q\) | \(H_{kv}\) | \(d_h\) | \(m_{\text{KV/token}}\) @ FP16 (KiB) | Native `max_position_embeddings` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3-8B | 32 | 4096 | 32 | 8 | 128 | 128.0 | 8192 |
| Qwen2.5-32B | 64 | 5120 | 40 | 8 | 128 | 256.0 | 131072 |
| Llama-3-70B | 80 | 8192 | 64 | 8 | 128 | 320.0 | 8192 |
| Mixtral-8x7B | 32 | 4096 | 32 | 8 | 128 | 128.0 | 32768 |
| LLaVA-1.5 (LLM+CLIP; KV uses Llama-2-7B base) | 32 | 4096 | 32 | 32 | 128 | 512.0 | 4096 |

Notes: Llama 3 uses GQA across models (reducing KV vs MHA), and was trained on 8192-token sequences; Llama 3.1 expands context length to 128K. citeturn1search0turn1search13

### Dense ~8B example: Llama-3-8B

All KV cache values below use \(b_{KV}=2\) bytes (FP16/BF16 KV). Architecture values are from config. citeturn2view1turn1search0

| Precision | \(S\) tokens | \(B\) | Weights (GiB) | KV cache (GiB) | Tensor only (GiB) | Dominant term |
|---|---:|---:|---:|---:|---:|---|
| FP16/BF16 | 4096 | 1 | 14.90 | 0.50 | 15.40 | Weights |
| FP16/BF16 | 8192 | 1 | 14.90 | 1.00 | 15.90 | Weights |
| FP16/BF16 | 131072 | 1 | 14.90 | 16.00 | 30.90 | KV cache |
| INT8 (w/scale) | 4096 | 1 | 7.51 | 0.50 | 8.01 | Weights |
| INT8 (w/scale) | 131072 | 1 | 7.51 | 16.00 | 23.51 | KV cache |
| INT4 (w/scale+zp) | 4096 | 1 | 3.90 | 0.50 | 4.40 | Weights |
| INT4 (w/scale+zp) | 131072 | 1 | 3.90 | 16.00 | 19.90 | KV cache |
| FP16/BF16 | 131072 | 8 | 14.90 | 128.00 | 142.90 | KV cache |

Planning overlays (example policy): CUDA fixed cost 1–2 GiB (documented), plus headroom factor (heuristic) because caching allocators reserve memory and fragmentation/rounding can create apparent “mystery OOM.” citeturn6search0turn6search13turn6search33

### Dense ~32B example: Qwen2.5-32B

Qwen2.5-32B supports `max_position_embeddings=131072` in config, so 128K context is in-family. citeturn2view0

| Precision | \(S\) tokens | \(B\) | Weights (GiB) | KV cache (GiB) | Tensor only (GiB) | Dominant term |
|---|---:|---:|---:|---:|---:|---|
| FP16/BF16 | 4096 | 1 | 59.60 | 1.00 | 60.60 | Weights |
| FP16/BF16 | 32768 | 1 | 59.60 | 8.00 | 67.60 | Weights |
| FP16/BF16 | 131072 | 1 | 59.60 | 32.00 | 91.60 | Weights |
| INT8 (w/scale) | 4096 | 1 | 30.00 | 1.00 | 31.00 | Weights |
| INT8 (w/scale) | 131072 | 1 | 30.00 | 32.00 | 62.00 | KV cache (slightly) |
| INT4 (w/scale+zp) | 4096 | 1 | 15.62 | 1.00 | 16.62 | Weights |
| INT4 (w/scale+zp) | 131072 | 1 | 15.62 | 32.00 | 47.62 | KV cache |
| INT4 (w/scale+zp) | 131072 | 8 | 15.62 | 256.00 | 271.62 | KV cache |

Operational implication: for 32B-class models, **weights dominate** at moderate contexts even with long \(S\), but KV dominates quickly under concurrency. This is exactly why serving systems focus on KV cache paging and reuse for higher batch sizes. citeturn7search1turn5search1

### Dense ~70B example: Llama-3-70B

Architecture values are from the published config. Llama 3 native max position is 8192; Llama 3.1 provides 128K context family variants. citeturn2view2turn1search0turn1search13

| Precision | \(S\) tokens | \(B\) | Weights (GiB) | KV cache (GiB) | Tensor only (GiB) | Dominant term |
|---|---:|---:|---:|---:|---:|---|
| FP16/BF16 | 4096 | 1 | 130.39 | 1.25 | 131.64 | Weights |
| FP16/BF16 | 8192 | 1 | 130.39 | 2.50 | 132.89 | Weights |
| FP16/BF16 | 131072 | 1 | 130.39 | 40.00 | 170.39 | Weights |
| INT8 (w/scale) | 4096 | 1 | 65.69 | 1.25 | 66.94 | Weights |
| INT4 (w/scale+zp) | 4096 | 1 | 34.10 | 1.25 | 35.35 | Weights |
| INT4 (w/scale+zp) | 131072 | 1 | 34.10 | 40.00 | 74.10 | KV cache |
| INT4 (w/scale+zp) | 131072 | 8 | 34.10 | 320.00 | 354.10 | KV cache |

Key insight: with 70B at INT4 weight-only, the breakpoint where KV cache dominates arrives at long context (e.g., 128K), making **KV precision / KV management strategy** a first-order choice for long-context deployments. This is the same bottleneck that motivates PagedAttention-style KV paging and sharing. citeturn7search1turn7search5

### MoE example: Mixtral-8x7B

Mixtral’s MoE property changes compute/activation behavior, but memory planning still depends on **total weights** unless experts are sharded across devices. Mistral states Mixtral has **46.7B total parameters** but uses **12.9B parameters per token** (top-2 routing); the config also states `num_experts_per_tok=2` and `num_local_experts=8`. citeturn0search2turn8search0

| Precision | \(S\) tokens | \(B\) | Weights (GiB) | KV cache (GiB) | Tensor only (GiB) | Dominant term |
|---|---:|---:|---:|---:|---:|---|
| FP16/BF16 | 4096 | 1 | 86.96 | 0.50 | 87.46 | Weights |
| FP16/BF16 | 32768 | 1 | 86.96 | 4.00 | 90.96 | Weights |
| INT8 (w/scale) | 32768 | 1 | 43.79 | 4.00 | 47.79 | Weights |
| INT4 (w/scale+zp) | 32768 | 1 | 22.72 | 4.00 | 26.72 | Weights |
| INT4 (w/scale+zp) | 131072 | 1 | 22.72 | 16.00 | 38.72 | Weights (slightly) |
| INT4 (w/scale+zp) | 131072 | 8 | 22.72 | 128.00 | 150.72 | KV cache |

Where MoE changes capacity planning: if you introduce **expert parallelism** or expert sharding, you can reduce per-device expert weight storage; but absent that, you budget against full weight residency. citeturn0search2turn8search0

### Multimodal example: LLaVA-1.5 class model

LLaVA-1.5 is composed of **CLIP ViT-L/14** (vision encoder), **Vicuna-v1.5** (LLM base), and an **MLP connector**; the vision encoder is typically kept frozen during tuning. citeturn4search17turn0search3

CLIP ViT-L/14@336 has been reported in the literature at ~**427.94M parameters** (order-of-magnitude: 0.4B params), which is significant but still smaller than the 7B LLM base; thus multimodal towers add a fixed weight cost that is often non-negligible for small GPUs. citeturn4search27

For KV sizing, what matters is the underlying LLM attention type. If the base is Llama-2-7B (MHA: \(H_{kv}=H_q=32\)), KV per token is 4× larger than a comparable GQA model with \(H_{kv}=8\). citeturn3search2

Assuming the same FP16/BF16 KV cache dtype (\(b_{KV}=2\)):

| Scenario | \(S\) total tokens | Precision | Weights (GiB) | KV cache (GiB) | Tensor only (GiB) | Bottleneck |
|---|---:|---|---:|---:|---:|---|
| Text-only conversation | 4096 | INT4 (w/scale+zp) | 3.62 | 2.00 | 5.62 | Weights (slightly) |
| “4k text + 1 image” (adds visual tokens) | 4672 | INT4 (w/scale+zp) | 3.62 | 2.28 | 5.90 | KV grows linearly with added tokens |
| Long-context multimodal (hypothetical) | 131072 | INT4 (w/scale+zp) | 3.62 | 64.00 | 67.62 | KV cache dominates |

Practical interpretation: multimodal adds memory in two ways—(1) extra tower weights, and (2) increased effective prompt length (visual tokens), which directly inflates KV cache and prefill transient memory. citeturn4search17turn7search0

## Final cheat sheet: formulas, heuristics, and common estimation pitfalls

### Core formulas you should memorize

| Quantity | Formula | Notes |
|---|---|---|
| Weight memory | \(M_{\text{weights}} = P\cdot \frac{b_w}{8}\) | Use effective bpw if quantized. citeturn3search6 |
| Effective bpw (group quant) | \(b_w = b_q + \frac{b_{\text{scale}}+b_{\text{zp}}}{g}\) | Published example yields 0.1875 bpw overhead for \(g=128\). citeturn3search6 |
| KV cache (total) | \(M_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV}\cdot \sum_{i=1}^B S_i\) | This is the dominant long-context + concurrency term. citeturn7search1 |
| KV per token | \(m_{\text{KV/token}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{KV}\) | Multiply by total cached tokens across sessions. |
| Prefix caching savings | \(\Delta M_{\text{KV}} \approx 2(B-1)S_{\text{shared}}L H_{kv} d_h b_{KV}\) | Depends on identical prefix reuse. citeturn5search1 |
| Working planning envelope | \(M_{\text{plan}} = M_{\text{fixed}} + (M_{\text{weights}}+M_{\text{KV}}+M_{\text{act+tmp}})(1+\alpha)\) | \(M_{\text{fixed}}\) is documented (CUDA init 1–2 GiB); \(\alpha\) is calibration. citeturn6search0turn6search13 |

### Runtime-specific “gotchas” that break naive estimates

| Gotcha | Why it happens | What to do |
|---|---|---|
| “nvidia-smi shows used VRAM even after freeing” | PyTorch uses a caching allocator; reserved blocks remain “used” from the driver’s view. citeturn6search13 | Monitor `memory_allocated` vs `memory_reserved`; keep headroom. citeturn6search13turn6search33 |
| Immediate OOM on startup in vLLM | vLLM pre-allocates GPU cache per `gpu_memory_utilization`. citeturn5search0 | Lower `gpu_memory_utilization`, reduce `max_num_seqs` / `max_num_batched_tokens`, or reduce \(S_{\max}\). citeturn5search0 |
| TensorRT-LLM “eats the GPU” | Default KV cache can allocate 90% of remaining free VRAM. citeturn5search2 | Set `KVCacheConfig.maxTokens` or `freeGpuMemoryFraction` explicitly. citeturn5search2 |
| Long-context prefill behaves erratically | Standard attention is quadratic in \(S\); memory-efficient kernels matter. citeturn7search0turn7search4 | Require FlashAttention-class kernels for long-context regimes. citeturn7search0turn7search4 |
| Apple Silicon “can allocate” but performance collapses | Unified memory + working set performance limits; Metal exposes `recommendedMaxWorkingSetSize`. citeturn6search2turn6search3 | Budget against `recommendedMaxWorkingSetSize` for “runs well,” not total unified memory. citeturn6search2 |

### The most common estimation mistakes

| Mistake | Why it’s wrong | Correct practice |
|---|---|---|
| Using “parameter count × bytes” as the full estimate | Ignores KV cache and runtime allocation policies; KV grows with \(S\) and \(B\). citeturn7search1turn5search0 | Always compute \(M_{\text{KV}}\) explicitly and include fixed/runtime overhead. citeturn6search0turn6search13 |
| Forgetting GQA/MQA differences | KV scales with \(H_{kv}\), not \(H_q\); GQA reduces KV. citeturn2view1turn1search0 | Read `num_key_value_heads` from config and compute \(m_{\text{KV/token}}\). citeturn2view1 |
| Assuming “INT4 model” is exactly 0.5 bytes/param | Quant metadata adds bpw overhead; group size matters. citeturn3search6 | Use effective bpw and treat it as a range unless format is fixed. citeturn3search6 |
| Not modeling concurrency | Serving \(B\) users multiplies KV by \(\sum S_i\). citeturn7search1 | Capacity plan on total cached tokens across live sessions. |
| Ignoring runtime preallocation behaviors | vLLM / TensorRT-LLM may reserve memory ahead of time. citeturn5search0turn5search2 | Incorporate runtime policy as a hard constraint in \(M_{\text{usable}}\). citeturn5search0turn5search2 |

