/**
 * VRAM estimation constants derived from the deep-research report.
 *
 * Default assumption for V1: 4-bit weight-only quantization,
 * 8K context, single-user, NVIDIA GPU, FP16 KV cache.
 */

// --- Weight quantization ---
// Group-wise INT4 with FP16 scale + INT8 zero-point, group size 128
// Effective bpw = 4 + (16 + 8) / 128 = 4.1875
export const QUANT_BITS = 4;
export const GROUP_SIZE = 128;
export const SCALE_BITS = 16;
export const ZERO_POINT_BITS = 8;
export const EFFECTIVE_BPW = QUANT_BITS + (SCALE_BITS + ZERO_POINT_BITS) / GROUP_SIZE; // 4.1875

// --- KV cache ---
export const KV_BYTES_PER_ELEMENT = 2; // FP16

// --- Workload defaults ---
export const DEFAULT_CONTEXT_TOKENS = 8192;
export const DEFAULT_BATCH_SIZE = 1; // single-user

// --- Runtime overhead ---
// ~1.5 GiB for CUDA initialization + allocator reservation
export const FIXED_OVERHEAD_GB = 1.5;
// 10% headroom for fragmentation and allocator behavior
export const HEADROOM_FACTOR = 0.10;

// --- Compatibility thresholds ---
// yes:   estimated <= 80% of available (20%+ headroom)
// maybe: estimated <= 100% of available (tight)
// no:    estimated > 100% of available
export const COMPAT_YES_THRESHOLD = 0.80;
export const COMPAT_MAYBE_THRESHOLD = 1.00;

// --- Unit conversion ---
export const BYTES_PER_GIB = 1024 ** 3;
