export type ModelTypeBadge = 'vision' | 'tools' | 'thinking' | 'audio';

export interface ModelVariant {
  id: string;
  familyId: string;
  sizeLabel: string;
  parameterCount: number; // total params in billions
  layers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKVHeads: number;
  headDim: number;
  maxContext: number;
  typeBadges: ModelTypeBadge[];
  intelligenceScore?: number;
  isMoE: boolean;
  activeParameterCount?: number; // billions, for MoE models
}

export interface ModelFamily {
  id: string;
  name: string;
  variants: ModelVariant[];
}

export interface GpuProfile {
  id: string;
  name: string;
  vendor: 'NVIDIA';
  vramGB: number;
  tier: 'consumer' | 'professional' | 'datacenter';
}

export interface EstimationResult {
  weightMemoryGB: number;
  kvCacheMemoryGB: number;
  runtimeOverheadGB: number;
  totalEstimatedGB: number;
}

export interface CompatibilityResult {
  estimatedVramGB: number;
  availableVramGB: number;
  verdict: 'yes' | 'maybe' | 'no';
}

// --- V2: Machine catalog ---

export type Locale = 'ja' | 'en' | 'zh';

export interface MachineGpu {
  gpuId: string;
  count: number;
}

export interface Machine {
  id: string;
  slug: string;
  name: Record<Locale, string>;
  tagline: Record<Locale, string>;
  description: Record<Locale, string>;
  gpus: MachineGpu[];
  cpu: string;
  ramGB: number;
  storage: string;
  priceDisplay: Record<Locale, string>;
  images: string[];
  featured: boolean;
  available: boolean;
  specs: Record<string, string>;
}

export interface MachineCompatibility {
  variant: ModelVariant;
  totalVramAvailableGB: number;
  estimatedVramGB: number;
  perGpuUsageGB: number;
  perGpuAvailableGB: number;
  gpuCount: number;
  verdict: 'yes' | 'maybe' | 'no';
}
