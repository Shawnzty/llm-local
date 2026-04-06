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
