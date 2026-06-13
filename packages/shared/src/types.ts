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

// --- V2: Machine catalog (sales) ---

export type Locale = 'ja' | 'en';
export type Currency = 'JPY' | 'USD';
export type Tier = 'entry' | 'mid' | 'flagship';

export type LocalizedText = Record<Locale, string>;
export type LocalizedList = Record<Locale, string[]>;

export interface MachineGpu {
  model: string; // e.g. "NVIDIA Tesla V100 PCIe"
  memoryType: string; // e.g. "HBM2"
  count: number;
  configLabel: string; // e.g. "32GB×1 + 16GB×2"
  vramTotalGB: number;
}

export interface MachineCpu {
  model: string;
  cores: number;
  threads: number;
}

export interface MachineRam {
  sizeGB: number;
  type: string;
}

export interface MachineStorage {
  label: string;
  type: string;
}

export interface MachinePsu {
  wattage: number;
  rating: string;
  pseCertified: boolean;
  note?: LocalizedText;
}

export interface MachinePerformance {
  model: string; // e.g. "70B"
  quant: string; // e.g. "Q4"
  tokensPerSec: number | null; // null = not yet measured
  note?: LocalizedText;
}

export interface Machine {
  id: string;
  rank: number;
  slug: string;
  tier: Tier;
  featured: boolean;
  status: 'active' | 'inactive';
  name: string; // short model code, not translated (e.g. "Tadzuna 64GB Plus")
  category: LocalizedText;
  tagline: LocalizedText;
  prices: Record<Currency, number>;
  priceLabels: Record<Currency, string>;
  vramGB: number;
  gpu: MachineGpu;
  cpu: MachineCpu;
  motherboard: LocalizedText;
  ram: MachineRam;
  storage: MachineStorage;
  psu: MachinePsu;
  chassis: LocalizedText;
  cooling: LocalizedText;
  performance: MachinePerformance[];
  canRun: LocalizedList;
  bestFor: LocalizedList;
  valueProps: LocalizedList;
  badges: LocalizedList;
  warranty: LocalizedText;
  disclaimers: LocalizedText[];
}

export interface MarketAnchor {
  name: string;
  vramGB: number;
  approxJPY: number;
  note: LocalizedText;
}

export interface TierInfo {
  label: LocalizedText;
  vram: string;
}
