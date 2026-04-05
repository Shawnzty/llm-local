import { GpuProfile } from '../types';

export const GPU_PROFILES: GpuProfile[] = [
  // Consumer — GeForce RTX 30 series
  { id: 'rtx-3060-12gb', name: 'RTX 3060 12GB', vendor: 'NVIDIA', vramGB: 12, tier: 'consumer' },
  { id: 'rtx-3070-8gb', name: 'RTX 3070 8GB', vendor: 'NVIDIA', vramGB: 8, tier: 'consumer' },
  { id: 'rtx-3080-10gb', name: 'RTX 3080 10GB', vendor: 'NVIDIA', vramGB: 10, tier: 'consumer' },
  { id: 'rtx-3080ti-12gb', name: 'RTX 3080 Ti 12GB', vendor: 'NVIDIA', vramGB: 12, tier: 'consumer' },
  { id: 'rtx-3090-24gb', name: 'RTX 3090 24GB', vendor: 'NVIDIA', vramGB: 24, tier: 'consumer' },

  // Consumer — GeForce RTX 40 series
  { id: 'rtx-4060-8gb', name: 'RTX 4060 8GB', vendor: 'NVIDIA', vramGB: 8, tier: 'consumer' },
  { id: 'rtx-4060ti-16gb', name: 'RTX 4060 Ti 16GB', vendor: 'NVIDIA', vramGB: 16, tier: 'consumer' },
  { id: 'rtx-4070-12gb', name: 'RTX 4070 12GB', vendor: 'NVIDIA', vramGB: 12, tier: 'consumer' },
  { id: 'rtx-4070ti-12gb', name: 'RTX 4070 Ti 12GB', vendor: 'NVIDIA', vramGB: 12, tier: 'consumer' },
  { id: 'rtx-4080-16gb', name: 'RTX 4080 16GB', vendor: 'NVIDIA', vramGB: 16, tier: 'consumer' },
  { id: 'rtx-4090-24gb', name: 'RTX 4090 24GB', vendor: 'NVIDIA', vramGB: 24, tier: 'consumer' },

  // Consumer — GeForce RTX 50 series
  { id: 'rtx-5070ti-16gb', name: 'RTX 5070 Ti 16GB', vendor: 'NVIDIA', vramGB: 16, tier: 'consumer' },
  { id: 'rtx-5080-16gb', name: 'RTX 5080 16GB', vendor: 'NVIDIA', vramGB: 16, tier: 'consumer' },
  { id: 'rtx-5090-32gb', name: 'RTX 5090 32GB', vendor: 'NVIDIA', vramGB: 32, tier: 'consumer' },

  // Professional
  { id: 'rtx-a6000-48gb', name: 'RTX A6000 48GB', vendor: 'NVIDIA', vramGB: 48, tier: 'professional' },

  // Datacenter
  { id: 'a100-40gb', name: 'A100 40GB', vendor: 'NVIDIA', vramGB: 40, tier: 'datacenter' },
  { id: 'a100-80gb', name: 'A100 80GB', vendor: 'NVIDIA', vramGB: 80, tier: 'datacenter' },
  { id: 'h100-80gb', name: 'H100 80GB', vendor: 'NVIDIA', vramGB: 80, tier: 'datacenter' },
  { id: 'l40s-48gb', name: 'L40S 48GB', vendor: 'NVIDIA', vramGB: 48, tier: 'datacenter' },
];
