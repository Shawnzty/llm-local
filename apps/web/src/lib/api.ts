import {
  MODEL_FAMILIES,
  GPU_PROFILES,
  type ModelFamily,
  type GpuProfile,
} from '@llm-local/shared';

const API_URL = process.env.NEXT_PUBLIC_API_URL;
const REVALIDATE_SECONDS = 3600;

export async function fetchFamilies(): Promise<ModelFamily[]> {
  if (!API_URL) return MODEL_FAMILIES;
  try {
    const res = await fetch(`${API_URL}/models`, {
      next: { revalidate: REVALIDATE_SECONDS },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as ModelFamily[];
  } catch (err) {
    console.warn('[web] fetchFamilies failed, using bundled seed:', err);
    return MODEL_FAMILIES;
  }
}

export async function fetchGpus(): Promise<GpuProfile[]> {
  if (!API_URL) return GPU_PROFILES;
  try {
    const res = await fetch(`${API_URL}/gpus`, {
      next: { revalidate: REVALIDATE_SECONDS },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as GpuProfile[];
  } catch (err) {
    console.warn('[web] fetchGpus failed, using bundled seed:', err);
    return GPU_PROFILES;
  }
}
