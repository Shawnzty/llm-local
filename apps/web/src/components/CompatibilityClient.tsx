'use client';

import { useState, useMemo } from 'react';
import { ModelSelector } from '@/components/ModelSelector';
import { GpuSelector } from '@/components/GpuSelector';
import { ResultCard } from '@/components/ResultCard';
import { CompatibilityBadge } from '@/components/CompatibilityBadge';
import { formatGB } from '@/lib/utils';
import {
  checkCompatibility,
  type ModelFamily,
  type GpuProfile,
} from '@llm-local/shared';

interface CompatibilityClientProps {
  families: ModelFamily[];
  gpus: GpuProfile[];
}

export function CompatibilityClient({ families, gpus }: CompatibilityClientProps) {
  const [familyId, setFamilyId] = useState('');
  const [variantId, setVariantId] = useState('');
  const [gpuId, setGpuId] = useState('');

  const variant = useMemo(() => {
    if (!variantId) return undefined;
    for (const f of families) {
      const v = f.variants.find((x) => x.id === variantId);
      if (v) return v;
    }
    return undefined;
  }, [variantId, families]);

  const gpu = useMemo(
    () => (gpuId ? gpus.find((g) => g.id === gpuId) : undefined),
    [gpuId, gpus],
  );

  const result = useMemo(
    () => (variant && gpu ? checkCompatibility(variant, gpu) : undefined),
    [variant, gpu],
  );

  return (
    <div className="space-y-6">
      <ModelSelector
        families={families}
        selectedFamilyId={familyId}
        selectedVariantId={variantId}
        onFamilyChange={setFamilyId}
        onVariantChange={setVariantId}
      />

      <GpuSelector gpus={gpus} selectedGpuId={gpuId} onGpuChange={setGpuId} />

      {result && (
        <div className="space-y-5 animate-in fade-in duration-200">
          <ResultCard
            rows={[
              {
                label: 'Estimated VRAM need',
                value: formatGB(result.estimatedVramGB),
              },
              {
                label: 'Available VRAM',
                value: formatGB(result.availableVramGB),
              },
            ]}
          />

          <CompatibilityBadge verdict={result.verdict} />

          <p className="text-xs text-gray-400 dark:text-gray-500">
            Based on 4-bit quantization, 8K context, single-user, NVIDIA GPU.
          </p>
        </div>
      )}
    </div>
  );
}
