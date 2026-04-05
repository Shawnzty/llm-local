'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { ModelSelector } from '@/components/ModelSelector';
import { GpuSelector } from '@/components/GpuSelector';
import { ResultCard } from '@/components/ResultCard';
import { CompatibilityBadge } from '@/components/CompatibilityBadge';
import { getVariantById, getGpuById, formatGB } from '@/lib/utils';
import { checkCompatibility } from '@/lib/estimation/engine';

export default function CompatibilityPage() {
  const [familyId, setFamilyId] = useState('');
  const [variantId, setVariantId] = useState('');
  const [gpuId, setGpuId] = useState('');

  const variant = useMemo(
    () => (variantId ? getVariantById(variantId) : undefined),
    [variantId],
  );

  const gpu = useMemo(
    () => (gpuId ? getGpuById(gpuId) : undefined),
    [gpuId],
  );

  const result = useMemo(
    () => (variant && gpu ? checkCompatibility(variant, gpu) : undefined),
    [variant, gpu],
  );

  return (
    <div className="mx-auto max-w-2xl px-6 py-12 sm:py-16">
      <div className="mb-10">
        <Link
          href="/"
          className="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          &larr; Back
        </Link>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mt-4">
          Can this model run on my GPU?
        </h1>
      </div>

      <div className="space-y-6">
        <ModelSelector
          selectedFamilyId={familyId}
          selectedVariantId={variantId}
          onFamilyChange={setFamilyId}
          onVariantChange={setVariantId}
        />

        <GpuSelector selectedGpuId={gpuId} onGpuChange={setGpuId} />

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
    </div>
  );
}
