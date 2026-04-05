'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { ModelSelector } from '@/components/ModelSelector';
import { ResultCard } from '@/components/ResultCard';
import { Badge } from '@/components/Badge';
import { getVariantById, formatGB, formatContext } from '@/lib/utils';
import { estimateVram } from '@/lib/estimation/engine';

export default function EstimatePage() {
  const [familyId, setFamilyId] = useState('');
  const [variantId, setVariantId] = useState('');

  const variant = useMemo(
    () => (variantId ? getVariantById(variantId) : undefined),
    [variantId],
  );

  const estimation = useMemo(
    () => (variant ? estimateVram(variant) : undefined),
    [variant],
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
          How much VRAM do I need?
        </h1>
      </div>

      <div className="space-y-8">
        <ModelSelector
          selectedFamilyId={familyId}
          selectedVariantId={variantId}
          onFamilyChange={setFamilyId}
          onVariantChange={setVariantId}
        />

        {variant && estimation && (
          <div className="space-y-5 animate-in fade-in duration-200">
            {variant.typeBadges.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {variant.typeBadges.map((badge) => (
                  <Badge key={badge} type={badge} />
                ))}
              </div>
            )}

            <ResultCard
              rows={[
                {
                  label: 'Max context',
                  value: formatContext(variant.maxContext),
                },
                ...(variant.intelligenceScore != null
                  ? [
                      {
                        label: 'Intelligence score',
                        sublabel: '(external)',
                        value: `${variant.intelligenceScore}`,
                      },
                    ]
                  : []),
                {
                  label: 'Estimated VRAM',
                  value: formatGB(estimation.totalEstimatedGB),
                  prominent: true,
                },
              ]}
            />

            <p className="text-xs text-gray-400 dark:text-gray-500">
              Based on 4-bit quantization, 8K context, single-user, NVIDIA GPU.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
