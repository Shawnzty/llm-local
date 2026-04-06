'use client';

import { useState, useMemo } from 'react';
import { ModelSelector } from '@/components/ModelSelector';
import { ResultCard } from '@/components/ResultCard';
import { Badge } from '@/components/Badge';
import { formatGB, formatContext } from '@/lib/utils';
import { estimateVram, type ModelFamily } from '@llm-local/shared';

interface EstimateClientProps {
  families: ModelFamily[];
}

export function EstimateClient({ families }: EstimateClientProps) {
  const [familyId, setFamilyId] = useState('');
  const [variantId, setVariantId] = useState('');

  const variant = useMemo(() => {
    if (!variantId) return undefined;
    for (const f of families) {
      const v = f.variants.find((x) => x.id === variantId);
      if (v) return v;
    }
    return undefined;
  }, [variantId, families]);

  const estimation = useMemo(
    () => (variant ? estimateVram(variant) : undefined),
    [variant],
  );

  return (
    <div className="space-y-8">
      <ModelSelector
        families={families}
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
  );
}
