'use client';

import type { ModelFamily, ModelVariant } from '@llm-local/shared';

interface ModelSelectorProps {
  families: ModelFamily[];
  selectedFamilyId: string;
  selectedVariantId: string;
  onFamilyChange: (familyId: string) => void;
  onVariantChange: (variantId: string) => void;
}

export function ModelSelector({
  families,
  selectedFamilyId,
  selectedVariantId,
  onFamilyChange,
  onVariantChange,
}: ModelSelectorProps) {
  const variants = selectedFamilyId
    ? families.find((f) => f.id === selectedFamilyId)?.variants ?? []
    : [];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <label
          htmlFor="model-family"
          className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider"
        >
          Model
        </label>
        <select
          id="model-family"
          value={selectedFamilyId}
          onChange={(e) => {
            onFamilyChange(e.target.value);
            onVariantChange('');
          }}
          className="w-full h-11 px-4 rounded-xl border border-gray-200 dark:border-gray-700
            bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100
            text-[15px] appearance-none cursor-pointer
            focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500
            transition-all duration-150"
        >
          <option value="">Select model family</option>
          {families.map((f) => (
            <option key={f.id} value={f.id}>
              {f.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label
          htmlFor="model-size"
          className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider"
        >
          Size
        </label>
        <select
          id="model-size"
          value={selectedVariantId}
          onChange={(e) => onVariantChange(e.target.value)}
          disabled={!selectedFamilyId}
          className="w-full h-11 px-4 rounded-xl border border-gray-200 dark:border-gray-700
            bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100
            text-[15px] appearance-none cursor-pointer
            disabled:opacity-40 disabled:cursor-not-allowed
            focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500
            transition-all duration-150"
        >
          <option value="">
            {selectedFamilyId ? 'Select size' : 'Select model first'}
          </option>
          {variants.map((v: ModelVariant) => (
            <option key={v.id} value={v.id}>
              {v.sizeLabel}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
