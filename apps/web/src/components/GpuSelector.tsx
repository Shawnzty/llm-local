'use client';

import type { GpuProfile } from '@llm-local/shared';

interface GpuSelectorProps {
  gpus: GpuProfile[];
  selectedGpuId: string;
  onGpuChange: (gpuId: string) => void;
}

export function GpuSelector({ gpus, selectedGpuId, onGpuChange }: GpuSelectorProps) {
  const consumerGpus = gpus.filter((g) => g.tier === 'consumer');
  const professionalGpus = gpus.filter((g) => g.tier === 'professional');
  const datacenterGpus = gpus.filter((g) => g.tier === 'datacenter');

  return (
    <div>
      <label
        htmlFor="gpu-select"
        className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider"
      >
        GPU
      </label>
      <select
        id="gpu-select"
        value={selectedGpuId}
        onChange={(e) => onGpuChange(e.target.value)}
        className="w-full h-11 px-4 rounded-xl border border-gray-200 dark:border-gray-700
          bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100
          text-[15px] appearance-none cursor-pointer
          focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500
          transition-all duration-150"
      >
        <option value="">Select GPU</option>
        <optgroup label="Consumer">
          {consumerGpus.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </optgroup>
        {professionalGpus.length > 0 && (
          <optgroup label="Professional">
            {professionalGpus.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}
              </option>
            ))}
          </optgroup>
        )}
        <optgroup label="Datacenter">
          {datacenterGpus.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </optgroup>
      </select>
    </div>
  );
}
