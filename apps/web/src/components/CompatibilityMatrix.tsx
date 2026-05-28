'use client';

import { useTranslations } from 'next-intl';
import type { MachineCompatibility } from '@llm-local/shared';
import { formatGB, formatContext } from '@/lib/utils';

interface CompatibilityMatrixProps {
  matrix: MachineCompatibility[];
  gpuCount: number;
}

const verdictColors = {
  yes: 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/30',
  maybe: 'text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30',
  no: 'text-red-500 dark:text-red-400 bg-red-50 dark:bg-red-950/30',
};

export function CompatibilityMatrix({ matrix, gpuCount }: CompatibilityMatrixProps) {
  const t = useTranslations('machines');

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-800">
            <th className="text-left py-3 pr-4 font-medium text-gray-500 dark:text-gray-400">{t('compatibleModels')}</th>
            <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">{t('estimatedVram')}</th>
            <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">{t('maxContext')}</th>
            <th className="text-center py-3 pl-4 font-medium text-gray-500 dark:text-gray-400">Status</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row) => (
            <tr key={row.variant.id} className="border-b border-gray-100 dark:border-gray-800/50">
              <td className="py-2.5 pr-4 text-gray-900 dark:text-gray-100">
                {row.variant.sizeLabel}
                <span className="ml-1.5 text-xs text-gray-400">
                  ({row.variant.parameterCount}B)
                </span>
              </td>
              <td className="py-2.5 px-4 text-right text-gray-600 dark:text-gray-300 tabular-nums">
                {gpuCount > 1
                  ? `${formatGB(row.perGpuUsageGB)} ×${gpuCount}`
                  : formatGB(row.estimatedVramGB)}
              </td>
              <td className="py-2.5 px-4 text-right text-gray-600 dark:text-gray-300 tabular-nums">
                {formatContext(row.variant.maxContext)}
              </td>
              <td className="py-2.5 pl-4 text-center">
                <span className={`inline-block px-2 py-0.5 rounded-md text-xs font-medium ${verdictColors[row.verdict]}`}>
                  {t(`verdict.${row.verdict}`)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
