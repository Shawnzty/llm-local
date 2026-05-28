import Link from 'next/link';
import type { Machine } from '@llm-local/shared';

interface MachineCardProps {
  machine: Machine;
  locale: string;
}

export function MachineCard({ machine, locale }: MachineCardProps) {
  const loc = locale as 'ja' | 'en' | 'zh';
  return (
    <Link href={`/${locale}/machines/${machine.slug}`} className="group block">
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 shadow-sm hover:shadow-md hover:border-gray-300 dark:hover:border-gray-700 transition-all duration-200 h-full flex flex-col">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
            {machine.name[loc]}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
            {machine.tagline[loc]}
          </p>
          <div className="text-xs text-gray-400 dark:text-gray-500 space-y-1">
            <div>{machine.gpus.map((g) => `${g.gpuId.replace(/-/g, ' ').toUpperCase()} ×${g.count}`).join(', ')}</div>
            <div>{machine.cpu} / {machine.ramGB}GB RAM</div>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between">
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {machine.priceDisplay[loc]}
          </span>
          {machine.available ? (
            <span className="text-xs text-green-600 dark:text-green-400">●</span>
          ) : (
            <span className="text-xs text-gray-400">—</span>
          )}
        </div>
      </div>
    </Link>
  );
}
