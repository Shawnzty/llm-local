import Link from 'next/link';
import type { Locale, Machine } from '@llm-local/shared';
import { LOCALE_CURRENCY } from '@llm-local/shared';

interface MachineCardProps {
  machine: Machine;
  locale: string;
}

export function MachineCard({ machine, locale }: MachineCardProps) {
  const loc = locale as Locale;
  const currency = LOCALE_CURRENCY[loc];

  return (
    <Link href={`/${locale}/machines/${machine.slug}`} className="group block">
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 shadow-sm hover:shadow-md hover:border-gray-300 dark:hover:border-gray-700 transition-all duration-200 h-full flex flex-col">
        <div className="flex-1">
          <div className="flex items-baseline justify-between gap-2 mb-1">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">{machine.name}</h3>
            <span className="shrink-0 text-xs font-medium text-blue-600 dark:text-blue-400 tabular-nums">
              {machine.vramGB}GB VRAM
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{machine.tagline[loc]}</p>
          <div className="flex flex-wrap gap-1.5 mb-4">
            {machine.badges[loc].map((b) => (
              <span
                key={b}
                className="inline-block px-2 py-0.5 rounded-md bg-gray-100 dark:bg-gray-800 text-xs text-gray-600 dark:text-gray-300"
              >
                {b}
              </span>
            ))}
          </div>
          <div className="text-xs text-gray-400 dark:text-gray-500 space-y-1">
            <div>{machine.gpu.model.replace('NVIDIA ', '')} · {machine.gpu.configLabel}</div>
            <div>{machine.cpu.model} / {machine.ram.sizeGB}GB RAM</div>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between">
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 tabular-nums">
            {machine.priceLabels[currency]}
          </span>
          <span className="text-xs text-gray-400 dark:text-gray-500 group-hover:text-gray-600 dark:group-hover:text-gray-300 transition-colors">
            →
          </span>
        </div>
      </div>
    </Link>
  );
}
