import { getTranslations, setRequestLocale } from 'next-intl/server';
import { MACHINES, TIERS, TIER_ORDER } from '@llm-local/shared';
import type { Locale } from '@llm-local/shared';
import { MachineCard } from '@/components/MachineCard';

export default async function MachinesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const loc = locale as Locale;
  const t = await getTranslations('machines');

  const active = MACHINES.filter((m) => m.status === 'active').sort((a, b) => a.rank - b.rank);

  return (
    <div className="mx-auto max-w-5xl px-6 py-12 sm:py-16">
      <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{t('title')}</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 mb-10">{t('subtitle')}</p>

      {TIER_ORDER.map((tier) => {
        const machines = active.filter((m) => m.tier === tier);
        if (machines.length === 0) return null;
        return (
          <section key={tier} className="mb-12 last:mb-0">
            <div className="flex items-baseline gap-3 mb-5">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {TIERS[tier].label[loc]}
              </h2>
              <span className="text-xs text-gray-400 dark:text-gray-500">{TIERS[tier].vram} VRAM</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
              {machines.map((machine) => (
                <MachineCard key={machine.id} machine={machine} locale={locale} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
