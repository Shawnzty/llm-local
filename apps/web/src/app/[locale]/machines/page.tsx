import { getTranslations, setRequestLocale } from 'next-intl/server';
import { MACHINES } from '@llm-local/shared';
import { MachineCard } from '@/components/MachineCard';

export default async function MachinesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations('machines');

  return (
    <div className="mx-auto max-w-5xl px-6 py-12 sm:py-16">
      <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-8">
        {t('title')}
      </h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {MACHINES.filter((m) => m.available).map((machine) => (
          <MachineCard key={machine.id} machine={machine} locale={locale} />
        ))}
      </div>
    </div>
  );
}
