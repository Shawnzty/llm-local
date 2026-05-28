import { getTranslations, setRequestLocale } from 'next-intl/server';
import Link from 'next/link';
import { MACHINES } from '@llm-local/shared';
import { MachineCard } from '@/components/MachineCard';

export default async function HomePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations();
  const featuredMachines = MACHINES.filter((m) => m.featured);

  return (
    <div>
      {/* Hero */}
      <section className="px-6 py-20 sm:py-28">
        <div className="mx-auto max-w-3xl text-center">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            {t('hero.headline')}
          </h1>
          <p className="mt-4 text-lg text-gray-500 dark:text-gray-400 max-w-xl mx-auto">
            {t('hero.subheadline')}
          </p>
          <div className="mt-8">
            <Link
              href={`/${locale}/machines`}
              className="inline-flex items-center px-6 py-3 rounded-xl bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:opacity-90 transition-opacity"
            >
              {t('hero.cta')}
            </Link>
          </div>
        </div>
      </section>

      {/* Featured machines */}
      <section className="px-6 pb-16">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
            {t('machines.featured')}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {featuredMachines.map((machine) => (
              <MachineCard key={machine.id} machine={machine} locale={locale} />
            ))}
          </div>
        </div>
      </section>

      {/* Why local LLM */}
      <section className="px-6 py-16 border-t border-gray-200 dark:border-gray-800">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-8 text-center">
            {t('whyLocal.title')}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {(['privacy', 'cost', 'latency', 'control'] as const).map((key) => (
              <div
                key={key}
                className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6"
              >
                <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                  {t(`whyLocal.${key}.title`)}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {t(`whyLocal.${key}.description`)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="px-6 py-16 text-center">
        <Link
          href={`/${locale}/contact`}
          className="inline-flex items-center px-6 py-3 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-medium text-gray-900 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {t('nav.contact')}
        </Link>
      </section>
    </div>
  );
}
