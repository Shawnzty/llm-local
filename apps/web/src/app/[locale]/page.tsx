import { getTranslations, setRequestLocale } from 'next-intl/server';
import Link from 'next/link';
import { MACHINES, MARKET_ANCHORS, LOCALE_CURRENCY } from '@tadzuna/shared';
import type { Locale } from '@tadzuna/shared';
import { MachineCard } from '@/components/MachineCard';

function formatJPY(n: number, locale: Locale): string {
  if (locale === 'en') {
    const usd = Math.round((n * 0.006452) / 10) * 10;
    return `~$${usd.toLocaleString('en-US')}`;
  }
  return `¥${n.toLocaleString('ja-JP')}`;
}

export default async function HomePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const loc = locale as Locale;
  const t = await getTranslations();
  const featuredMachines = MACHINES.filter((m) => m.featured).sort((a, b) => a.rank - b.rank);

  return (
    <div>
      {/* Hero */}
      <section className="px-6 py-20 sm:py-28">
        <div className="mx-auto max-w-3xl text-center">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            {t('hero.headline')}
          </h1>
          <p className="mt-4 text-lg text-gray-500 dark:text-gray-400 max-w-2xl mx-auto">
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
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {featuredMachines.map((machine) => (
              <MachineCard key={machine.id} machine={machine} locale={locale} />
            ))}
          </div>
        </div>
      </section>

      {/* Comparison table */}
      <section className="px-6 py-16 border-t border-gray-200 dark:border-gray-800">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1 text-center">
            {t('comparison.title')}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-8 text-center">
            {t('comparison.subtitle')}
          </p>
          <div className="overflow-x-auto rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400">
                  <th className="text-left font-medium py-3 px-4">{t('comparison.product')}</th>
                  <th className="text-right font-medium py-3 px-4">{t('comparison.vram')}</th>
                  <th className="text-right font-medium py-3 px-4">{t('comparison.approxPrice')}</th>
                  <th className="text-left font-medium py-3 px-4 hidden sm:table-cell">{t('comparison.note')}</th>
                </tr>
              </thead>
              <tbody>
                {/* Our lineup highlighted */}
                <tr className="border-b border-gray-100 dark:border-gray-800/50 bg-blue-50/50 dark:bg-blue-950/20">
                  <td className="py-3 px-4 font-semibold text-gray-900 dark:text-gray-100">
                    {t('comparison.ourLineup')}
                  </td>
                  <td className="py-3 px-4 text-right tabular-nums text-gray-900 dark:text-gray-100">32–128GB</td>
                  <td className="py-3 px-4 text-right tabular-nums text-gray-900 dark:text-gray-100">
                    {t('comparison.ourRange')}
                  </td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-300 hidden sm:table-cell">
                    {t('comparison.ourNote')}
                  </td>
                </tr>
                {MARKET_ANCHORS.map((a) => (
                  <tr key={a.name} className="border-b border-gray-100 dark:border-gray-800/50 last:border-0">
                    <td className="py-3 px-4 text-gray-700 dark:text-gray-300">{a.name}</td>
                    <td className="py-3 px-4 text-right tabular-nums text-gray-600 dark:text-gray-400">{a.vramGB}GB</td>
                    <td className="py-3 px-4 text-right tabular-nums text-gray-600 dark:text-gray-400">
                      {formatJPY(a.approxJPY, loc)}
                    </td>
                    <td className="py-3 px-4 text-gray-500 dark:text-gray-400 hidden sm:table-cell">{a.note[loc]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
      <section className="px-6 py-16 text-center border-t border-gray-200 dark:border-gray-800">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 max-w-xl mx-auto">
          {t('hero.valueProp')}
        </p>
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
