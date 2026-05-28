import { notFound } from 'next/navigation';
import Link from 'next/link';
import { getTranslations, setRequestLocale } from 'next-intl/server';
import {
  MACHINES,
  MODEL_FAMILIES,
  GPU_PROFILES,
  getMachineCompatibilityMatrix,
} from '@llm-local/shared';
import type { Locale } from '@llm-local/shared';
import { CompatibilityMatrix } from '@/components/CompatibilityMatrix';

export function generateStaticParams() {
  return MACHINES.map((m) => ({ slug: m.slug }));
}

export default async function MachineDetailPage({
  params,
}: {
  params: Promise<{ locale: string; slug: string }>;
}) {
  const { locale, slug } = await params;
  setRequestLocale(locale);
  const loc = locale as Locale;
  const t = await getTranslations('machines');
  const tc = await getTranslations('nav');

  const machine = MACHINES.find((m) => m.slug === slug);
  if (!machine) notFound();

  const matrix = getMachineCompatibilityMatrix(machine, MODEL_FAMILIES, GPU_PROFILES);
  const gpuCount = machine.gpus.reduce((n, g) => n + g.count, 0);
  const gpuLabel = machine.gpus
    .map((g) => {
      const gpu = GPU_PROFILES.find((p) => p.id === g.gpuId);
      return gpu ? `${gpu.name}${g.count > 1 ? ` ×${g.count}` : ''}` : g.gpuId;
    })
    .join(' + ');

  const yesCount = matrix.filter((r) => r.verdict === 'yes').length;
  const maybeCount = matrix.filter((r) => r.verdict === 'maybe').length;

  return (
    <div className="mx-auto max-w-4xl px-6 py-12 sm:py-16">
      {/* Back */}
      <Link
        href={`/${locale}/machines`}
        className="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
      >
        &larr; {t('title')}
      </Link>

      {/* Hero */}
      <div className="mt-6 mb-10">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          {machine.name[loc]}
        </h1>
        <p className="mt-2 text-gray-500 dark:text-gray-400">
          {machine.tagline[loc]}
        </p>
        <div className="mt-4 flex items-center gap-4">
          <span className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            {machine.priceDisplay[loc]}
          </span>
          <Link
            href={`/${locale}/contact?machine=${machine.id}`}
            className="inline-flex items-center px-5 py-2 rounded-xl bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            {t('inquire')}
          </Link>
        </div>
      </div>

      {/* Description */}
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-8 max-w-2xl">
        {machine.description[loc]}
      </p>

      {/* Specs */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 mb-8">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
          {t('specs')}
        </h2>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
          <div>
            <dt className="text-gray-400 dark:text-gray-500">{t('gpuConfig')}</dt>
            <dd className="font-medium text-gray-900 dark:text-gray-100">{gpuLabel}</dd>
          </div>
          <div>
            <dt className="text-gray-400 dark:text-gray-500">{t('cpu')}</dt>
            <dd className="font-medium text-gray-900 dark:text-gray-100">{machine.cpu}</dd>
          </div>
          <div>
            <dt className="text-gray-400 dark:text-gray-500">{t('ram')}</dt>
            <dd className="font-medium text-gray-900 dark:text-gray-100">{machine.ramGB} GB</dd>
          </div>
          <div>
            <dt className="text-gray-400 dark:text-gray-500">{t('storage')}</dt>
            <dd className="font-medium text-gray-900 dark:text-gray-100">{machine.storage}</dd>
          </div>
          {Object.entries(machine.specs).map(([key, value]) => (
            <div key={key}>
              <dt className="text-gray-400 dark:text-gray-500">{key}</dt>
              <dd className="font-medium text-gray-900 dark:text-gray-100">{value}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Compatibility matrix */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider">
            {t('compatibleModels')}
          </h2>
          <span className="text-xs text-gray-400 dark:text-gray-500">
            {yesCount} ✓ / {maybeCount} △ / {matrix.length} total
          </span>
        </div>
        <CompatibilityMatrix matrix={matrix} gpuCount={gpuCount} />
      </div>

      {/* Bottom CTA */}
      <div className="mt-8 text-center">
        <Link
          href={`/${locale}/contact?machine=${machine.id}`}
          className="inline-flex items-center px-6 py-3 rounded-xl bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:opacity-90 transition-opacity"
        >
          {t('inquire')}
        </Link>
      </div>
    </div>
  );
}
