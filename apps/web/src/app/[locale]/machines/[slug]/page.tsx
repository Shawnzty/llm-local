import { notFound } from 'next/navigation';
import Link from 'next/link';
import { getTranslations, setRequestLocale } from 'next-intl/server';
import { MACHINES, LOCALE_CURRENCY, GPU_PLATFORM, PSE_COMPLIANCE } from '@tadzuna/shared';
import type { Locale } from '@tadzuna/shared';

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
  const currency = LOCALE_CURRENCY[loc];
  const t = await getTranslations('machines');
  const tc = await getTranslations('common');

  const machine = MACHINES.find((m) => m.slug === slug);
  if (!machine) notFound();

  const measuredPerf = machine.performance.filter((p) => p.tokensPerSec != null);
  const perfNote = machine.performance.find((p) => p.note)?.note;

  const specs: { label: string; value: string }[] = [
    { label: t('gpuConfig'), value: `${machine.gpu.model.replace('NVIDIA ', '')} · ${machine.gpu.configLabel}` },
    { label: t('vram'), value: `${machine.gpu.vramTotalGB}GB (${machine.gpu.memoryType})` },
    { label: t('cpu'), value: `${machine.cpu.model} (${machine.cpu.cores}C/${machine.cpu.threads}T)` },
    { label: t('motherboard'), value: machine.motherboard[loc] },
    { label: t('ram'), value: `${machine.ram.sizeGB}GB ${machine.ram.type}` },
    { label: t('storage'), value: `${machine.storage.label} ${machine.storage.type}` },
    {
      label: t('psu'),
      value: `${machine.psu.wattage}W ${machine.psu.rating}${machine.psu.pseCertified ? ' · PSE' : ''}`,
    },
    { label: t('chassis'), value: machine.chassis[loc] },
    { label: t('cooling'), value: machine.cooling[loc] },
  ];

  const inquireHref = `/${locale}/contact?machine=${machine.id}`;

  return (
    <div className="mx-auto max-w-4xl px-6 py-12 sm:py-16">
      {/* Back */}
      <Link
        href={`/${locale}/machines`}
        className="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
      >
        &larr; {t('back')}
      </Link>

      {/* Hero */}
      <div className="mt-6 mb-8">
        <p className="text-xs font-medium uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-2">
          {machine.category[loc]}
        </p>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">{machine.name}</h1>
        <p className="mt-2 text-gray-500 dark:text-gray-400 max-w-2xl">{machine.tagline[loc]}</p>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {machine.badges[loc].map((b) => (
            <span
              key={b}
              className="inline-block px-2 py-0.5 rounded-md bg-gray-100 dark:bg-gray-800 text-xs text-gray-600 dark:text-gray-300"
            >
              {b}
            </span>
          ))}
        </div>

        <div className="mt-5 flex items-center gap-4">
          <span className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tabular-nums">
            {machine.priceLabels[currency]}
          </span>
          <Link
            href={inquireHref}
            className="inline-flex items-center px-5 py-2 rounded-xl bg-leather text-[#FBF4EC] text-sm font-medium hover:opacity-90 transition-opacity"
          >
            {t('inquire')}
          </Link>
        </div>
      </div>

      {/* Performance */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 mb-6">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
          {t('performance')}
        </h2>
        {measuredPerf.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400">
                <th className="text-left font-medium py-2">{tc('model')}</th>
                <th className="text-left font-medium py-2">{t('quant')}</th>
                <th className="text-right font-medium py-2">{t('tokensPerSec')}</th>
              </tr>
            </thead>
            <tbody>
              {measuredPerf.map((p) => (
                <tr key={`${p.model}-${p.quant}`} className="border-b border-gray-100 dark:border-gray-800/50 last:border-0">
                  <td className="py-2.5 text-gray-900 dark:text-gray-100">{p.model}</td>
                  <td className="py-2.5 text-gray-600 dark:text-gray-300">{p.quant}</td>
                  <td className="py-2.5 text-right tabular-nums font-medium text-gray-900 dark:text-gray-100">
                    {p.tokensPerSec}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {perfNote ? perfNote[loc] : t('perfPending')}
          </p>
        )}
      </div>

      {/* What it can run */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 mb-6">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
          {t('canRun')}
        </h2>
        <ul className="space-y-2">
          {machine.canRun[loc].map((item) => (
            <li key={item} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
              <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Specs */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 mb-6">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
          {t('specs')}
        </h2>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-3 text-sm">
          {specs.map((s) => (
            <div key={s.label} className="flex justify-between gap-4 border-b border-gray-100 dark:border-gray-800/50 pb-2">
              <dt className="text-gray-400 dark:text-gray-500 shrink-0">{s.label}</dt>
              <dd className="font-medium text-gray-900 dark:text-gray-100 text-right">{s.value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-gray-400 dark:text-gray-500">
          {t('gpuPlatform')}: {GPU_PLATFORM.gpu} — {GPU_PLATFORM.note[loc]}
        </p>
      </div>

      {/* Value props + best for */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
        <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
            {t('valueProps')}
          </h2>
          <ul className="space-y-2">
            {machine.valueProps[loc].map((item) => (
              <li key={item} className="text-sm text-gray-700 dark:text-gray-300">{item}</li>
            ))}
          </ul>
        </div>
        <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider mb-4">
            {t('bestFor')}
          </h2>
          <div className="flex flex-wrap gap-2">
            {machine.bestFor[loc].map((item) => (
              <span
                key={item}
                className="inline-block px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-sm text-gray-700 dark:text-gray-300"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Warranty / compliance / disclaimers */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 p-6 mb-8 text-sm space-y-2">
        <div className="flex gap-2">
          <span className="text-gray-400 dark:text-gray-500 shrink-0">{t('warranty')}:</span>
          <span className="text-gray-700 dark:text-gray-300">{machine.warranty[loc]}</span>
        </div>
        <div className="flex gap-2">
          <span className="text-gray-400 dark:text-gray-500 shrink-0">{t('compliance')}:</span>
          <span className="text-gray-700 dark:text-gray-300">{PSE_COMPLIANCE[loc]}</span>
        </div>
        <ul className="pt-1 text-xs text-gray-400 dark:text-gray-500 space-y-1">
          {machine.disclaimers.map((d) => (
            <li key={d.en}>· {d[loc]}</li>
          ))}
        </ul>
      </div>

      {/* Bottom CTA */}
      <div className="text-center">
        <Link
          href={inquireHref}
          className="inline-flex items-center px-6 py-3 rounded-xl bg-leather text-[#FBF4EC] text-sm font-medium hover:opacity-90 transition-opacity"
        >
          {t('inquire')}
        </Link>
      </div>
    </div>
  );
}
