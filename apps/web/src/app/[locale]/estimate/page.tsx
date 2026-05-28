import Link from 'next/link';
import { getTranslations, setRequestLocale } from 'next-intl/server';
import { EstimateClient } from '@/components/EstimateClient';
import { MODEL_FAMILIES } from '@llm-local/shared';

export const revalidate = 3600;

export default async function EstimatePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations('estimate');

  return (
    <div className="mx-auto max-w-2xl px-6 py-12 sm:py-16">
      <div className="mb-10">
        <Link
          href={`/${locale}`}
          className="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          &larr; {t('back')}
        </Link>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mt-4">
          {t('title')}
        </h1>
      </div>

      <EstimateClient families={MODEL_FAMILIES} />
    </div>
  );
}
