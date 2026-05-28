import { Suspense } from 'react';
import { getTranslations, setRequestLocale } from 'next-intl/server';
import { InquiryForm } from '@/components/InquiryForm';

export default async function ContactPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations('contact');

  return (
    <div className="mx-auto max-w-xl px-6 py-12 sm:py-16">
      <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {t('title')}
      </h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        {t('subtitle')}
      </p>
      <Suspense>
        <InquiryForm />
      </Suspense>
    </div>
  );
}
