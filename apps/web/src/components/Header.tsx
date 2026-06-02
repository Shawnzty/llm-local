'use client';

import Link from 'next/link';
import { useLocale, useTranslations } from 'next-intl';
import { ThemeToggle } from './ThemeToggle';
import { LocaleSwitcher } from './LocaleSwitcher';

export function Header() {
  const locale = useLocale();
  const t = useTranslations('nav');
  const tb = useTranslations();

  return (
    <header className="sticky top-0 z-50 border-b border-gray-200/60 dark:border-gray-800/60 bg-gray-50/80 dark:bg-gray-950/80 backdrop-blur-xl">
      <div className="mx-auto max-w-5xl px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link
            href={`/${locale}`}
            className="text-[15px] font-semibold tracking-tight text-gray-900 dark:text-gray-100 hover:opacity-70 transition-opacity"
          >
            {tb('brand')}
          </Link>
          <nav className="hidden sm:flex items-center gap-4">
            <Link
              href={`/${locale}/machines`}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              {t('machines')}
            </Link>
            <Link
              href={`/${locale}/estimate`}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              {t('estimate')}
            </Link>
            <Link
              href={`/${locale}/contact`}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              {t('contact')}
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-3">
          <LocaleSwitcher />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
