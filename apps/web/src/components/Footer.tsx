import { BRAND } from '@tadzuna/shared';
import type { Locale } from '@tadzuna/shared';

export function Footer({ locale }: { locale: Locale }) {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 px-6 py-8">
      <div className="mx-auto max-w-5xl flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {BRAND.name[locale]}
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {BRAND.tagline[locale]}
          </p>
        </div>
        <p className="text-xs text-gray-400 dark:text-gray-500">
          © {year} {BRAND.name[locale]}
        </p>
      </div>
    </footer>
  );
}
