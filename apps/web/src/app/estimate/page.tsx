import Link from 'next/link';
import { EstimateClient } from '@/components/EstimateClient';
import { fetchFamilies } from '@/lib/api';

export const revalidate = 3600;

export default async function EstimatePage() {
  const families = await fetchFamilies();

  return (
    <div className="mx-auto max-w-2xl px-6 py-12 sm:py-16">
      <div className="mb-10">
        <Link
          href="/"
          className="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          &larr; Back
        </Link>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mt-4">
          How much VRAM do I need?
        </h1>
      </div>

      <EstimateClient families={families} />
    </div>
  );
}
