import Link from 'next/link';
import { CompatibilityClient } from '@/components/CompatibilityClient';
import { fetchFamilies, fetchGpus } from '@/lib/api';

export const revalidate = 3600;

export default async function CompatibilityPage() {
  const [families, gpus] = await Promise.all([fetchFamilies(), fetchGpus()]);

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
          Can this model run on my GPU?
        </h1>
      </div>

      <CompatibilityClient families={families} gpus={gpus} />
    </div>
  );
}
