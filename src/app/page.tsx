import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex-1 flex items-center justify-center px-6 py-20">
      <div className="w-full max-w-2xl space-y-4">
        <Link href="/estimate" className="group block">
          <div className="rounded-2xl border border-gray-200 dark:border-gray-800
            bg-white dark:bg-gray-900 p-8 sm:p-10
            shadow-sm hover:shadow-md hover:border-gray-300 dark:hover:border-gray-700
            transition-all duration-200">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1.5">
                  How much VRAM do I need?
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Select a model and get an estimate.
                </p>
              </div>
              <span className="text-gray-300 dark:text-gray-600 group-hover:text-gray-400 dark:group-hover:text-gray-500 transition-colors mt-1">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </span>
            </div>
          </div>
        </Link>

        <Link href="/compatibility" className="group block">
          <div className="rounded-2xl border border-gray-200 dark:border-gray-800
            bg-white dark:bg-gray-900 p-8 sm:p-10
            shadow-sm hover:shadow-md hover:border-gray-300 dark:hover:border-gray-700
            transition-all duration-200">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1.5">
                  Can this model run on my GPU?
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Check model-to-hardware compatibility.
                </p>
              </div>
              <span className="text-gray-300 dark:text-gray-600 group-hover:text-gray-400 dark:group-hover:text-gray-500 transition-colors mt-1">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </span>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
