'use client';

import Link from 'next/link';
import { ThemeToggle } from './ThemeToggle';

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-gray-200/60 dark:border-gray-800/60 bg-gray-50/80 dark:bg-gray-950/80 backdrop-blur-xl">
      <div className="mx-auto max-w-4xl px-6 h-14 flex items-center justify-between">
        <Link
          href="/"
          className="text-[15px] font-semibold tracking-tight text-gray-900 dark:text-gray-100 hover:opacity-70 transition-opacity"
        >
          LLM Local
        </Link>
        <ThemeToggle />
      </div>
    </header>
  );
}
