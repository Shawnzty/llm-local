const VERDICT_STYLES = {
  yes: {
    bg: 'bg-emerald-50 dark:bg-emerald-950/40',
    border: 'border-emerald-200 dark:border-emerald-800/50',
    text: 'text-emerald-700 dark:text-emerald-400',
    dot: 'bg-emerald-500',
    label: 'Yes',
    description: 'This model should run comfortably on this GPU.',
  },
  maybe: {
    bg: 'bg-amber-50 dark:bg-amber-950/40',
    border: 'border-amber-200 dark:border-amber-800/50',
    text: 'text-amber-700 dark:text-amber-400',
    dot: 'bg-amber-500',
    label: 'Maybe',
    description: 'Tight fit. May work with careful configuration.',
  },
  no: {
    bg: 'bg-red-50 dark:bg-red-950/40',
    border: 'border-red-200 dark:border-red-800/50',
    text: 'text-red-700 dark:text-red-400',
    dot: 'bg-red-500',
    label: 'No',
    description: 'Insufficient VRAM for this model.',
  },
} as const;

export function CompatibilityBadge({ verdict }: { verdict: 'yes' | 'maybe' | 'no' }) {
  const style = VERDICT_STYLES[verdict];

  return (
    <div className={`rounded-2xl border ${style.bg} ${style.border} p-5`}>
      <div className="flex items-center gap-3 mb-2">
        <span className={`h-3 w-3 rounded-full ${style.dot}`} />
        <span className={`text-lg font-semibold ${style.text}`}>{style.label}</span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400">{style.description}</p>
    </div>
  );
}
