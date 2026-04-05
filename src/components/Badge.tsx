import type { ModelTypeBadge } from '@/lib/types';

const BADGE_CONFIG: Record<ModelTypeBadge, { label: string; icon: string }> = {
  vision: { label: 'Vision', icon: '👁' },
  tools: { label: 'Tools', icon: '🔧' },
  thinking: { label: 'Thinking', icon: '💭' },
  audio: { label: 'Audio', icon: '🎵' },
};

export function Badge({ type }: { type: ModelTypeBadge }) {
  const config = BADGE_CONFIG[type];

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg
      bg-gray-100 dark:bg-gray-800 text-xs font-medium text-gray-600 dark:text-gray-300">
      <span className="text-[11px]">{config.icon}</span>
      {config.label}
    </span>
  );
}
