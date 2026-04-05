interface ResultRowProps {
  label: string;
  value: string;
  sublabel?: string;
  prominent?: boolean;
}

function ResultRow({ label, value, sublabel, prominent }: ResultRowProps) {
  return (
    <div className="flex items-baseline justify-between py-3">
      <div>
        <span className="text-sm text-gray-500 dark:text-gray-400">{label}</span>
        {sublabel && (
          <span className="ml-1.5 text-xs text-gray-400 dark:text-gray-500">{sublabel}</span>
        )}
      </div>
      <span
        className={`font-medium tabular-nums ${
          prominent
            ? 'text-lg text-gray-900 dark:text-gray-50'
            : 'text-[15px] text-gray-700 dark:text-gray-200'
        }`}
      >
        {value}
      </span>
    </div>
  );
}

interface ResultCardProps {
  rows: ResultRowProps[];
}

export function ResultCard({ rows }: ResultCardProps) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800
      bg-white dark:bg-gray-900 p-6 divide-y divide-gray-100 dark:divide-gray-800/80">
      {rows.map((row, i) => (
        <ResultRow key={i} {...row} />
      ))}
    </div>
  );
}
