import type { Currency, Locale, LocalizedText, MarketAnchor, Tier, TierInfo } from '../types';

/** Which currency to display for each locale. */
export const LOCALE_CURRENCY: Record<Locale, Currency> = {
  ja: 'JPY',
  en: 'USD',
};

export const BRAND = {
  name: { ja: 'LLMローカル', en: 'LLM Local' } as LocalizedText,
  tagline: {
    ja: 'ローカルLLM推論を、誰でも手の届く価格で。',
    en: "Local LLM inference, at a price within everyone's reach.",
  } as LocalizedText,
};

/** Global value proposition shared across all machines. */
export const VALUE_PROP_GLOBAL: LocalizedText = {
  ja: '完成・テスト・PSE換装・1年保証済みの整機を、自作パーツを買い揃える金額とほぼ同額で。',
  en: 'A fully assembled, tested, PSE-converted, 1-year-warrantied unit for about the cost of buying DIY parts in Japan.',
};

/** Common GPU platform note (all machines use Tesla V100). */
export const GPU_PLATFORM = {
  gpu: 'NVIDIA Tesla V100 PCIe (Volta, HBM2)',
  note: {
    ja: 'FP16/INT4量子化推論に最適。FP8/FP4非対応のため最新学習用途には非推奨。',
    en: 'Best for FP16/INT4 quantized inference. No FP8/FP4, so not recommended for cutting-edge training.',
  } as LocalizedText,
};

export const WARRANTY: LocalizedText = {
  ja: '1年整機保証 + GPU専用保証',
  en: '1-year whole-unit warranty + dedicated GPU warranty',
};

export const PSE_COMPLIANCE: LocalizedText = {
  ja: '内蔵電源はPSE認証品（電気用品安全法 準拠）',
  en: 'PSE-certified power supply unit in Japan (Electrical Appliances Safety Act)',
};

export const TIERS: Record<Tier, TierInfo> = {
  entry: { label: { ja: 'エントリー', en: 'Entry' }, vram: '32–48GB' },
  mid: { label: { ja: 'ミドル', en: 'Mid-range' }, vram: '64–96GB' },
  flagship: { label: { ja: 'フラッグシップ', en: 'Flagship' }, vram: '128GB' },
};

export const TIER_ORDER: Tier[] = ['entry', 'mid', 'flagship'];

/** Competitor reference points for the comparison table. */
export const MARKET_ANCHORS: MarketAnchor[] = [
  {
    name: 'RTX 5060 Ti 16GB',
    vramGB: 16,
    approxJPY: 85000,
    note: {
      ja: '16GBはコンシューマが安い → 当社は32GB以上で勝負',
      en: 'Consumers win at 16GB → we compete at 32GB+',
    },
  },
  {
    name: 'RTX 5090 32GB',
    vramGB: 32,
    approxJPY: 450000,
    note: {
      ja: '32GBの対抗。速いが高い・VRAMは頭打ち',
      en: '32GB rival. Faster but pricey, VRAM capped',
    },
  },
  {
    name: 'Applied CERVO/HPC (RTX PRO 6000 Blackwell)',
    vramGB: 192,
    approxJPY: 12911000,
    note: {
      ja: 'エンタープライズ向け。当社の約9倍価格',
      en: 'Enterprise-grade. ~9× our price',
    },
  },
  {
    name: 'Mac Studio M3 Ultra 512GB',
    vramGB: 512,
    approxJPY: 1000000,
    note: {
      ja: '統合メモリで大型可だが演算弱・MLX',
      en: 'Big models via unified memory but weak compute / MLX',
    },
  },
];
