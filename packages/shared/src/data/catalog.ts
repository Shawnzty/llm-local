import type { Currency, Locale, LocalizedText, MarketAnchor, Tier, TierInfo } from '../types';
import { PRODUCTS_DATA } from './products';

/**
 * Catalog-level constants, all derived from products.json (the single
 * source of truth). Import names are stable so pages don't change when
 * the underlying data does.
 */

/** Which currency to display for each locale. */
export const LOCALE_CURRENCY: Record<Locale, Currency> = PRODUCTS_DATA.meta.localeCurrency;

export const BRAND = PRODUCTS_DATA.brand;

/** Global value proposition shared across all machines. */
export const VALUE_PROP_GLOBAL: LocalizedText = PRODUCTS_DATA.shared.valuePropGlobal;

/** Common GPU platform note (all machines use Tesla V100). */
export const GPU_PLATFORM = PRODUCTS_DATA.shared.gpuPlatform;

export const WARRANTY: LocalizedText = PRODUCTS_DATA.shared.warranty.text;

export const PSE_COMPLIANCE: LocalizedText = PRODUCTS_DATA.shared.compliance.pse;

export const TIERS: Record<Tier, TierInfo> = PRODUCTS_DATA.tiers;

export const TIER_ORDER: Tier[] = ['entry', 'mid', 'flagship'];

/** Competitor reference points for the comparison table. */
export const MARKET_ANCHORS: MarketAnchor[] = PRODUCTS_DATA.shared.marketAnchors;
