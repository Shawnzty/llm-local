import raw from './products.json';
import type { Currency, Locale, LocalizedText, Machine, MarketAnchor, Tier } from '../types';

/**
 * products.json is the single source of truth for the catalog.
 * Edit the JSON to change prices, specs, copy, or add SKUs; everything
 * below derives from it. This module just gives the JSON a precise type.
 *
 * Raw products carry a few fields the app doesn't use (label, images);
 * RawProduct documents that they may be present but are ignored.
 */
interface RawProduct extends Machine {
  label?: string;
  images?: Record<string, unknown>;
}

export interface ProductsData {
  meta: {
    version: string;
    lastUpdated: string;
    locales: Locale[];
    defaultLocale: Locale;
    localeCurrency: Record<Locale, Currency>;
  };
  brand: {
    name: LocalizedText;
    tagline: LocalizedText;
    supportEmail: string;
  };
  shared: {
    warranty: { text: LocalizedText; note: LocalizedText };
    compliance: { pse: LocalizedText };
    gpuPlatform: { gpu: string; note: LocalizedText };
    marketAnchors: MarketAnchor[];
    valuePropGlobal: LocalizedText;
  };
  tiers: Record<Tier, { label: LocalizedText; vram: string; skus: string[] }>;
  products: RawProduct[];
}

export const PRODUCTS_DATA = raw as unknown as ProductsData;
