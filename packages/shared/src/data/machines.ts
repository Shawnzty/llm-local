import type { Machine } from '../types';
import { PRODUCTS_DATA } from './products';

/** Full catalog, derived from products.json. Pages filter/sort as needed. */
export const MACHINES: Machine[] = PRODUCTS_DATA.products;
