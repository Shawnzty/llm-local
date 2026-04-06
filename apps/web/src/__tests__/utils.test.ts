import { describe, it, expect } from 'vitest';
import { formatGB, formatContext } from '@/lib/utils';

describe('formatGB', () => {
  it('formats small values with one decimal', () => {
    expect(formatGB(3.9)).toBe('3.9 GB');
    expect(formatGB(7.5)).toBe('7.5 GB');
  });

  it('formats medium values with one decimal', () => {
    expect(formatGB(15.62)).toBe('15.6 GB');
  });

  it('formats large values as integers', () => {
    expect(formatGB(130.39)).toBe('130 GB');
  });
});

describe('formatContext', () => {
  it('formats thousands as K', () => {
    expect(formatContext(8192)).toBe('8K');
    expect(formatContext(131072)).toBe('128K');
    expect(formatContext(32768)).toBe('32K');
  });
});
