import {
  pgTable,
  text,
  integer,
  real,
  boolean,
  jsonb,
  timestamp,
} from 'drizzle-orm/pg-core';
import type { ModelTypeBadge } from '@llm-local/shared';

export const modelFamilies = pgTable('model_families', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
});

export const modelVariants = pgTable('model_variants', {
  id: text('id').primaryKey(),
  familyId: text('family_id')
    .references(() => modelFamilies.id, { onDelete: 'cascade' })
    .notNull(),
  sizeLabel: text('size_label').notNull(),
  parameterCount: real('parameter_count').notNull(),
  layers: integer('layers').notNull(),
  hiddenSize: integer('hidden_size').notNull(),
  numAttentionHeads: integer('num_attention_heads').notNull(),
  numKVHeads: integer('num_kv_heads').notNull(),
  headDim: integer('head_dim').notNull(),
  maxContext: integer('max_context').notNull(),
  typeBadges: jsonb('type_badges').$type<ModelTypeBadge[]>().notNull(),
  intelligenceScore: real('intelligence_score'),
  isMoE: boolean('is_moe').notNull(),
  activeParameterCount: real('active_parameter_count'),
  source: text('source'),
  lastUpdated: timestamp('last_updated', { withTimezone: true }).defaultNow().notNull(),
});

export const gpuProfiles = pgTable('gpu_profiles', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  vendor: text('vendor').notNull(),
  vramGB: real('vram_gb').notNull(),
  tier: text('tier').notNull(),
  source: text('source'),
  lastUpdated: timestamp('last_updated', { withTimezone: true }).defaultNow().notNull(),
});
