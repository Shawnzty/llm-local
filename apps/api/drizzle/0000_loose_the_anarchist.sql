CREATE TABLE IF NOT EXISTS "gpu_profiles" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"vendor" text NOT NULL,
	"vram_gb" real NOT NULL,
	"tier" text NOT NULL,
	"source" text,
	"last_updated" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "model_families" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "model_variants" (
	"id" text PRIMARY KEY NOT NULL,
	"family_id" text NOT NULL,
	"size_label" text NOT NULL,
	"parameter_count" real NOT NULL,
	"layers" integer NOT NULL,
	"hidden_size" integer NOT NULL,
	"num_attention_heads" integer NOT NULL,
	"num_kv_heads" integer NOT NULL,
	"head_dim" integer NOT NULL,
	"max_context" integer NOT NULL,
	"type_badges" jsonb NOT NULL,
	"intelligence_score" real,
	"is_moe" boolean NOT NULL,
	"active_parameter_count" real,
	"source" text,
	"last_updated" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "model_variants" ADD CONSTRAINT "model_variants_family_id_model_families_id_fk" FOREIGN KEY ("family_id") REFERENCES "public"."model_families"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
