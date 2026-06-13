import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";
import path from "path";
import { fileURLToPath } from "url";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

// Pin the workspace root to the monorepo root. Otherwise Next infers it by
// walking up for a lockfile and can latch onto an unrelated one higher in the
// tree (there is a real ~/package-lock.json). Derived from this file's path so
// it stays correct on CI / Vercel too.
const workspaceRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const nextConfig: NextConfig = {
  transpilePackages: ["@tadzuna/shared"],
  turbopack: {
    root: workspaceRoot,
  },
};

export default withNextIntl(nextConfig);
