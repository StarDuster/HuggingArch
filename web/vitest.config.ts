// Kept apart from `vite.config.ts` on purpose: that file is what the
// production build reads, and giving it a typed `test` block would make the
// build import `vitest/config`. Nothing here needs the app's plugins — the
// modules under test are plain TypeScript, so esbuild's TS/JSX transform
// (configured by `tsconfig.json`) is enough.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // These are the pure shared modules: no DOM, so no jsdom dependency.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
