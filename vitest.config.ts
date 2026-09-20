import { defineConfig } from "vitest/config";
import solid from "vite-plugin-solid";
export default defineConfig({
  plugins: [solid()],
  test: {
    environment: "node",
    include: ["tests/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      include: [
        "src/**/*.ts",
        "examples/minimal/app.tsx",
        "scripts/validate-pages.mjs",
      ],
    },
  },
});
