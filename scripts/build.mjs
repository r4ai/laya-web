import { execFileSync } from "node:child_process";
import { build } from "esbuild";
execFileSync(
  "pnpm",
  ["exec", "tsc", "-p", "tsconfig.build.json", "--emitDeclarationOnly"],
  { stdio: "inherit" },
);
// Bundle the patched tokenizer so package consumers do not need pnpm patches.
await build({
  entryPoints: ["src/index.ts"],
  outfile: "dist/index.js",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: true,
  external: ["onnxruntime-web/webgpu"],
  legalComments: "eof",
});

// Keep Node built-ins and the Node-compatible ORT entry out of browser bundles.
await build({
  entryPoints: ["src/node.ts"],
  outfile: "dist/node.js",
  bundle: true,
  format: "esm",
  platform: "node",
  target: "node22",
  sourcemap: true,
  external: ["onnxruntime-web"],
  legalComments: "eof",
});
