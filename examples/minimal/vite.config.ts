import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";

const repoRoot = new URL("../../", import.meta.url);
const ortFile = (name: string) =>
  new URL(`node_modules/onnxruntime-web/dist/${name}`, repoRoot);
// ONNX Runtime Web resolves these by name under wasmPaths, and its JS and Wasm
// must come from the same version, so dev and build serve the same two files.
const ortFiles = [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
];

/** Serves the ONNX Runtime Web binaries from `/ort/` and ships the license files. */
const selfHostOrt = (): Plugin => ({
  name: "self-host-ort",
  configureServer(server) {
    server.middlewares.use("/ort", (req, res, next) => {
      const name = req.url?.slice(1).split("?")[0] ?? "";
      if (!ortFiles.includes(name)) return next();
      res.setHeader(
        "Content-Type",
        name.endsWith(".wasm") ? "application/wasm" : "text/javascript",
      );
      createReadStream(ortFile(name)).on("error", next).pipe(res);
    });
  },
  async generateBundle() {
    const assets: Record<string, URL> = {
      LICENSE: new URL("LICENSE", repoRoot),
      NOTICE: new URL("NOTICE", repoRoot),
      ...Object.fromEntries(
        ortFiles.map((name) => [`ort/${name}`, ortFile(name)]),
      ),
    };
    for (const [fileName, url] of Object.entries(assets))
      this.emitFile({ type: "asset", fileName, source: await readFile(url) });
  },
});

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  plugins: [solid(), selfHostOrt()],
});
