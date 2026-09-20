import solid from "vite-plugin-solid";
import { defineConfig } from "vite";
import {
  createReadStream,
  existsSync,
  readdirSync,
  readFileSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
const ortDir = fileURLToPath(
  new URL("../../node_modules/onnxruntime-web/dist/", import.meta.url),
);
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  plugins: [
    solid(),
    {
      name: "self-host-ort",
      configureServer(server) {
        server.middlewares.use("/ort/", (req, res, next) => {
          const name = req.url?.split("?")[0]?.slice(1) ?? "";
          if (!/^ort-[\w.-]+\.(wasm|mjs)$/.test(name)) return next();
          const path = resolve(ortDir, name);
          if (!existsSync(path)) return next();
          res.setHeader(
            "Content-Type",
            name.endsWith(".wasm") ? "application/wasm" : "text/javascript",
          );
          createReadStream(path).pipe(res);
        });
      },
      generateBundle() {
        // ORT's JS and WASM must have exactly the same version, including in production.
        for (const file of readdirSync(ortDir).filter((f) =>
          /^ort-wasm.*\.(wasm|mjs)$/.test(f),
        )) {
          this.emitFile({
            type: "asset",
            fileName: `ort/${file}`,
            source: readFileSync(resolve(ortDir, file)),
          });
        }
      },
    },
  ],
});
