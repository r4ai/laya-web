import * as ort from "onnxruntime-web";
import { loadWithRuntime } from "./runtime.js";
import { readAsset, resolveModelUrl } from "./node-assets.js";
import type { LoadOptions } from "./types.js";

/**
 * Loads a Laya checkpoint using the Node.js WASM runtime.
 *
 * @remarks
 * Requires Node.js 22+. Relative paths resolve against the working directory.
 * Both `auto` and `wasm` select single-threaded WASM; `webgpu` is unsupported.
 * Bundled WASM assets resolve automatically, so omit `wasmPaths`.
 *
 * @param options - Local model directory or HTTP(S)/file URL and load controls
 * @returns An initialized agent; call its `dispose()` method when finished
 * @throws TypeError - Missing model location or unsupported backend/protocol
 * @throws Error - Invalid assets, read failure, or canceled initialization
 *
 * @example
 * ```ts
 * const agent = await load({ modelUrl: "./models/laya" });
 * try {
 *   console.log(await agent.predict("Refund requested", {
 *     refund: { type: "noul", instructions: "Is a refund requested?" },
 *   }));
 * } finally {
 *   await agent.dispose();
 * }
 * ```
 */
export async function load(options: LoadOptions) {
  if (!options.modelUrl) throw new TypeError("modelUrl is required");
  if (options.backend === "webgpu")
    throw new TypeError('Node.js supports the "wasm" backend, not "webgpu"');
  return loadWithRuntime(
    {
      ...options,
      backend:
        options.backend === undefined || options.backend === "auto"
          ? "wasm"
          : options.backend,
    },
    ort,
    resolveModelUrl(options.modelUrl),
    readAsset,
  );
}

export type { Agent } from "./agent.js";
export type {
  Answer,
  Backend,
  Json,
  LoadOptions,
  LoadProgress,
  Prediction,
  Question,
  Questions,
  State,
} from "./types.js";
