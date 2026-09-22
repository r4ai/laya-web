import * as ort from "onnxruntime-web/webgpu";
import { loadWithRuntime } from "./runtime.js";
import type { LoadOptions } from "./types.js";

/**
 * Loads a Laya model checkpoint into browser memory.
 *
 * @remarks
 * Runtime characteristics:
 * - Downloads assets from `options.modelUrl` (~900 MB total)
 * - Validates manifest structure and file sizes
 * - Initializes ONNX Runtime Web session (WebGPU with WASM fallback)
 * - Executes entirely client-side without remote server calls
 * - Operates in either main thread or Web Worker
 *
 * Global environment configuration:
 * - Sets `ort.env.wasm.numThreads` to `1`
 * - Applies `options.wasmPaths` to ONNX Runtime environment
 *
 * @param options - Checkpoint location and runtime configuration
 * @returns Initialized {@link Agent} instance
 * @throws TypeError - Missing `modelUrl`, invalid backend, or malformed config
 * @throws Error - Asset download failure, size mismatch, or invalid manifest
 * @throws Error - Operation aborted via `options.signal`
 *
 * @example
 * ```ts
 * const agent = await load({
 *   modelUrl: "/models/laya/",
 *   backend: "auto",
 *   wasmPaths: "/ort/",
 *   signal: AbortSignal.timeout(300_000),
 *   onProgress: (event) => {
 *     if (event.phase === "download" && event.total) {
 *       console.log(`${event.file}: ${event.loaded} / ${event.total}`);
 *     }
 *   },
 * });
 * console.log("Running on", agent.backend);
 * ```
 *
 * @see {@link Agent.dispose}
 */
export async function load(options: LoadOptions) {
  if (!options.modelUrl) throw new TypeError("modelUrl is required");
  const base = new URL(
    options.modelUrl.endsWith("/") ? options.modelUrl : `${options.modelUrl}/`,
    globalThis.location?.href,
  );
  return loadWithRuntime(options, ort, base);
}
