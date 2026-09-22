import { expect, it } from "vitest";
import * as ort from "onnxruntime-web";
import { loadWithRuntime } from "../src/runtime.js";
import { readAsset } from "../src/node-assets.js";
import type { LoadOptions, LoadProgress } from "../src/types.js";

const base = new URL("./fixtures/node-model/", import.meta.url);

// A stateful session fake isolates provider selection and ownership from device availability.
function runtime(failGpu = false, duringCreate = () => {}) {
  let released = 0;
  const providers: unknown[] = [];
  const session = {
    release: async () => {
      released++;
    },
    run: async () => ({
      logits: new ort.Tensor("float32", [0, 0], [2]),
      action_logits: new ort.Tensor("float32", [0, 0], [2]),
    }),
  };
  const api = {
    Tensor: ort.Tensor,
    env: { wasm: {} },
    InferenceSession: {
      create: async (
        _graph: unknown,
        options: ort.InferenceSession.SessionOptions,
      ) => {
        providers.push(options.executionProviders);
        if (failGpu && options.executionProviders?.[0] === "webgpu")
          throw new Error("No GPU");
        duringCreate();
        return session;
      },
    },
  } as unknown as typeof ort;
  return {
    api,
    providers,
    get released() {
      return released;
    },
  };
}

it.each([
  [undefined, false, "webgpu", 1],
  ["auto", true, "wasm", 2],
  ["webgpu", false, "webgpu", 1],
  ["wasm", true, "wasm", 1],
] as const)(
  "resolves backend %s with GPU failure %s",
  async (backend, failGpu, expected, attempts) => {
    const fake = runtime(failGpu);
    const progress: LoadProgress[] = [];
    const agent = await loadWithRuntime(
      { modelUrl: base.href, backend, onProgress: (e) => progress.push(e) },
      fake.api,
      base,
      readAsset,
    );
    expect(agent.backend).toBe(expected);
    expect(fake.providers).toHaveLength(attempts);
    expect(progress.filter((e) => e.phase === "fallback")).toHaveLength(
      attempts - 1,
    );
    await agent.predict("test", {
      question: { type: "noul", instructions: "True?" },
    });
    await agent.dispose();
    expect(fake.released).toBe(1);
  },
);
it("explicit WebGPU failure rejects without fallback", async () => {
  const fake = runtime(true);
  await expect(
    loadWithRuntime(
      { modelUrl: base.href, backend: "webgpu" },
      fake.api,
      base,
      readAsset,
    ),
  ).rejects.toThrow("No GPU");
  expect(fake.providers).toHaveLength(1);
});
it("releases a session when canceled during initialization", async () => {
  const controller = new AbortController();
  const fake = runtime(false, () => controller.abort());
  await expect(
    loadWithRuntime(
      { modelUrl: base.href, signal: controller.signal },
      fake.api,
      base,
      readAsset,
    ),
  ).rejects.toThrow();
  expect(fake.released).toBe(1);
});
it.each([{ modelUrl: "" }, { modelUrl: base.href, backend: "invalid" }])(
  "rejects invalid options before reading assets",
  async (options) => {
    const fake = runtime();
    await expect(
      loadWithRuntime(options as LoadOptions, fake.api, base, readAsset),
    ).rejects.toThrow();
    expect(fake.providers).toEqual([]);
  },
);
