import type * as ort from "onnxruntime-web/webgpu";
import { Agent } from "./agent.js";
import type { Driver } from "./agent.js";
import { createTokenizer, download, downloadAssets } from "./assets.js";
import type { AssetReader } from "./assets.js";
import { halfToFloat, validateConfig } from "./core.js";
import type {
  Backend,
  LoadOptions,
  ModelConfig,
  PreparedQuestion,
} from "./types.js";

class OnnxDriver implements Driver {
  constructor(
    private readonly runtime: typeof ort,
    readonly backend: Backend,
    private readonly session: ort.InferenceSession,
    private embeddings: Uint16Array,
    private readonly config: ModelConfig,
  ) {}

  async run(question: PreparedQuestion) {
    const { hiddenSize, vocabSize } = this.config;
    const values = new Float32Array(question.ids.length * hiddenSize);
    question.ids.forEach((id, row) => {
      if (!Number.isInteger(id) || id < 0 || id >= vocabSize)
        throw new RangeError(`Invalid token id ${id}`);
      for (let col = 0; col < hiddenSize; col++)
        values[row * hiddenSize + col] = halfToFloat(
          this.embeddings[id * hiddenSize + col],
        );
    });
    const markers =
      question.markers.length === 1
        ? [...question.markers, 0]
        : question.markers;
    const mask = markers.map((_, i) => (i < question.markers.length ? 1 : 0));
    const feeds = {
      embeddings: new this.runtime.Tensor("float32", values, [
        1,
        question.ids.length,
        hiddenSize,
      ]),
      marker_pos: new this.runtime.Tensor(
        "int64",
        BigInt64Array.from(markers.map(BigInt)),
        [markers.length],
      ),
      marker_mask: new this.runtime.Tensor("bool", Uint8Array.from(mask), [
        markers.length,
      ]),
      qtype: new this.runtime.Tensor(
        "int64",
        BigInt64Array.from([BigInt(question.qtype)]),
        [1],
      ),
    };
    let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
    try {
      outputs = await this.session.run(feeds);
      return {
        logits: Array.from(outputs.logits.data as Float32Array),
        action: Array.from(outputs.action_logits.data as Float32Array),
      };
    } finally {
      for (const tensor of Object.values(feeds)) tensor.dispose();
      for (const tensor of Object.values(outputs ?? {})) tensor.dispose();
    }
  }
  async dispose() {
    try {
      await this.session.release();
    } finally {
      this.embeddings = new Uint16Array();
    }
  }
}

async function createSession(
  runtime: typeof ort,
  graph: Uint8Array,
  data: Uint8Array,
  options: LoadOptions,
): Promise<{ session: ort.InferenceSession; backend: Backend }> {
  const create = (backend: Backend) =>
    runtime.InferenceSession.create(graph, {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      externalData: [{ path: "model.onnx.data", data }],
      // Preserve portable ONNX ops rather than introducing CPU-only fused operators.
      graphOptimizationLevel: "disabled",
    });
  if (options.backend === "wasm")
    return { session: await create("wasm"), backend: "wasm" };
  try {
    return { session: await create("webgpu"), backend: "webgpu" };
  } catch (error) {
    if (options.backend === "webgpu") throw error;
    options.signal?.throwIfAborted();
    options.onProgress?.({
      phase: "fallback",
      message: `WebGPU initialization failed; using WASM. ${String(error)}`,
    });
    return { session: await create("wasm"), backend: "wasm" };
  }
}

export async function loadWithRuntime(
  options: LoadOptions,
  runtime: typeof ort,
  base: URL,
  read?: AssetReader,
): Promise<Agent> {
  if (!options.modelUrl) throw new TypeError("modelUrl is required");
  if (options.backend && !["auto", "wasm", "webgpu"].includes(options.backend))
    throw new TypeError("Unknown backend");
  options.signal?.throwIfAborted();
  // ORT environment settings are process-wide: configure consistently before first load.
  runtime.env.wasm.numThreads = 1;
  if (options.wasmPaths) runtime.env.wasm.wasmPaths = options.wasmPaths;
  const get = (file: string, bytes?: number) =>
    download(base, file, options, bytes, read);
  const json = (data: Uint8Array) => JSON.parse(new TextDecoder().decode(data));
  const config: unknown = json(await get("config.json"));
  validateConfig(config);
  const files = [
    "embeddings.f16.bin",
    "model.onnx.data",
    "model.onnx",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
  ];
  const assets = await downloadAssets(
    base,
    Object.fromEntries(
      files.map((file) => [file, config.files?.[file]?.bytes]),
    ),
    options,
    read,
  );
  const tokenizer = createTokenizer(
    json(assets["tokenizer/tokenizer.json"]),
    json(assets["tokenizer/tokenizer_config.json"]),
  );
  const embeddingBytes = assets["embeddings.f16.bin"];
  if (embeddingBytes.byteLength !== config.vocabSize * config.hiddenSize * 2)
    throw new Error("Embedding size does not match config");
  const embeddings = new Uint16Array(
    embeddingBytes.buffer,
    embeddingBytes.byteOffset,
    embeddingBytes.byteLength / 2,
  );
  const graph = assets["model.onnx"];
  const data = assets["model.onnx.data"];
  options.signal?.throwIfAborted();
  options.onProgress?.({ phase: "initialize" });
  const { session, backend } = await createSession(
    runtime,
    graph,
    data,
    options,
  );
  if (options.signal?.aborted) {
    await session.release();
    options.signal.throwIfAborted();
  }
  return new Agent(
    config,
    tokenizer,
    new OnnxDriver(runtime, backend, session, embeddings, config),
  );
}
