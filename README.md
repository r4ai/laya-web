# @r4ai/laya-web

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-brightgreen)](https://r4ai.github.io/laya-web/)

Client-side runtime for [Laya-MLX](https://github.com/mizorewww/laya-mlx) decision models in the browser, powered by ONNX Runtime Web (WebGPU & Wasm).

Instead of generating unstructured, free-form text like standard LLMs, Laya evaluates structured decision schemas (**typed decisions**) directly over input context:

- **Categorical Choice (`choice`)**: Selects the best-matching option from discrete candidate labels.
- **Ordinal Scoring (`score`)**: Evaluates input against an ordered rubric or severity scale.
- **Proposition Verification (`noul`)**: Determines whether a binary condition holds true.

[Live Demo](https://r4ai.github.io/laya-web/)

---

## Why @r4ai/laya-web?

- **100% Client-Side Execution**: Runs entirely on the user's device. No user text is sent across the network, eliminating API costs, rate limits, and latency spikes.
- **Non-Blocking Inference**: Designed for dedicated Web Workers, ensuring complex classification passes never degrade 60fps UI responsiveness.
- **Adaptive Hardware Acceleration**: Targets WebGPU for GPU execution and falls back seamlessly to single-threaded SIMD WebAssembly when WebGPU is unavailable.
- **Memory-Conscious Embedding Slicing**: Multilingual vocabularies (128k+ tokens) exceed browser WebGPU storage buffer limits. `@r4ai/laya-web` retains raw FP16 embeddings in CPU memory, slices only the active token vectors per inference step, converts them to FP32, and feeds the resulting tensor to the ONNX graph.
- **Calibrated Confidence**: Computes normalized entropy metrics ($0.0$ to $1.0$) alongside raw probabilities, allowing client applications to distinguish high certainty from ambiguity.

---

## Architecture

The runtime decouples UI state orchestration from tensor compute across a Web Worker boundary:

```mermaid
flowchart TD
    subgraph MainThread ["Main Thread (UI)"]
        UI["Web Application"]
        Agent["Agent Client (@r4ai/laya-web)"]
    end

    subgraph WorkerThread ["Web Worker"]
        Driver["OnnxDriver"]
        CPUEmb["CPU Embedding Lookup<br/>(FP16 → FP32 Token Slicing)"]
        ORT["ONNX Runtime Web"]
    end

    subgraph Backends ["Execution Backends"]
        WebGPU["WebGPU (Preferred)"]
        Wasm["Wasm SIMD (Fallback)"]
    end

    UI -->|predict(state, questions)| Agent
    Agent -->|postMessage| Driver
    Driver -->|Token IDs| CPUEmb
    CPUEmb -->|Active Embeddings Tensor| ORT
    ORT -->|Primary Provider| WebGPU
    ORT -.->|Fallback Provider| Wasm
    Driver -->|Formatted Answers| Agent
    Agent -->|Prediction Result| UI
```

---

## Installation

```sh
npm install @r4ai/laya-web onnxruntime-web
# or
pnpm add @r4ai/laya-web onnxruntime-web
```

---

## Usage

Inference should run inside a Web Worker to avoid blocking the browser's UI thread. Ensure model assets (produced by `pnpm model:export`) and ONNX Runtime Web binaries are served statically (see the [Vite configuration example](examples/minimal/vite.config.ts)).

```ts
import { load } from "@r4ai/laya-web";

// 1. Initialize the runtime and load model assets
const agent = await load({
  modelUrl: "/models/laya/",
  backend: "auto", // "webgpu" | "wasm" | "auto"
  wasmPaths: "/ort/",
  onProgress: (progress) => console.log("Load progress:", progress),
});

try {
  // 2. Execute structured predictions over input text or state
  const result = await agent.predict(
    "I was charged twice for my subscription this month. Please issue a refund.",
    {
      department: {
        type: "choice",
        instructions: "Which support department should handle this request?",
        criteria: ["Billing & Refunds", "Technical Support", "Sales"],
      },
      urgency: {
        type: "score",
        instructions: "How urgent is this ticket?",
        criteria: ["Normal", "Elevated", "Immediate"],
      },
      refund: {
        type: "noul",
        instructions: "Does the user explicitly demand a refund?",
      },
    },
  );

  console.log("Active backend:", agent.backend);

  const department = result.answers.department;
  if (department.type === "choice") {
    console.log("Selected department:", department.choice);
    console.log("Distribution:", department.probabilities);
  }
  console.log("Confidence:", department.confidence);
} finally {
  // 3. Release ONNX sessions and memory buffers
  await agent.dispose();
}
```

---

## Decision Types

| Type         | Target Problem                | `criteria` Schema                                                    | Output Structure                                                                              |
| :----------- | :---------------------------- | :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| **`choice`** | Categorical classification    | String array `["A", "B"]` or dictionary `{ [label]: "description" }` | Selected label (`choice`), per-class distribution (`probabilities`), and `confidence`         |
| **`score`**  | Ordinal grading on a rubric   | Ordered array of level descriptions `["low", "medium", "high"]`      | Expected numeric value (`score`), level mapping (`legend`), `probabilities`, and `confidence` |
| **`noul`**   | Binary statement verification | Optional `{ false?: "description", true?: "description" }`           | Probability of truth (`noul` as $P(\text{true})$) and decision `confidence`                   |

---

## Confidence Calibration

Each answer includes a normalized `confidence` value in the range $[0.0, 1.0]$. This value is calculated from the Shannon entropy of the calibrated probability distribution:

$$H(P) = -\sum_{i=1}^{K} P(i) \ln P(i)$$

$$\text{confidence} = \max\left(0, 1 - \frac{H(P)}{\ln K}\right)$$

### Interpretation by Question Type

- **`choice` and `score` ($K \ge 2$)**: Evaluated over option count $K$. Yields $1.0$ when probability concentrates entirely on a single choice, and scales down to $0.0$ when probabilities are uniformly dispersed across all candidates.
- **`noul` (Binary)**: Evaluated directly as the classification margin $\max(P(\text{true}), P(\text{false}))$, spanning $[0.5, 1.0]$. A score of $0.5$ represents maximum ambiguity, whereas $1.0$ indicates total certainty.

---

## Model Assets & Hosting

A standard `@r4ai/laya-web` deployment requires serving the following files under your `modelUrl` directory:

| File                 | Size    | Description                                                                |
| :------------------- | :------ | :------------------------------------------------------------------------- |
| `config.json`        | ~1 KB   | Model dimensions, token budgets, temperature calibration, and file digests |
| `model.onnx`         | ~2.5 MB | ONNX computation graph structure (without weights)                         |
| `model.onnx.data`    | ~684 MB | Partitioned model weights loaded on demand by ONNX Runtime Web             |
| `embeddings.f16.bin` | ~248 MB | Raw FP16 token embedding table read directly by CPU memory                 |
| `tokenizer/`         | ~1.6 MB | Hugging Face tokenizer configuration and vocabulary files                  |

Generate these files locally using:

```sh
pnpm install --frozen-lockfile
pnpm model:export
```

---

## Local Development & Demo

To launch the SolidJS demo application locally:

```sh
pnpm install --frozen-lockfile
pnpm model:export
pnpm dev
```

Open `http://127.0.0.1:5173/` in your browser.

---

## Documentation

- [Development Guide](docs/development.md): Environment setup, model export pipeline, build commands, and CI/CD workflows.
- [Validation Specification](docs/validation.md): Multi-tiered verification strategy, numerical parity benchmarks, and hardware troubleshooting.

---

## License & Attribution

Distributed under the [Apache-2.0 License](LICENSE).

- **Upstream Project**: [Laya-MLX](https://github.com/mizorewww/laya-mlx)
- **Base Checkpoint**: [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
- **Inference Runtime**: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

For copyright notices and attribution statements, see [NOTICE](NOTICE).
