/**
 * JSON-serializable value.
 *
 * @remarks
 * Inputs such as {@link State} and question criteria must conform to this type.
 * Values are serialized to match Python `json.dumps(..., ensure_ascii=False)`.
 * Non-finite numbers (`NaN`, `Infinity`) are rejected at validation time.
 */
export type Json =
  | null
  | boolean
  | number
  | string
  | Json[]
  | { [key: string]: Json };

/**
 * Input content evaluated by decision questions.
 *
 * @remarks
 * Representation and truncation rules:
 * - String: Tokenized directly as plain text
 * - Object or array: Serialized to deterministic JSON before tokenization
 * - Budget overflow: Truncated from the end when exceeding maximum token length
 *
 * @example
 * ```ts
 * const asText: State = "I was charged twice this month. Please refund me.";
 * const asRecord: State = { subject: "Double charge", plan: "pro", seats: 12 };
 * ```
 */
export type State = string | { [key: string]: Json } | Json[];

/**
 * Typed decision specification evaluated against a {@link State}.
 *
 * @remarks
 * Evaluation rules:
 * - Discriminant `type` determines criteria format and resulting {@link Answer} variant
 * - Criteria and instructions share a fixed token budget
 * - Truncation priority: criteria first, then instructions
 * - Questions exceeding budget limit throw `RangeError` rather than dropping options
 *
 * @example
 * ```ts
 * const department: Question = {
 *   type: "choice",
 *   instructions: "Which support department should handle this request?",
 *   criteria: { billing: "refunds and invoices", tech: "bugs and outages" },
 * };
 *
 * const urgency: Question = {
 *   type: "score",
 *   instructions: "How urgent is this ticket?",
 *   criteria: ["Normal", "Elevated", "Immediate"],
 * };
 *
 * const refund: Question = {
 *   type: "noul",
 *   instructions: "Does the user explicitly demand a refund?",
 * };
 * ```
 *
 * @see {@link Answer} for matching output variants
 */
export type Question =
  | {
      /** Discrete single-label classification */
      type: "choice";
      /** Decision goal in plain language */
      instructions: string;
      /**
       * Candidate labels or label-to-description mapping
       *
       * @remarks
       * Rendered as `"label: description"` when defined as an object.
       */
      criteria: string[] | Record<string, Json>;
    }
  | {
      /** Ordinal grading against an ordered scale */
      type: "score";
      /** Rubric goal in plain language */
      instructions: string;
      /**
       * Ordered rubric levels in ascending order (minimum 1 level)
       *
       * @remarks
       * Indices define the output score range `[0, criteria.length - 1]`.
       */
      criteria: Json[];
    }
  | {
      /** Binary verification probability */
      type: "noul";
      /** Statement to verify as true or false */
      instructions: string;
      /** Optional descriptions for true and false outcomes */
      criteria?: { false?: Json; true?: Json };
    };

/**
 * Map of questions keyed by caller-defined identifiers.
 *
 * @remarks
 * Keys match identifiers returned in {@link Prediction.answers}.
 * All questions in a batch evaluate against the same {@link State}.
 */
export type Questions = Record<string, Question>;

/**
 * ONNX Runtime Web execution backend.
 *
 * @remarks
 * Supported providers:
 * - `"webgpu"`: Hardware-accelerated GPU execution
 * - `"wasm"`: Single-threaded WebAssembly SIMD CPU execution
 */
export type Backend = "webgpu" | "wasm";

/** Common metadata included in every {@link Answer}. */
export interface AnswerBase {
  /**
   * Distribution sharpness score in range `[0, 1]`, rounded to 4 decimals.
   *
   * @remarks
   * Metric computation:
   * - `choice` / `score`: Normalized Shannon entropy (`1` = concentrated on single option, `0` = uniform distribution)
   * - `noul`: Binary margin `max(p, 1 - p)` in range `[0.5, 1]`
   *
   * Measures probability concentration rather than ground-truth correctness.
   */
  confidence: number;
  /**
   * Reinforcement learning action head output.
   *
   * @remarks
   * `act_probability` represents the first-class softmax probability in `[0, 1]`.
   * Retained for compatibility with reference implementations.
   */
  action: { act_probability: number };
}

/**
 * Result of an evaluated question, discriminated by `type`.
 *
 * @remarks
 * Discriminant `type` matches the corresponding {@link Question.type}.
 *
 * @example
 * ```ts
 * const answer = prediction.answers.department;
 * if (answer.type === "choice") {
 *   console.log(answer.choice, answer.confidence);
 * }
 * ```
 *
 * @see {@link AnswerBase} for shared base properties
 */
export type Answer = AnswerBase &
  (
    | {
        type: "choice";
        /** Highest-probability candidate label from `criteria` */
        choice: string;
        /** Calibrated probabilities per label (normalized to sum to 1.0) */
        probabilities: Record<string, number>;
      }
    | {
        type: "score";
        /**
         * Expected score calculated as probability-weighted mean level.
         * Fractional value in range `[0, criteria.length - 1]`.
         */
        score: number;
        /** Map of string level indices to original rubric criteria */
        legend: Record<string, Json>;
        /** Calibrated probabilities keyed by level index */
        probabilities: Record<string, number>;
      }
    | {
        type: "noul";
        /** Calibrated probability that the statement holds in range `[0, 1]` */
        noul: number;
      }
  );

/** Result returned by {@link Agent.predict}. */
export interface Prediction {
  /** Model identifier */
  model: "laya-rl-agent";
  /** Evaluation results mapped by question identifier */
  answers: Record<string, Answer>;
  /**
   * Token usage metrics for the evaluated batch.
   *
   * @remarks
   * - `input_tokens`: Total tokens across all questions (state tokens repeated per question)
   * - `output_tokens`: Always `0` (classification outputs produce no text tokens)
   */
  usage: { input_tokens: number; output_tokens: 0 };
}

/** Configuration options for {@link load}. */
export interface LoadOptions {
  /** Base URL of exported model directory served over HTTP(S) */
  modelUrl: string;
  /**
   * Preferred execution backend.
   *
   * @remarks
   * Provider resolution:
   * - `"auto"`: Attempts WebGPU; falls back to WASM on failure with a `"fallback"` event
   * - `"webgpu"`: Requires WebGPU; rejects if unavailable
   * - `"wasm"`: Uses WebAssembly directly
   *
   * @defaultValue `"auto"`
   */
  backend?: Backend | "auto";
  /**
   * Directory containing ONNX Runtime `.wasm` and `.mjs` assets.
   *
   * @remarks
   * Required when WASM binaries cannot be resolved from host origin.
   * Applies globally to `ort.env.wasm.wasmPaths`.
   */
  wasmPaths?: string;
  /**
   * Signal to abort asset downloads and session creation.
   *
   * @remarks
   * Releases intermediate session allocations before rejecting.
   */
  signal?: AbortSignal;
  /**
   * Progress callback invoked during asset download and session setup.
   *
   * @remarks
   * Fires initial `loaded: 0` event for all manifest assets before transfer.
   */
  onProgress?: (event: LoadProgress) => void;
}

/** Progress notification emitted to {@link LoadOptions.onProgress}. */
export interface LoadProgress {
  /**
   * Current loading phase:
   * - `"download"`: Transferring model asset files
   * - `"initialize"`: Constructing ONNX Runtime session
   * - `"fallback"`: Switching from WebGPU to WASM fallback
   */
  phase: "download" | "initialize" | "fallback";
  /** Relative asset file path (download phase only) */
  file?: string;
  /** Transferred byte count (download phase only) */
  loaded?: number;
  /** Total expected byte count from manifest or Content-Length (download phase only) */
  total?: number;
  /** Informational diagnostic message (e.g. fallback reason) */
  message?: string;
}

/**
 * Minimal tokenizer interface required by runtime.
 *
 * @internal
 */
export interface Tokenizer {
  cls: number;
  sep: number;
  mask: number;
  maskToken: string;
  encode(text: string): number[];
}

/**
 * Validated schema for model `config.json`.
 *
 * @internal
 */
export interface ModelConfig {
  format: "laya-web-v1";
  maxLength: number;
  headMaxLength: number;
  hiddenSize: number;
  vocabSize: number;
  temperature: [number, number, number];
  temperatureByOptions: Record<string, number>;
  files?: Record<string, { bytes: number; sha256: string }>;
}

/**
 * Encoded tensor payload and metadata for model inference.
 *
 * @internal
 */
export interface PreparedQuestion {
  ids: number[];
  markers: number[];
  qtype: number;
  question: Question;
  labels: string[];
}
