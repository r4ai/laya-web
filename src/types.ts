/**
 * Any value that survives a JSON round trip.
 *
 * @remarks
 * Everything the model reads — a {@link State}, a criterion description, a
 * score legend entry — must be JSON. Non-string values are serialized with the
 * same spacing as Python's `json.dumps(..., ensure_ascii=False)` so that a
 * browser prediction matches the upstream Laya reference implementation.
 * `NaN` and `Infinity` have no JSON form and are rejected at prediction time.
 */
export type Json =
  | null
  | boolean
  | number
  | string
  | Json[]
  | { [key: string]: Json };

/**
 * The input a decision is made about: the ticket, message, or record to judge.
 *
 * @remarks
 * A string is tokenized as written. An object or array is serialized to JSON
 * first, so its keys become part of what the model reads — prefer descriptive
 * field names over abbreviations.
 *
 * The state is appended after the question and truncated from the end once it
 * exceeds the model's token budget, so put the decisive information first.
 *
 * @example
 * ```ts
 * const asText: State = "I was charged twice this month. Please refund me.";
 * const asRecord: State = { subject: "Double charge", plan: "pro", seats: 12 };
 * ```
 */
export type State = string | { [key: string]: Json } | Json[];

/**
 * One typed decision to evaluate against a {@link State}.
 *
 * @remarks
 * `type` selects the decision head, and therefore the shape of both `criteria`
 * and the matching {@link Answer} variant. `instructions` says what to decide;
 * it is not a prompt template, and the model never generates text from it.
 *
 * Instructions and criteria share a fixed token budget. Criteria are truncated
 * first and instructions second, so keep option descriptions short. A question
 * with more options than the budget can hold is rejected outright rather than
 * silently losing options.
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
 * @see {@link Answer} for the result each variant produces.
 */
export type Question =
  | {
      /** Pick the single best label out of discrete candidates. */
      type: "choice";
      /** What to decide, in plain language. */
      instructions: string;
      /**
       * The candidate labels, as unique nonempty strings or as a
       * `label -> description` map. Descriptions are rendered to the model as
       * `"label: description"`, which helps when labels alone are terse.
       */
      criteria: string[] | Record<string, Json>;
    }
  | {
      /** Grade the state against an ordered rubric. */
      type: "score";
      /** What to grade, in plain language. */
      instructions: string;
      /**
       * The rubric levels in ascending order, lowest first, and at least one.
       * Index positions form the scale, so three levels score within `[0, 2]`.
       */
      criteria: Json[];
    }
  | {
      /** Estimate the probability that a statement holds. */
      type: "noul";
      /** The statement to verify, phrased so that "true" is unambiguous. */
      instructions: string;
      /**
       * Optional descriptions of what each outcome means. Missing entries fall
       * back to generic "the statement holds" / "does not hold" wording.
       */
      criteria?: { false?: Json; true?: Json };
    };

/**
 * A batch of questions, keyed by ids you choose.
 *
 * @remarks
 * The same ids come back as the keys of {@link Prediction.answers}, so pick
 * names that are stable in your code (`"department"`, `"urgency"`) rather than
 * positional indices. Every question in a batch is evaluated against the one
 * {@link State} passed alongside it.
 */
export type Questions = Record<string, Question>;

/**
 * The ONNX Runtime Web execution provider actually running the model.
 *
 * @remarks
 * `"webgpu"` runs on the GPU. `"wasm"` runs single-threaded SIMD WebAssembly
 * on the CPU: available everywhere, but markedly slower. Which one a load
 * settled on is readable from `Agent.backend`.
 */
export type Backend = "webgpu" | "wasm";

/** The fields every {@link Answer} carries, whatever the question type. */
export interface AnswerBase {
  /**
   * How concentrated the decision is, in `[0, 1]`, rounded to four decimals.
   *
   * @remarks
   * For `choice` and `score` this is Shannon entropy normalized by the option
   * count: `1` when the probability mass sits on one option, approaching `0`
   * as it spreads out evenly, and a fixed `1` for a single-option question.
   * For `noul` it is the binary margin `max(p, 1 - p)`, so it spans `[0.5, 1]`.
   *
   * Confidence describes the sharpness of the distribution, not correctness —
   * a confidently wrong answer still scores near `1`. Use it to route
   * uncertain cases to a human, not as an accuracy estimate.
   */
  confidence: number;
  /**
   * The checkpoint's reinforcement-learning action head.
   *
   * @remarks
   * `act_probability` is the softmax probability of the head's first class,
   * in `[0, 1]`, rounded to four decimals. It is carried through for parity
   * with the upstream reference implementation; this library never acts on it.
   */
  action: { act_probability: number };
}

/**
 * The result for one {@link Question}, discriminated by `type`.
 *
 * @remarks
 * `type` always mirrors the question that produced it, so narrowing on it is
 * safe and is the intended way to read an answer.
 *
 * @example
 * ```ts
 * const answer = prediction.answers.department;
 * if (answer.type === "choice") console.log(answer.choice, answer.confidence);
 * ```
 *
 * @see {@link AnswerBase} for `confidence` and `action`, shared by all variants.
 */
export type Answer = AnswerBase &
  (
    | {
        type: "choice";
        /** The highest-probability label, taken verbatim from `criteria`. */
        choice: string;
        /** Calibrated probability per label; sums to `1` before rounding. */
        probabilities: Record<string, number>;
      }
    | {
        type: "score";
        /**
         * The probability-weighted mean level, so a fractional value between
         * `0` and `criteria.length - 1` rather than a single chosen level.
         */
        score: number;
        /** Maps each level index, as a string, back to its rubric criterion. */
        legend: Record<string, Json>;
        /** Calibrated probability per level index, keyed as `"0"`, `"1"`, … */
        probabilities: Record<string, number>;
      }
    | {
        type: "noul";
        /** Calibrated probability that the statement holds, in `[0, 1]`. */
        noul: number;
      }
  );

/** Everything one `Agent.predict` call returns. */
export interface Prediction {
  /** Identifies the decision model that produced these answers. */
  model: "laya-rl-agent";
  /** One {@link Answer} per question, under the ids you supplied. */
  answers: Record<string, Answer>;
  /**
   * Tokens consumed by the batch.
   *
   * @remarks
   * `input_tokens` sums the encoded length of every question in the batch,
   * including the shared state repeated per question. `output_tokens` is
   * always `0`: a typed decision reads logits and generates no text.
   */
  usage: { input_tokens: number; output_tokens: 0 };
}

/** Options for `load`. */
export interface LoadOptions {
  /** Directory produced by the exporter, served over HTTP(S). */
  modelUrl: string;
  /**
   * Which execution provider to use.
   *
   * @remarks
   * Auto tries WebGPU initialization, then creates a WASM session if it fails,
   * reporting a `"fallback"` progress event on the way. Passing `"webgpu"`
   * explicitly turns that failure into a rejection instead.
   *
   * @defaultValue `"auto"`
   */
  backend?: Backend | "auto";
  /**
   * Directory containing ONNX Runtime's matching .wasm and .mjs files.
   *
   * @remarks
   * Needed when those binaries are not resolvable from the page's own origin,
   * which is the usual case for a bundled app. ONNX Runtime keeps this setting
   * process-wide, so the last load to set it wins.
   */
  wasmPaths?: string;
  /**
   * Cancels the download and session creation.
   *
   * @remarks
   * Aborting rejects the `load` promise with the signal's reason and
   * releases any session that had already been created, so no cleanup is left
   * to the caller.
   */
  signal?: AbortSignal;
  /**
   * Called as assets download and the session initializes.
   *
   * @remarks
   * Every file in the manifest is announced with `loaded: 0` before any bytes
   * arrive, so a progress bar can show a stable total from the first event.
   * Throwing from this callback fails the load.
   */
  onProgress?: (event: LoadProgress) => void;
}

/**
 * A single step reported to {@link LoadOptions.onProgress}.
 *
 * @remarks
 * Which fields are populated depends on `phase`, so treat the optional ones as
 * genuinely absent rather than assuming they are always present.
 */
export interface LoadProgress {
  /**
   * `"download"` while fetching model assets, `"initialize"` while ONNX
   * Runtime builds the session, and `"fallback"` once when WebGPU failed and
   * WASM is being used instead.
   */
  phase: "download" | "initialize" | "fallback";
  /** The asset being downloaded, relative to `modelUrl`. Download phase only. */
  file?: string;
  /** Bytes received for `file` so far. Download phase only. */
  loaded?: number;
  /**
   * Expected size of `file`, from the model manifest or the `Content-Length`
   * header. Absent when the server reports neither.
   */
  total?: number;
  /** Human-readable detail, currently the reason WebGPU was abandoned. */
  message?: string;
}

/**
 * The tokenizer surface the runtime needs, reduced to the special token ids
 * and an encode call.
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
 * The validated contents of the exporter's `config.json`.
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
 * One question encoded into the tensors the graph consumes, plus the labels
 * needed to turn its logits back into an {@link Answer}.
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
