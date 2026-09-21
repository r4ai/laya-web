import { formatAnswer, prepare } from "./core.js";
import type {
  Backend,
  ModelConfig,
  Prediction,
  PreparedQuestion,
  Questions,
  State,
  Tokenizer,
} from "./types.js";

/**
 * Backend driver abstraction for tensor execution.
 *
 * @remarks
 * Separates inference execution from tokenization, calibration, and formatting.
 *
 * @internal
 */
export interface Driver {
  /** Execution provider used by the driver */
  readonly backend: Backend;
  /** Evaluates an encoded question and returns raw decision and action logits */
  run(
    question: PreparedQuestion,
  ): Promise<{ logits: number[]; action: number[] }>;
  /** Releases session resources and allocated weights */
  dispose(): Promise<void>;
}

/**
 * Loaded Laya model instance for local typed-decision inference.
 *
 * @remarks
 * Created via {@link load}. Retains active inference session and weights:
 * - Load once and reuse across the application
 * - Execute inside a Web Worker to avoid blocking UI threads
 * - Stateless across {@link Agent.predict} calls
 * - Release resources via {@link Agent.dispose} when finished
 *
 * @example
 * ```ts
 * const agent = await load({ modelUrl: "/models/laya/" });
 * try {
 *   const { answers } = await agent.predict("Refund me, I was charged twice.", {
 *     refund: { type: "noul", instructions: "Does the user demand a refund?" },
 *   });
 *   if (answers.refund.type === "noul") {
 *     console.log(answers.refund.noul);
 *   }
 * } finally {
 *   await agent.dispose();
 * }
 * ```
 */
export class Agent {
  /**
   * Active execution provider resolved during initialization.
   *
   * @remarks
   * Fixed after load. Value is `"webgpu"` or `"wasm"`.
   */
  readonly backend: Backend;
  private tail: Promise<unknown> = Promise.resolve();
  private closing?: Promise<void>;
  constructor(
    private readonly config: ModelConfig,
    private readonly tokenizer: Tokenizer,
    private readonly driver: Driver,
  ) {
    this.backend = driver.backend;
  }

  /**
   * Evaluates a batch of typed decisions against a shared input state.
   *
   * @remarks
   * Execution behavior:
   * - Sequential question evaluation within a batch to bound memory
   * - Serialized batch calls per agent instance (concurrent calls queue)
   * - Deep-cloned arguments via `structuredClone` on entry
   * - Full batch validation and encoding before device execution
   *
   * @param state - Target input evaluated by all questions in the batch
   * @param questions - Questions mapped by caller-defined identifier
   * @param options - Optional settings for inference execution
   *   - `signal`: AbortSignal checked before and between questions
   * @returns Batch predictions mapped by question identifier, plus token usage
   * @throws TypeError - Invalid input state or malformed question definition
   * @throws RangeError - Question option count exceeds token budget
   * @throws Error - Agent disposed, or operation aborted via `signal`
   *
   * @example
   * ```ts
   * const { answers, usage } = await agent.predict(
   *   ticket,
   *   {
   *     department: {
   *       type: "choice",
   *       instructions: "Which team should handle this?",
   *       criteria: ["Billing", "Technical", "Sales"],
   *     },
   *     urgency: {
   *       type: "score",
   *       instructions: "How urgent is this ticket?",
   *       criteria: ["Normal", "Elevated", "Immediate"],
   *     },
   *   },
   *   { signal: AbortSignal.timeout(30_000) },
   * );
   *
   * if (answers.department.type === "choice") {
   *   console.log(answers.department.choice, answers.department.confidence);
   * }
   * console.log(`${usage.input_tokens} tokens`);
   * ```
   *
   * @see {@link Question} for decision schema definitions
   */
  predict(
    state: State,
    questions: Questions,
    options: { signal?: AbortSignal } = {},
  ): Promise<Prediction> {
    if (this.closing) return Promise.reject(new Error("Agent is disposed"));
    // Prepare synchronously inside the queued job; snapshot caller-owned values now.
    let snapshot: { state: State; questions: Questions };
    try {
      snapshot = structuredClone({ state, questions });
    } catch (error) {
      return Promise.reject(error);
    }
    const signal = options.signal;
    const task = this.tail.then(async () => {
      signal?.throwIfAborted();
      if (
        !snapshot.questions ||
        Array.isArray(snapshot.questions) ||
        typeof snapshot.questions !== "object"
      )
        throw new TypeError("Questions must be an object");
      const entries = Object.entries(snapshot.questions).map(
        ([id, q]) =>
          [
            id,
            prepare(snapshot.state, q, this.tokenizer, this.config),
          ] as const,
      );
      const answers: Prediction["answers"] = Object.create(null);
      let tokens = 0;
      for (const [id, question] of entries) {
        signal?.throwIfAborted();
        const result = await this.driver.run(question);
        signal?.throwIfAborted();
        answers[id] = formatAnswer(
          question,
          result.logits,
          result.action,
          this.config,
        );
        tokens += question.ids.length;
      }
      return {
        model: "laya-rl-agent" as const,
        answers,
        usage: { input_tokens: tokens, output_tokens: 0 as const },
      };
    });
    this.tail = task.catch(() => undefined);
    return task;
  }

  /**
   * Drains pending requests and releases model resources.
   *
   * @remarks
   * Lifecycle behavior:
   * - Idempotent (subsequent calls return the existing promise)
   * - Queued `predict` calls complete before session release
   * - New `predict` calls reject immediately
   *
   * @returns Promise resolving when session and weights are released
   *
   * @example
   * ```ts
   * try {
   *   await agent.predict(state, questions);
   * } finally {
   *   await agent.dispose();
   * }
   * ```
   */
  dispose(): Promise<void> {
    this.closing ??= this.tail.then(() => this.driver.dispose());
    return this.closing;
  }
}
