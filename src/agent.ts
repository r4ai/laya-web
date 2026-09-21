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
 * The small inference boundary keeps model execution separate from the Laya
 * protocol.
 *
 * @remarks
 * {@link Agent} owns tokenization, calibration, and answer formatting; a driver
 * owns only tensor execution. The runtime ships an ONNX Runtime Web
 * implementation, and tests substitute their own.
 *
 * @internal
 */
export interface Driver {
  /** The execution provider this driver actually obtained. */
  readonly backend: Backend;
  /** Run one encoded question and return its raw decision and action logits. */
  run(
    question: PreparedQuestion,
  ): Promise<{ logits: number[]; action: number[] }>;
  /** Release the session and any retained weights. */
  dispose(): Promise<void>;
}

/**
 * A loaded Laya model, ready to answer typed decisions entirely in the browser.
 *
 * @remarks
 * Instances come from `load`; the constructor is not part of the public
 * API. One agent owns an ONNX session and a few hundred megabytes of weights,
 * so load it once, share it across the app — a dedicated Web Worker is the
 * recommended home — and {@link Agent.dispose | dispose} of it when done.
 *
 * The agent holds no conversation state. Each {@link Agent.predict | predict}
 * call is independent, and nothing from one call influences the next.
 *
 * @example
 * ```ts
 * const agent = await load({ modelUrl: "/models/laya/" });
 * try {
 *   const { answers } = await agent.predict("Refund me, I was charged twice.", {
 *     refund: { type: "noul", instructions: "Does the user demand a refund?" },
 *   });
 *   if (answers.refund.type === "noul") console.log(answers.refund.noul);
 * } finally {
 *   await agent.dispose();
 * }
 * ```
 */
export class Agent {
  /**
   * The execution provider this agent ended up on.
   *
   * @remarks
   * Resolved once at load time and fixed thereafter. Worth reporting in a
   * diagnostics panel: `"wasm"` explains an otherwise puzzling slowdown when
   * WebGPU was unavailable and `LoadOptions.backend` was `"auto"`.
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
   * Answer a batch of typed decisions about one state.
   *
   * @remarks
   * Questions run sequentially to bound browser memory, and calls are
   * serialized per agent: concurrent calls queue rather than overlap, so a
   * second call waits out the first. A rejected call does not poison the queue.
   *
   * `state` and `questions` are deep-cloned on entry, so later mutation of
   * your own objects cannot affect a queued call — and values that
   * `structuredClone` refuses, such as functions or class instances, reject
   * immediately.
   *
   * Every question is validated and encoded before any inference starts, so a
   * malformed question fails the whole batch without wasting GPU work.
   *
   * @param state - The input all questions in this batch are judged against.
   * @param questions - Questions keyed by ids that reappear in the result. An
   * empty object is valid and yields an empty `answers`.
   * @param options - `signal` cancels the call. It is checked when the job
   * starts and between questions, so an abort while queued skips the work
   * entirely, but it does not interrupt an inference already running on the
   * device.
   * @returns The answers, keyed by your ids, plus token usage for the batch.
   * @throws TypeError if `questions` is not a plain object, or if a question or
   * the state is malformed.
   * @throws RangeError if a question has more options than the model's token
   * budget can hold.
   * @throws Error if the agent has been disposed, or `signal`'s abort reason if
   * the call was cancelled.
   *
   * @example
   * ```ts
   * const { answers, usage } = await agent.predict(ticket, {
   *   department: {
   *     type: "choice",
   *     instructions: "Which team should handle this?",
   *     criteria: ["Billing", "Technical", "Sales"],
   *   },
   *   urgency: {
   *     type: "score",
   *     instructions: "How urgent is this ticket?",
   *     criteria: ["Normal", "Elevated", "Immediate"],
   *   },
   * }, { signal: AbortSignal.timeout(30_000) });
   *
   * if (answers.department.type === "choice") {
   *   console.log(answers.department.choice, answers.department.confidence);
   * }
   * console.log(`${usage.input_tokens} tokens`);
   * ```
   *
   * @see `Question` for how to phrase each decision type.
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
   * Reject new work, finish accepted requests, then release model resources.
   * Idempotent.
   *
   * @remarks
   * Requests already queued still run to completion, so pending
   * {@link Agent.predict | predict} promises resolve normally — this drains the
   * agent rather than cancelling it. Use an `AbortSignal` on the individual
   * calls if you need them to stop early.
   *
   * After the first call the agent is permanently closed: further `predict`
   * calls reject with `Agent is disposed`, and repeat `dispose` calls return
   * the same promise instead of releasing twice. Not disposing leaks the ONNX
   * session and its weights for the lifetime of the page or worker.
   *
   * @returns A promise that settles once the session and weights are released.
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
