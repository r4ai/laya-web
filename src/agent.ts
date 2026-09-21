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

/** The small inference boundary keeps model execution separate from the Laya protocol. */
export interface Driver {
  readonly backend: Backend;
  run(
    question: PreparedQuestion,
  ): Promise<{ logits: number[]; action: number[] }>;
  dispose(): Promise<void>;
}

export class Agent {
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

  /** Questions run sequentially to bound browser memory. Calls are serialized per agent. */
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

  /** Reject new work, finish accepted requests, then release model resources. Idempotent. */
  dispose(): Promise<void> {
    this.closing ??= this.tail.then(() => this.driver.dispose());
    return this.closing;
  }
}
