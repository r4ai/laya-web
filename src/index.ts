/**
 * Client-side inference for Laya typed-decision models, running entirely in
 * the browser on ONNX Runtime Web.
 *
 * @remarks
 * Laya evaluates structured decisions over input state without generating
 * free-form text. Three decision types are available, selected per question:
 * `choice` picks a label out of discrete candidates, `score` grades against an
 * ordered rubric, and `noul` estimates the probability that a statement holds.
 * Every answer carries a calibrated `confidence` for filtering uncertain
 * cases.
 *
 * Start at {@link load}, which fetches a self-hosted checkpoint and returns an
 * {@link Agent}. The agent answers batches of {@link Questions} about a
 * {@link State} and must be disposed when finished.
 *
 * @example
 * ```ts
 * import { load } from "@r4ai/laya-web";
 *
 * const agent = await load({ modelUrl: "/models/laya/", wasmPaths: "/ort/" });
 * try {
 *   const { answers } = await agent.predict(ticketText, {
 *     department: {
 *       type: "choice",
 *       instructions: "Which support department should handle this request?",
 *       criteria: ["Billing & Refunds", "Technical Support", "Sales"],
 *     },
 *   });
 *   if (answers.department.type === "choice") {
 *     console.log(answers.department.choice, answers.department.confidence);
 *   }
 * } finally {
 *   await agent.dispose();
 * }
 * ```
 *
 * @packageDocumentation
 */

export { load } from "./runtime.js";
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
