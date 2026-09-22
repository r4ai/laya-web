/**
 * Browser and Node.js inference engine for Laya typed-decision models using ONNX Runtime Web.
 *
 * @remarks
 * Evaluates structured decisions over input states without text generation:
 * - `choice`: Selects the single best label from discrete candidates
 * - `score`: Grades input against an ordered numeric rubric
 * - `noul`: Estimates the probability that a statement holds
 *
 * Call {@link load} to initialize an {@link Agent} from a model checkpoint.
 * Call {@link Agent.predict} to evaluate questions against an input {@link State}.
 * Call {@link Agent.dispose} when finished to release runtime resources.
 *
 * @example
 * ```ts
 * import { load } from "@r4ai/laya-web";
 *
 * const agent = await load({ modelUrl: "/models/laya/", wasmPaths: "/ort/" });
 * try {
 *   const { answers } = await agent.predict("Payment failed twice.", {
 *     department: {
 *       type: "choice",
 *       instructions: "Which support department should handle this request?",
 *       criteria: ["Billing", "Technical Support", "Sales"],
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

export { load } from "./browser.js";
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
