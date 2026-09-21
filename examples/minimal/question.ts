import type { Question } from "@r4ai/laya-web";
import type { Request } from "./protocol.js";

/**
 * Raw form input, and the values the demo starts from. Every decision type keeps
 * its own draft, so switching the type preserves what was typed for the others.
 */
export const defaultDraft = {
  state:
    "I was charged twice for my subscription this month. Please issue a refund.",
  type: "choice" as Question["type"],
  instructions: "Which support department should handle this request?",
  choices: "Billing & Refunds\nTechnical Support\nSales",
  scale: "Normal\nElevated\nImmediate",
  falseCriterion: "",
  trueCriterion: "",
  backend: "auto" as Request["backend"],
};
export type Draft = typeof defaultDraft;

const lines = (text: string) =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

/**
 * Validates the draft at the form boundary.
 *
 * @returns The question to send, or an `Error` describing what the user must fix.
 */
export function toQuestion(draft: Draft): Question | Error {
  const instructions = draft.instructions;
  if (draft.type === "noul") {
    const criteria = Object.fromEntries(
      Object.entries({
        false: draft.falseCriterion.trim(),
        true: draft.trueCriterion.trim(),
      }).filter(([, value]) => value !== ""),
    );
    return Object.keys(criteria).length
      ? { type: "noul", instructions, criteria }
      : { type: "noul", instructions };
  }
  const criteria = lines(draft.type === "choice" ? draft.choices : draft.scale);
  if (draft.type === "score")
    return criteria.length
      ? { type: "score", instructions, criteria }
      : new Error("Enter at least one evaluation scale level");
  return criteria.length && new Set(criteria).size === criteria.length
    ? { type: "choice", instructions, criteria }
    : new Error("Choices must be unique");
}
