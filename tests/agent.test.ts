import { expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import type { Driver } from "../src/agent.js";
import type { ModelConfig, Tokenizer } from "../src/types.js";
const cfg: ModelConfig = {
  format: "laya-web-v1",
  maxLength: 64,
  headMaxLength: 24,
  hiddenSize: 8,
  vocabSize: 256,
  temperature: [1, 1, 1],
  temperatureByOptions: {},
};
const tok: Tokenizer = {
  cls: 1,
  sep: 2,
  mask: 3,
  maskToken: "<mask>",
  encode: (s) => Array.from(s, (c) => c.codePointAt(0)!),
};
const questions = { a: { type: "noul" as const, instructions: "" } };
class FakeDriver implements Driver {
  backend = "wasm" as const;
  events: string[] = [];
  fail = false;
  async run() {
    this.events.push("run");
    if (this.fail) throw new Error("inference failed");
    return { logits: [0, 1], action: [0, 0] };
  }
  async dispose() {
    this.events.push("dispose");
  }
}
it("serializes work, drains before dispose, and rejects use after dispose", async () => {
  const driver = new FakeDriver();
  const agent = new Agent(cfg, tok, driver);
  const a = agent.predict("a", questions);
  const b = agent.predict("b", questions);
  const close = agent.dispose();
  await expect(a).resolves.toMatchObject({ usage: { output_tokens: 0 } });
  await b;
  await close;
  await agent.dispose();
  expect(driver.events).toEqual(["run", "run", "dispose"]);
  await expect(agent.predict("", questions)).rejects.toThrow(/disposed/);
});
it("a failed request does not poison subsequent requests", async () => {
  const driver = new FakeDriver();
  const agent = new Agent(cfg, tok, driver);
  driver.fail = true;
  await expect(agent.predict("", questions)).rejects.toThrow(
    "inference failed",
  );
  driver.fail = false;
  await expect(agent.predict("", questions)).resolves.toHaveProperty(
    "answers.a.noul",
  );
});
it("validates all questions before running and handles empty requests", async () => {
  const driver = new FakeDriver();
  const agent = new Agent(cfg, tok, driver);
  await expect(
    agent.predict("", {
      ...questions,
      b: { type: "choice", instructions: "", criteria: [] },
    }),
  ).rejects.toThrow();
  expect(driver.events).toEqual([]);
  expect((await agent.predict("", {})).usage.input_tokens).toBe(0);
});
it("aborted queued work does not run", async () => {
  const driver = new FakeDriver();
  const agent = new Agent(cfg, tok, driver);
  const controller = new AbortController();
  controller.abort();
  await expect(
    agent.predict("", questions, { signal: controller.signal }),
  ).rejects.toThrow();
  expect(driver.events).toEqual([]);
});
