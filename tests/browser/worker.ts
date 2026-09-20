import { load } from "@r4ai/laya-web";
import { createTokenizer } from "../../src/assets.js";
import { prepare } from "../../src/core.js";
import type { ModelConfig, Questions, State } from "../../src/types.js";
interface Fixture {
  state: State;
  questions: Questions;
  result: unknown;
  rows: { ids: number[]; markers: number[] }[];
}
onmessage = async ({ data }) => {
  let agent;
  try {
    const modelUrl = `${data.base}/models/laya/`;
    const json = async (name: string) => {
      const response = await fetch(modelUrl + name);
      if (!response.ok) throw new Error(`${name}: ${response.status}`);
      return response.json();
    };
    const cfg: ModelConfig = await json("config.json");
    const parity: { fixtures: Fixture[] } = await json("parity.json");
    const tok = createTokenizer(
      await json("tokenizer/tokenizer.json"),
      await json("tokenizer/tokenizer_config.json"),
    );
    let questions = 0;
    for (const fixture of parity.fixtures) {
      for (const [i, q] of Object.values(fixture.questions).entries()) {
        const p = prepare(fixture.state, q, tok, cfg);
        if (
          JSON.stringify(p.ids) !== JSON.stringify(fixture.rows[i].ids) ||
          JSON.stringify(p.markers) !== JSON.stringify(fixture.rows[i].markers)
        )
          throw new Error(`Tokenization parity failed: ${questions}`);
        questions++;
      }
    }
    agent = await load({
      modelUrl,
      backend: data.backend,
      wasmPaths: `${data.base}/ort/`,
      onProgress: (p) => postMessage(p),
    });
    let maxError = 0;
    function compare(actual: unknown, expected: unknown, path = "") {
      if (typeof expected === "number") {
        if (typeof actual !== "number" || !Number.isFinite(actual))
          throw new Error(`Nonfinite output at ${path}`);
        const error = Math.abs(actual - expected);
        maxError = Math.max(maxError, error);
        if (error > 0.000101)
          throw new Error(`Parity error at ${path}: ${actual} vs ${expected}`);
      } else if (expected !== null && typeof expected === "object") {
        for (const [key, value] of Object.entries(expected))
          compare(
            (actual as Record<string, unknown>)[key],
            value,
            `${path}.${key}`,
          );
      } else if (actual !== expected)
        throw new Error(`Mismatch at ${path}: ${actual} vs ${expected}`);
    }
    const start = performance.now();
    for (const [i, fixture] of parity.fixtures.entries()) {
      postMessage({
        phase: "inference",
        backend: agent.backend,
        fixture: i + 1,
        total: parity.fixtures.length,
      });
      compare(
        await agent.predict(fixture.state, fixture.questions),
        fixture.result,
      );
    }
    const sample = parity.fixtures[0];
    const first = await agent.predict(sample.state, sample.questions);
    const second = await agent.predict(sample.state, sample.questions);
    compare(second, first);
    const backend = agent.backend;
    await agent.dispose();
    let disposed = false;
    try {
      await agent.predict("", {});
    } catch {
      disposed = true;
    }
    if (!disposed) throw new Error("Use after dispose was accepted");
    postMessage({
      complete: true,
      passed: true,
      backend,
      questions,
      tokenParity: "exact",
      maxRoundedOutputError: maxError,
      repeatAndDispose: "passed",
      elapsedMs: Math.round(performance.now() - start),
    });
  } catch (error) {
    postMessage({ complete: true, passed: false, error: String(error) });
  } finally {
    await agent?.dispose();
  }
};
