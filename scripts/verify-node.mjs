import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { load } from "@r4ai/laya-web";

const modelUrl = resolve(
  process.argv[2] ?? "examples/minimal/public/models/laya",
);
const parity = JSON.parse(
  await readFile(resolve(modelUrl, "parity.json"), "utf8"),
);
let maxError = 0;
function compare(actual, expected, path = "result") {
  if (typeof expected === "number") {
    assert(Number.isFinite(actual), `${path}: nonfinite output`);
    const error = Math.abs(actual - expected);
    maxError = Math.max(maxError, error);
    assert(error <= 0.000101, `${path}: ${actual} vs ${expected}`);
  } else if (expected !== null && typeof expected === "object") {
    assert.deepEqual(
      Object.keys(actual).sort(),
      Object.keys(expected).sort(),
      path,
    );
    for (const [key, value] of Object.entries(expected))
      compare(actual[key], value, `${path}.${key}`);
  } else {
    assert.equal(actual, expected, path);
  }
}
const agent = await load({ modelUrl });
try {
  for (const fixture of parity.fixtures) {
    compare(
      await agent.predict(fixture.state, fixture.questions),
      fixture.result,
    );
  }
  console.log(
    JSON.stringify({
      backend: agent.backend,
      questions: parity.questions,
      maxRoundedOutputError: maxError,
    }),
  );
} finally {
  await agent.dispose();
}
