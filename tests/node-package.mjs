import assert from "node:assert/strict";
import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { load } from "@r4ai/laya-web";

const model = new URL("./fixtures/node-model/", import.meta.url);
const questions = {
  choice: { type: "choice", instructions: "Choose", criteria: ["a", "b"] },
  score: { type: "score", instructions: "Grade", criteria: ["low", "high"] },
  noul: { type: "noul", instructions: "True?" },
  single: { type: "choice", instructions: "Choose", criteria: ["a"] },
};

async function verify(modelUrl, backend) {
  const phases = [];
  const agent = await load({
    modelUrl,
    backend,
    onProgress: (e) => phases.push(e.phase),
  });
  try {
    assert.equal(agent.backend, "wasm");
    assert(!phases.includes("fallback"));
    const result = await agent.predict("sample", questions);
    assert.equal(result.answers.choice.choice, "a");
    assert.deepEqual(result.answers.choice.probabilities, { a: 0.5, b: 0.5 });
    assert.equal(result.answers.score.score, 0.5);
    assert.equal(result.answers.noul.noul, 0.5);
    assert.equal(result.answers.single.confidence, 1);
    assert.equal(result.answers.choice.action.act_probability, 0.5);
    assert.deepEqual(await agent.predict("sample", questions), result);
  } finally {
    await agent.dispose();
  }
  await agent.dispose();
  await assert.rejects(agent.predict("sample", questions), /disposed/);
}

test("published Node entry selects WASM and loads paths and file URLs", async () => {
  for (const [location, backend] of [
    [fileURLToPath(model), undefined],
    [model.href, "auto"],
    ["tests/fixtures/node-model", "wasm"],
  ]) {
    await verify(location, backend);
  }
});

test("HTTP model assets use the same Node inference lifecycle", async () => {
  const server = createServer((req, res) => {
    const stream = createReadStream(new URL(`.${req.url}`, model));
    stream.on("error", () => res.writeHead(404).end());
    stream.pipe(res);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    await verify(`http://127.0.0.1:${server.address().port}`, "auto");
  } finally {
    await new Promise((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
});

test("invalid options and cancellation reject before initialization", async () => {
  await assert.rejects(load({ modelUrl: "" }), /modelUrl/);
  await assert.rejects(
    load({ modelUrl: model.href, backend: "webgpu" }),
    /Node.js supports/,
  );
  await assert.rejects(
    load({ modelUrl: model.href, backend: "cpu" }),
    /Unknown backend/,
  );
  await assert.rejects(
    load({ modelUrl: model.href, signal: AbortSignal.abort() }),
    { name: "AbortError" },
  );
});
