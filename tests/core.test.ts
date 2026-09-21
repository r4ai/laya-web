import { describe, expect, it } from "vitest";
import {
  prepare,
  formatAnswer,
  validateConfig,
  halfToFloat,
} from "../src/core.js";
import type { Tokenizer, ModelConfig } from "../src/types.js";
const tok: Tokenizer = {
  cls: 1,
  sep: 2,
  mask: 3,
  maskToken: "<mask>",
  encode: (s) => Array.from(s, (c) => c.codePointAt(0)!),
};
const cfg: ModelConfig = {
  format: "laya-web-v1",
  maxLength: 512,
  headMaxLength: 192,
  hiddenSize: 8,
  vocabSize: 256,
  temperature: [1, 1, 1],
  temperatureByOptions: {},
};

describe("question preparation state table", () => {
  it.each(["choice", "score", "noul"] as const)(
    "places real markers for %s and removes injected mask tokens",
    (type) => {
      const question =
        type === "noul"
          ? { type, instructions: "<mask>" }
          : { type, instructions: "<mask>", criteria: ["one", "two"] };
      const p = prepare("hello<mask>world", question, tok, cfg);
      expect(p.markers).toHaveLength(2);
      expect(p.markers.map((i) => p.ids[i])).toEqual([3, 3]);
      expect(p.ids.filter((id) => id === 3)).toHaveLength(2);
      expect(p.ids[0]).toBe(1);
      expect(p.ids.at(-1)).toBe(2);
    },
  );
  it.each([
    { type: "unknown", instructions: "" },
    { type: "choice", instructions: "", criteria: [] },
    { type: "choice", instructions: "", criteria: ["x", "x"] },
    { type: "score", instructions: "", criteria: {} },
    { type: "noul", instructions: "", criteria: [] },
    { type: "noul" },
  ])("rejects invalid question %j", (question) =>
    expect(() => prepare("", question as never, tok, cfg)).toThrow(),
  );
  it("truncates the state, preserving prefix and final separator", () => {
    const p = prepare(
      "x".repeat(1000),
      { type: "choice", instructions: "", criteria: ["a", "b"] },
      tok,
      cfg,
    );
    expect(p.ids).toHaveLength(512);
    expect(p.ids.at(-1)).toBe(2);
  });
  it("rejects options whose markers cannot fit", () => {
    expect(() =>
      prepare(
        "",
        { type: "score", instructions: "", criteria: Array(200).fill("x") },
        tok,
        cfg,
      ),
    ).toThrow(/options/i);
  });
  it("serializes structured state with upstream JSON spacing", () => {
    const p = prepare(
      { a: [1, true] },
      { type: "noul", instructions: "" },
      tok,
      cfg,
    );
    expect(String.fromCodePoint(...p.ids.slice(-17, -1))).toBe(
      '{"a": [1, true]}',
    );
  });
});
describe("calibrated outputs", () => {
  it("uses option bucket temperature and preserves label mapping", () => {
    const p = prepare(
      "",
      {
        type: "choice",
        instructions: "",
        criteria: { billing: "money", technical: "bugs" },
      },
      tok,
      cfg,
    );
    const a = formatAnswer(p, [0, 2], [0, 0], {
      ...cfg,
      temperatureByOptions: { "choice:2": 2 },
    });
    expect(a).toEqual({
      type: "choice",
      choice: "technical",
      probabilities: { billing: 0.2689, technical: 0.7311 },
      confidence: 0.1601,
      action: { act_probability: 0.5 },
    });
  });
  it("returns expected rubric level for scores", () => {
    const p = prepare(
      "",
      { type: "score", instructions: "", criteria: ["low", "medium", "high"] },
      tok,
      cfg,
    );
    expect(formatAnswer(p, [0, 0, 0], [1], cfg)).toMatchObject({
      score: 1,
      confidence: 0,
      legend: { 0: "low", 1: "medium", 2: "high" },
    });
  });
  it("returns P(true) and binary confidence for noul", () => {
    const p = prepare("", { type: "noul", instructions: "" }, tok, cfg);
    expect(formatAnswer(p, [0, 0], [0, 0], cfg)).toMatchObject({
      noul: 0.5,
      confidence: 0.5,
    });
  });
  it("handles one option and large logits", () => {
    const p = prepare(
      "",
      { type: "choice", instructions: "", criteria: ["only"] },
      tok,
      cfg,
    );
    expect(formatAnswer(p, [10000, -10000], [10000, 10000], cfg)).toMatchObject(
      { choice: "only", confidence: 1, probabilities: { only: 1 } },
    );
  });
  it("rejects nonfinite outputs", () => {
    const p = prepare("", { type: "noul", instructions: "" }, tok, cfg);
    expect(() => formatAnswer(p, [NaN, 0], [0], cfg)).toThrow(/finite/);
  });
  it.each([0, -1, Infinity, NaN])("rejects invalid temperature %s", (t) =>
    expect(() => validateConfig({ ...cfg, temperature: [1, t, 1] })).toThrow(),
  );
});
it("converts binary16 normal, subnormal, infinity and signed values", () => {
  expect([0, 0x3c00, 0xc000, 1, 0x7c00].map(halfToFloat)).toEqual([
    0,
    1,
    -2,
    2 ** -24,
    Infinity,
  ]);
  expect(halfToFloat(0x7e00)).toBeNaN();
});

it.each([-1, 1.5, Infinity, "3"])(
  "rejects invalid manifest size %s",
  (bytes) => {
    expect(() =>
      validateConfig({
        ...cfg,
        files: { "model.onnx": { bytes, sha256: "a".repeat(64) } },
      }),
    ).toThrow(/manifest/i);
  },
);
it.each([new Date(), new Map(), new Set(), [, "x"]])(
  "rejects non-JSON state %j",
  (state) => {
    expect(() =>
      prepare(state as never, { type: "noul", instructions: "" }, tok, cfg),
    ).toThrow(/JSON/);
  },
);
