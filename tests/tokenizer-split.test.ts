import { Tokenizer } from "@huggingface/tokenizers";
import { expect, it } from "vitest";
// Reduced reproduction of mmBERT's Metaspace(split=true) and unnormalized added tokens.
const tokenizer = new Tokenizer(
  {
    normalizer: { type: "Replace", pattern: { String: " " }, content: "▁" },
    pre_tokenizer: {
      type: "Metaspace",
      replacement: "▁",
      prepend_scheme: "always",
      split: true,
    },
    post_processor: null,
    decoder: null,
    added_tokens: [
      {
        id: 4,
        content: "▁▁",
        normalized: false,
        special: false,
        single_word: false,
        lstrip: false,
        rstrip: false,
      },
    ],
    model: {
      type: "BPE",
      vocab: { "<unk>": 0, "▁": 1, x: 2, "▁x": 3, "▁▁": 4 },
      merges: [
        ["▁", "x"],
        ["▁", "▁"],
      ],
      unk_token: "<unk>",
    },
  },
  {},
);
it.each([
  ["x  x", [3, 1, 3]],
  ["  ", [1, 1]],
  ["▁▁", [4]],
  ["x ", [3, 1]],
  ["", []],
])("preserves Rust Metaspace boundaries for %j", (text, ids) => {
  expect(
    tokenizer.encode(text as string, { add_special_tokens: false }).ids,
  ).toEqual(ids);
});
