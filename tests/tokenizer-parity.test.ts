import { existsSync, readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { createTokenizer } from "../src/assets.js";
import { prepare } from "../src/core.js";
const base = new URL(
  "../examples/minimal/public/models/laya/",
  import.meta.url,
);
const available = existsSync(new URL("parity.json", base));
it.skipIf(!available)(
  "matches the Rust tokenizer on all reference fixtures, including long text and mask literals",
  () => {
    const json = (name: string) =>
      JSON.parse(readFileSync(new URL(name, base), "utf8"));
    const tok = createTokenizer(
      json("tokenizer/tokenizer.json"),
      json("tokenizer/tokenizer_config.json"),
    );
    for (const fixture of json("parity.json").fixtures) {
      for (const [i, q] of Object.values(fixture.questions).entries()) {
        const p = prepare(fixture.state, q as never, tok, json("config.json"));
        expect(p.ids).toEqual(fixture.rows[i].ids);
        expect(p.markers).toEqual(fixture.rows[i].markers);
      }
    }
  },
);
