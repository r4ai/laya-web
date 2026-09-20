import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { afterEach, expect, it } from "vitest";
import { validatePages } from "../scripts/validate-pages.mjs";

const roots: string[] = [];
afterEach(() =>
  roots
    .splice(0)
    .forEach((root) => rmSync(root, { recursive: true, force: true })),
);
function fixture() {
  const root = mkdtempSync(join(tmpdir(), "laya-pages-"));
  roots.push(root);
  const files: Record<string, { bytes: number; sha256: string }> = {};
  for (const name of [
    "model.onnx",
    "model.onnx.data",
    "embeddings.f16.bin",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
  ]) {
    const path = join(root, "models/laya", name);
    mkdirSync(join(path, ".."), { recursive: true });
    writeFileSync(path, "model");
    files[name] = {
      bytes: 5,
      sha256: createHash("sha256").update("model").digest("hex"),
    };
  }
  writeFileSync(
    join(root, "models/laya/config.json"),
    JSON.stringify({ files }),
  );
  mkdirSync(join(root, "ort"));
  for (const file of [
    "index.html",
    "LICENSE",
    "NOTICE",
    "ort/ort-wasm-simd-threaded.jsep.mjs",
    "ort/ort-wasm-simd-threaded.jsep.wasm",
  ])
    writeFileSync(join(root, file), "asset");
  return root;
}
it("accepts a complete site below the size limit", async () => {
  expect(await validatePages(fixture())).toBeGreaterThan(0);
});
it("rejects a missing model", async () => {
  const root = fixture();
  rmSync(join(root, "models/laya/model.onnx.data"));
  await expect(validatePages(root)).rejects.toThrow();
});
it("rejects same-size model corruption", async () => {
  const root = fixture();
  writeFileSync(join(root, "models/laya/model.onnx.data"), "wrong");
  await expect(validatePages(root)).rejects.toThrow(/hash/i);
});
it("rejects an oversized site", async () => {
  await expect(validatePages(fixture(), 1)).rejects.toThrow(/limit/i);
});
it("rejects an incomplete manifest", async () => {
  const root = fixture();
  writeFileSync(join(root, "models/laya/config.json"), '{"files":{}}');
  await expect(validatePages(root)).rejects.toThrow();
});
