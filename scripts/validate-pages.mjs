import { createReadStream } from "node:fs";
import { readFile, readdir, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";

const modelFiles = [
  "model.onnx",
  "model.onnx.data",
  "embeddings.f16.bin",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
];

/** Reject incomplete or corrupted model artifacts and oversized Pages sites. */
export async function validatePages(root, maxBytes = 1_000_000_000) {
  for (const file of [
    "index.html",
    "LICENSE",
    "NOTICE",
    "ort/ort-wasm-simd-threaded.jsep.mjs",
    "ort/ort-wasm-simd-threaded.jsep.wasm",
  ]) {
    if (!(await stat(join(root, file))).size)
      throw new Error(`Empty asset: ${file}`);
  }
  const model = join(root, "models/laya");
  const config = JSON.parse(await readFile(join(model, "config.json"), "utf8"));
  for (const file of modelFiles) {
    const expected = config.files[file];
    if (!expected) throw new Error(`Missing manifest entry: ${file}`);
    const path = join(model, file);
    if ((await stat(path)).size !== expected.bytes)
      throw new Error(`Size mismatch: ${file}`);
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(path)) hash.update(chunk);
    if (hash.digest("hex") !== expected.sha256)
      throw new Error(`Hash mismatch: ${file}`);
  }
  let bytes = 0;
  for (const entry of await readdir(root, {
    recursive: true,
    withFileTypes: true,
  })) {
    if (entry.isSymbolicLink())
      throw new Error(`Symlink in Pages artifact: ${entry.name}`);
    if (entry.isFile())
      bytes += (await stat(join(entry.parentPath, entry.name))).size;
  }
  if (bytes > maxBytes)
    throw new Error(`Pages size limit exceeded: ${bytes} > ${maxBytes}`);
  return bytes;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  console.log(
    `Pages artifact validated: ${await validatePages(resolve(process.argv[2] ?? "examples/minimal/dist"))} bytes`,
  );
}
