import { createServer } from "node:http";
import { afterAll, beforeAll, expect, it } from "vitest";
import { download, createTokenizer } from "../src/assets.js";
import type { LoadProgress } from "../src/types.js";
let base: URL;
const server = createServer((req, res) => {
  if (req.url === "/missing") {
    res.writeHead(404).end();
    return;
  }
  if (req.url === "/large") {
    res.end("abcde");
    return;
  }
  res.end("abc");
});
beforeAll(async () => {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = new URL(
    `http://127.0.0.1:${(server.address() as { port: number }).port}/`,
  );
});
afterAll(async () => {
  await new Promise<void>((resolve, reject) =>
    server.close((e) => (e ? reject(e) : resolve())),
  );
});
it("streams exact bytes and reports progress", async () => {
  const events: LoadProgress[] = [];
  const bytes = await download(
    base,
    "ok",
    { modelUrl: base.href, onProgress: (e) => events.push(e) },
    3,
  );
  expect(new TextDecoder().decode(bytes)).toBe("abc");
  expect(events.at(-1)).toMatchObject({ loaded: 3, total: 3 });
});
it.each(["missing", "large", "short"])(
  "rejects %s instead of returning corrupt model data",
  async (file) => {
    await expect(
      download(base, file, { modelUrl: base.href }, file === "short" ? 4 : 3),
    ).rejects.toThrow();
  },
);
it("accepts files without a content length", async () => {
  expect((await download(base, "ok", { modelUrl: base.href })).byteLength).toBe(
    3,
  );
});
it("honors abort before fetch", async () => {
  const controller = new AbortController();
  controller.abort();
  await expect(
    download(base, "ok", { modelUrl: base.href, signal: controller.signal }),
  ).rejects.toThrow();
});
it("rejects a tokenizer with missing special tokens", () => {
  expect(() =>
    createTokenizer(
      {
        model: { type: "WordLevel", vocab: { "[UNK]": 0 }, unk_token: "[UNK]" },
      },
      {},
    ),
  ).toThrow();
});
