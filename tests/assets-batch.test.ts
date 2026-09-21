import { createServer, type ServerResponse } from "node:http";
import { afterAll, afterEach, beforeAll, expect, it, vi } from "vitest";
import { downloadAssets } from "../src/assets.js";
import type { LoadProgress } from "../src/types.js";

let base: URL;
const pending = new Map<string, ServerResponse>();
const started: string[] = [];
const closed: string[] = [];
const server = createServer((req, res) => {
  const file = req.url!.slice(1);
  started.push(file);
  pending.set(file, res);
  res.on("close", () => closed.push(file));
});
beforeAll(async () => {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = new URL(
    `http://127.0.0.1:${(server.address() as { port: number }).port}/`,
  );
});
afterEach(() => {
  for (const res of pending.values()) res.destroy();
  pending.clear();
  started.length = 0;
  closed.length = 0;
});
afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});
const files = { a: 3, b: 3, c: 3, d: 3 };
const waitForRequests = (count: number) =>
  vi.waitFor(() => expect(started).toHaveLength(count));

it("downloads at most two assets at once, refills free slots and returns exact bytes", async () => {
  const events: LoadProgress[] = [];
  const result = downloadAssets(base, files, {
    modelUrl: base.href,
    onProgress: (p) => events.push(p),
  });
  await waitForRequests(2);
  expect(new Set(started)).toEqual(new Set(["a", "b"]));
  expect(events.slice(0, 4)).toEqual(
    Object.keys(files).map((file) => ({
      phase: "download",
      file,
      loaded: 0,
      total: 3,
    })),
  );
  pending.get("b")!.end("bbb");
  await waitForRequests(3);
  expect(started[2]).toBe("c");
  pending.get("c")!.end("ccc");
  await waitForRequests(4);
  pending.get("d")!.end("ddd");
  pending.get("a")!.end("aaa");
  const assets = await result;
  for (const file of Object.keys(files)) {
    expect(new TextDecoder().decode(assets[file])).toBe(file.repeat(3));
  }
  expect(events.filter((e) => e.loaded === 3)).toHaveLength(4);
});

it.each(["http", "short", "large"])(
  "aborts its sibling and does not start queued assets after %s failure",
  async (failure) => {
    const result = downloadAssets(base, files, { modelUrl: base.href });
    const rejected = expect(result).rejects.toThrow(
      failure === "http" ? "HTTP 503" : /Incomplete|Size mismatch/,
    );
    await waitForRequests(2);
    const res = pending.get("a")!;
    if (failure === "http") res.writeHead(503).end();
    else res.end(failure === "short" ? "a" : "aaaa");
    await rejected;
    await vi.waitFor(() => expect(closed).toContain("b"));
    expect(new Set(started)).toEqual(new Set(["a", "b"]));
  },
);

it("propagates caller cancellation to both streams and does not start queued assets", async () => {
  const controller = new AbortController();
  const result = downloadAssets(base, files, {
    modelUrl: base.href,
    signal: controller.signal,
  });
  const rejected = expect(result).rejects.toThrow("Cancelled by caller");
  await waitForRequests(2);
  for (const res of pending.values()) res.write("a");
  controller.abort(new Error("Cancelled by caller"));
  await rejected;
  await vi.waitFor(() => expect(closed).toHaveLength(2));
  expect(new Set(started)).toEqual(new Set(["a", "b"]));
});

it("rejects an already aborted request without starting downloads", async () => {
  await expect(
    downloadAssets(base, files, {
      modelUrl: base.href,
      signal: AbortSignal.abort(),
    }),
  ).rejects.toThrow();
  expect(started).toEqual([]);
});

it("supports an empty batch and a single asset without a manifest size", async () => {
  expect(await downloadAssets(base, {}, { modelUrl: base.href })).toEqual({});
  const result = downloadAssets(
    base,
    { a: undefined },
    { modelUrl: base.href },
  );
  await waitForRequests(1);
  pending.get("a")!.end("abc");
  expect(new TextDecoder().decode((await result).a)).toBe("abc");
});
