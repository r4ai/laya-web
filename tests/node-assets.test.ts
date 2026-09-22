import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, beforeEach, expect, it } from "vitest";
import { download, downloadAssets } from "../src/assets.js";
import { readAsset, resolveModelUrl } from "../src/node-assets.js";

let directory: string;
beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), "laya model #"));
  await writeFile(join(directory, "weights"), new Uint8Array([1, 2, 3]));
});
afterEach(async () => {
  await rm(directory, { recursive: true, force: true });
});

it("resolves relative paths, absolute paths and file URLs as directories", () => {
  const expected = pathToFileURL(`${directory}/`).href;
  for (const value of [
    directory,
    `${directory}/`,
    relative(process.cwd(), directory),
    pathToFileURL(directory).href,
  ]) {
    expect(resolveModelUrl(value).href).toBe(expected);
  }
});
it("retains HTTP URL semantics and rejects unsupported schemes", () => {
  expect(resolveModelUrl("https://example.com/model").href).toBe(
    "https://example.com/model/",
  );
  expect(() => resolveModelUrl("ftp://example.com/model")).toThrow("protocol");
});
it("loads local assets through the shared size and progress contract", async () => {
  const events: unknown[] = [];
  const assets = await downloadAssets(
    resolveModelUrl(directory),
    { weights: 3 },
    {
      modelUrl: directory,
      onProgress: (event) => events.push(event),
    },
    readAsset,
  );
  expect([...assets.weights]).toEqual([1, 2, 3]);
  expect(events).toEqual([
    { phase: "download", file: "weights", loaded: 0, total: 3 },
    { phase: "download", file: "weights", loaded: 3, total: 3 },
  ]);
});
it.each([0, 2, 4])(
  "rejects local size mismatch (%i bytes)",
  async (expected) => {
    await expect(
      download(
        resolveModelUrl(directory),
        "weights",
        { modelUrl: directory },
        expected,
        readAsset,
      ),
    ).rejects.toThrow();
  },
);
it("rejects missing files and pre-aborted reads", async () => {
  const base = resolveModelUrl(directory);
  await expect(
    download(base, "missing", { modelUrl: directory }, undefined, readAsset),
  ).rejects.toThrow();
  await expect(
    download(
      base,
      "weights",
      { modelUrl: directory, signal: AbortSignal.abort() },
      3,
      readAsset,
    ),
  ).rejects.toThrow();
});
it("cancels a local read from its progress callback", async () => {
  await writeFile(join(directory, "large"), new Uint8Array(1024 * 1024));
  const controller = new AbortController();
  await expect(
    download(
      resolveModelUrl(directory),
      "large",
      {
        modelUrl: directory,
        signal: controller.signal,
        onProgress: () => controller.abort(),
      },
      1024 * 1024,
      readAsset,
    ),
  ).rejects.toThrow();
});
