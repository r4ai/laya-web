import { createReadStream } from "node:fs";
import { resolve, sep } from "node:path";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";
import type { AssetReader } from "./assets.js";

export function resolveModelUrl(value: string): URL {
  // Drive letters are paths, even though URL parsers treat them as schemes.
  const isUrl =
    /^[a-z][a-z\d+.-]*:/i.test(value) && !/^[a-z]:[\\/]/i.test(value);
  const url = isUrl ? new URL(value) : pathToFileURL(`${resolve(value)}${sep}`);
  if (!["file:", "http:", "https:"].includes(url.protocol))
    throw new TypeError(`Unsupported model URL protocol: ${url.protocol}`);
  if (!url.pathname.endsWith("/")) url.pathname += "/";
  return url;
}

export const readAsset: AssetReader = async (url, signal) => {
  signal?.throwIfAborted();
  if (url.protocol !== "file:") return fetch(url, { signal });
  const stream = createReadStream(url, { signal });
  return new Response(Readable.toWeb(stream) as ReadableStream<Uint8Array>);
};
