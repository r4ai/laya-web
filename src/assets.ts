import { Tokenizer as HFTokenizer } from "@huggingface/tokenizers";
import type { LoadOptions, Tokenizer } from "./types.js";

export async function download(
  base: URL,
  file: string,
  options: LoadOptions,
  expectedBytes?: number,
): Promise<Uint8Array> {
  const response = await fetch(new URL(file, base), { signal: options.signal });
  if (!response.ok)
    throw new Error(
      `Cannot load ${file}: HTTP ${response.status}. Run the model exporter and serve its output.`,
    );
  const total =
    expectedBytes ??
    (Number(response.headers.get("content-length")) || undefined);
  const progress = (loaded: number) =>
    options.onProgress?.({ phase: "download", file, loaded, total });
  if (!response.body || !expectedBytes) {
    const data = new Uint8Array(await response.arrayBuffer());
    progress(data.byteLength);
    return data;
  }
  // Allocate once rather than retaining hundreds of MB of chunks and a second copy.
  const data = new Uint8Array(expectedBytes);
  const reader = response.body.getReader();
  let loaded = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.length > data.length)
        throw new Error(`Size mismatch for ${file}`);
      data.set(value, loaded);
      loaded += value.length;
      progress(loaded);
    }
    if (loaded !== expectedBytes)
      throw new Error(
        `Incomplete ${file}: expected ${expectedBytes} bytes, received ${loaded}`,
      );
    return data;
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }
}

/** Download a batch with two slots; failure cancels and settles all in-flight reads. */
export async function downloadAssets(
  base: URL,
  files: Record<string, number | undefined>,
  options: LoadOptions,
): Promise<Record<string, Uint8Array>> {
  options.signal?.throwIfAborted();
  const controller = new AbortController();
  const abort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener("abort", abort, { once: true });
  const entries = Object.entries(files);
  const assets: Record<string, Uint8Array> = {};
  let next = 0;
  async function consume() {
    try {
      while (next < entries.length) {
        controller.signal.throwIfAborted();
        const [file, bytes] = entries[next++];
        assets[file] = await download(
          base,
          file,
          {
            ...options,
            signal: controller.signal,
          },
          bytes,
        );
      }
    } catch (error) {
      controller.abort(error);
      throw error;
    }
  }
  try {
    // Announce the complete manifest before bytes arrive so UI totals remain stable.
    for (const [file, total] of entries) {
      options.onProgress?.({ phase: "download", file, loaded: 0, total });
    }
    const readers = Array.from(
      { length: Math.min(2, entries.length) },
      consume,
    );
    try {
      await Promise.all(readers);
    } catch {
      await Promise.allSettled(readers);
      throw controller.signal.reason;
    }
    return assets;
  } finally {
    options.signal?.removeEventListener("abort", abort);
  }
}

export function createTokenizer(
  json: object,
  config: Record<string, unknown>,
): Tokenizer {
  const tokenizer = new HFTokenizer(json, config);
  function special(name: string): { text: string; id: number } {
    const value = config[name];
    const text =
      typeof value === "string"
        ? value
        : (value as { content?: string } | undefined)?.content;
    const id = text ? tokenizer.token_to_id(text) : undefined;
    if (!text || id === undefined)
      throw new Error(`Tokenizer is missing ${name}`);
    return { text, id };
  }
  return {
    cls: special("cls_token").id,
    sep: special("sep_token").id,
    mask: special("mask_token").id,
    maskToken: special("mask_token").text,
    encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids,
  };
}
