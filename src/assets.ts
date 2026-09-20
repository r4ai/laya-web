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
