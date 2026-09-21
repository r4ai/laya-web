import { load } from "@r4ai/laya-web";
// This benchmark's fresh worker bypasses HTTP cache for every asset, including ORT.
const originalFetch = fetch;
self.fetch = (input, init) =>
  originalFetch(input, { ...init, cache: "no-store" });
onmessage = async ({ data: base }: MessageEvent<string>) => {
  try {
    const started = performance.now();
    let downloaded = started;
    const agent = await load({
      modelUrl: `${base}/models/laya/`,
      wasmPaths: `${base}/ort/`,
      backend: "auto",
      onProgress: (event) => {
        if (event.phase === "initialize") downloaded = performance.now();
      },
    });
    const ready = performance.now();
    const result = await agent.predict(
      "I was charged twice for my subscription this month. Please issue a refund.",
      {
        result: {
          type: "choice",
          instructions: "Which support department should handle this request?",
          criteria: ["Billing & Refunds", "Technical Support", "Sales"],
        },
      },
    );
    const predicted = performance.now();
    const backend = agent.backend;
    await agent.dispose();
    postMessage({
      backend,
      downloadMs: downloaded - started,
      initializeMs: ready - downloaded,
      readyMs: ready - started,
      predictMs: predicted - ready,
      result,
    });
  } catch (error) {
    postMessage({ error: String(error) });
  }
};
