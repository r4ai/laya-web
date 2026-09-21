import { batch, createSignal, onCleanup } from "solid-js";
import type { LoadProgress } from "@r4ai/laya-web";
import type { Request, Response } from "./protocol.js";

export type InferenceWorker = Pick<
  Worker,
  "postMessage" | "terminate" | "onmessage" | "onerror"
>;
export type Result = Extract<Response, { type: "result" }>;
/** Phase of the single inference the UI is showing; `result` stays readable while the next one runs. */
export type View =
  | { phase: "idle" }
  | { phase: "loading"; progress?: LoadProgress }
  | { phase: "running" }
  | { phase: "result" }
  | { phase: "error"; message: string };
/** Byte counts of one asset, or of the current download as a whole. */
export type Download = { loaded: number; total?: number };

export const spawnWorker = (): InferenceWorker =>
  new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });

/** Owns the Worker lifecycle and the state of the inference it is running. */
export function createInference(createWorker: () => InferenceWorker) {
  const [view, setView] = createSignal<View>({ phase: "idle" });
  const [result, setResult] = createSignal<Result>();
  // Per-file byte counts of the current download; the totals shown are derived from them.
  const assets = new Map<string | undefined, Download>();
  let worker: InferenceWorker | undefined;

  const busy = () => view().phase === "loading" || view().phase === "running";
  const fail = (message: string) => setView({ phase: "error", message });
  /** Aggregated download, or `undefined` while nothing is downloading. */
  const download = (): Download | undefined => {
    const current = view();
    if (current.phase !== "loading" || current.progress?.phase !== "download")
      return undefined;
    const files = [...assets.values()];
    return {
      loaded: files.reduce((sum, file) => sum + file.loaded, 0),
      total: files.every((file) => file.total !== undefined)
        ? files.reduce((sum, file) => sum + file.total!, 0)
        : undefined,
    };
  };

  function release() {
    if (!worker) return;
    worker.onmessage = null;
    worker.onerror = null;
    worker.terminate();
    worker = undefined;
  }
  onCleanup(release);

  function receive(data: Response) {
    if (!busy()) return;
    switch (data.type) {
      case "progress":
        if (data.progress.phase === "download")
          assets.set(data.progress.file, {
            loaded: data.progress.loaded ?? 0,
            total: data.progress.total,
          });
        setView({ phase: "loading", progress: data.progress });
        break;
      case "running":
        setView({ phase: "running" });
        break;
      case "result":
        batch(() => {
          setResult(data);
          setView({ phase: "result" });
        });
        break;
      case "error":
        fail(data.error);
        break;
    }
  }

  function connect() {
    const created = createWorker();
    created.onmessage = ({ data }: MessageEvent<Response>) => receive(data);
    created.onerror = (event) => {
      event.preventDefault();
      release();
      fail(event.message || "Failed to start inference worker");
    };
    return created;
  }

  /** Sends one request, reusing the Worker unless it failed. */
  function start(request: Pick<Request, "state" | "questions" | "backend">) {
    assets.clear();
    setView({ phase: "loading" });
    try {
      worker ??= connect();
      worker.postMessage({
        ...request,
        modelUrl: new URL("./models/laya/", location.href).href,
        wasmPaths: new URL("./ort/", location.href).href,
      } satisfies Request);
    } catch (error) {
      release();
      fail(error instanceof Error ? error.message : String(error));
    }
  }

  return { view, result, busy, download, fail, start };
}
