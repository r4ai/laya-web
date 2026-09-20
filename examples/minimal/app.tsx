import { createSignal, For, onCleanup, Show } from "solid-js";
import type { Request, Response } from "./protocol.js";

export type InferenceWorker = Pick<
  Worker,
  "postMessage" | "terminate" | "onmessage" | "onerror"
>;
const createWorker = (): InferenceWorker =>
  new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
type Result = Extract<Response, { type: "result" }>;
type View =
  | { phase: "idle" }
  | {
      phase: "loading";
      progress?: Extract<Response, { type: "progress" }>["progress"];
    }
  | { phase: "running" }
  | { phase: "result"; data: Result }
  | { phase: "error"; message: string };
const initialState = "料金が二重に請求されています。重複分を返金してください。";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function status(view: View): string {
  switch (view.phase) {
    case "idle":
      return "初回はモデル約934 MBを読み込みます。";
    case "running":
      return "分類しています…";
    case "error":
      return view.message;
    case "result":
      return "完了 · 入力文はこのブラウザ内で処理しました。";
    case "loading": {
      const p = view.progress;
      if (!p) return "準備しています…";
      if (p.phase === "fallback")
        return "WebGPUを利用できないため、Wasmで準備しています…";
      if (p.phase === "initialize") return "モデルを初期化しています…";
      const mb = (n: number) => (n / 1e6).toFixed(1);
      return `読み込み中 · ${p.file} · ${mb(p.loaded ?? 0)}${p.total ? ` / ${mb(p.total)}` : ""} MB`;
    }
  }
}

export function App(props: { createWorker?: () => InferenceWorker }) {
  const [state, setState] = createSignal(initialState);
  const [instructions, setInstructions] =
    createSignal("この問い合わせを担当する部署は？");
  const [choices, setChoices] = createSignal("請求・返金\n技術サポート\n営業");
  const [backend, setBackend] = createSignal<Request["backend"]>("auto");
  const [view, setView] = createSignal<View>({ phase: "idle" });
  const busy = () => view().phase === "loading" || view().phase === "running";
  const progress = () => {
    const v = view();
    return v.phase === "loading" && v.progress?.phase === "download"
      ? v.progress
      : undefined;
  };
  const result = () => {
    const v = view();
    return v.phase === "result" ? v.data : undefined;
  };
  let worker: InferenceWorker | undefined;
  function releaseWorker() {
    if (!worker) return;
    worker.onmessage = null;
    worker.onerror = null;
    worker.terminate();
    worker = undefined;
  }
  onCleanup(releaseWorker);
  function fail(message: string) {
    setView({ phase: "error", message });
  }
  function receive(data: Response) {
    if (!busy()) return;
    switch (data.type) {
      case "error":
        fail(data.error);
        break;
      case "progress":
        setView({ phase: "loading", progress: data.progress });
        break;
      case "running":
        setView({ phase: "running" });
        break;
      case "result":
        setView({ phase: "result", data });
        break;
    }
  }
  function submit(event: SubmitEvent) {
    event.preventDefault();
    if (busy()) return;
    const criteria = choices()
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!criteria.length || new Set(criteria).size !== criteria.length) {
      fail("選択肢は重複しないように入力してください。");
      return;
    }
    setView({ phase: "loading" });
    try {
      if (!worker) {
        worker = (props.createWorker ?? createWorker)();
        worker.onmessage = ({ data }: MessageEvent<Response>) => receive(data);
        worker.onerror = (event) => {
          event.preventDefault();
          releaseWorker();
          fail(event.message || "推論用Workerを起動できませんでした。");
        };
      }
      const request: Request = {
        state: state(),
        questions: {
          result: { type: "choice", instructions: instructions(), criteria },
        },
        backend: backend(),
        modelUrl: new URL("./models/laya/", location.href).href,
        wasmPaths: new URL("./ort/", location.href).href,
      };
      worker.postMessage(request);
    } catch (error) {
      releaseWorker();
      fail(error instanceof Error ? error.message : String(error));
    }
  }
  return (
    <main>
      <h1>
        Laya Web<span aria-hidden="true">.</span>
      </h1>
      <p class="intro">
        文章から、ひとつ選ぶ。
        <br />
        推論はすべて、このブラウザの中で。
      </p>
      <form onSubmit={submit} aria-busy={busy()}>
        <label for="state">文章</label>
        <textarea
          id="state"
          rows="4"
          required
          value={state()}
          onInput={(e) => setState(e.currentTarget.value)}
          disabled={busy()}
        />
        <label for="instructions">質問</label>
        <input
          id="instructions"
          required
          value={instructions()}
          onInput={(e) => setInstructions(e.currentTarget.value)}
          disabled={busy()}
        />
        <label for="choices">
          選択肢 <span class="hint">1行にひとつ</span>
        </label>
        <textarea
          id="choices"
          rows="3"
          required
          value={choices()}
          onInput={(e) => setChoices(e.currentTarget.value)}
          disabled={busy()}
        />
        <div class="actions">
          <button type="submit" disabled={busy()}>
            {busy() ? "実行中…" : "実行する"}
          </button>
          <label class="backend-label" for="backend">
            実行環境
          </label>
          <select
            id="backend"
            value={backend()}
            onChange={(e) =>
              setBackend(e.currentTarget.value as Request["backend"])
            }
            disabled={busy()}
          >
            <option value="auto">自動 · WebGPU優先</option>
            <option value="webgpu">WebGPU</option>
            <option value="wasm">Wasm · CPU</option>
          </select>
        </div>
      </form>
      <p
        id="status"
        role="status"
        data-error={view().phase === "error" ? "" : undefined}
      >
        {status(view())}
      </p>
      <Show when={progress()}>
        {(p) => (
          <Show
            when={p().total}
            fallback={<progress aria-label="モデルの読み込み" />}
          >
            <progress
              aria-label="モデルの読み込み"
              max={p().total}
              value={p().loaded ?? 0}
            />
          </Show>
        )}
      </Show>
      <Show when={result()}>{(data) => <ResultView data={data()} />}</Show>
      <footer>
        <a href="https://github.com/mizorewww/laya-mlx">Laya-MLX</a>{" "}
        をもとにしたブラウザ向け実装。
        <br />
        モデルの読み込み後、入力文をサーバーへ送信せずに分類します。
      </footer>
    </main>
  );
}

function ResultView(props: { data: Result }) {
  const answer = () => {
    const a = props.data.result.answers.result;
    return a.type === "choice" ? a : undefined;
  };
  return (
    <section id="result" aria-label="分類結果">
      <div class="result-heading">
        <h2 id="choice">{answer()?.choice}</h2>
        <span id="timing">
          {props.data.backend === "wasm" ? "Wasm" : "WebGPU"} ·{" "}
          {(props.data.elapsed / 1000).toFixed(2)} s
        </span>
      </div>
      <p class="confidence">
        確信度 <strong>{percent(answer()?.confidence ?? 0)}</strong>
        <span>確率の集中度を表す指標です。正答率ではありません。</span>
      </p>
      <div id="probabilities">
        <For each={Object.entries(answer()?.probabilities ?? {})}>
          {([label, value]) => (
            <div class="probability">
              <div class="probability-label">
                <span>{label}</span>
                <span>{percent(value)}</span>
              </div>
              <div class="track">
                <div class="fill" style={{ width: percent(value) }} />
              </div>
            </div>
          )}
        </For>
      </div>
      <details>
        <summary>APIの出力</summary>
        <pre id="json">{JSON.stringify(props.data.result, null, 2)}</pre>
      </details>
    </section>
  );
}
