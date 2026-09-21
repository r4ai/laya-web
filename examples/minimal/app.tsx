import { batch, createSignal, For, onCleanup, Show } from "solid-js";
import type { Question } from "@r4ai/laya-web";
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
  | { phase: "result" }
  | { phase: "error"; message: string };
const initialState = "料金が二重に請求されています。重複分を返金してください。";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function status(view: View): string {
  switch (view.phase) {
    case "idle":
      return "初回はモデル約934 MBを読み込みます。";
    case "running":
      return "推論しています…";
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
      return `読み込み中 · ${mb(p.loaded ?? 0)}${p.total ? ` / ${mb(p.total)}` : ""} MB`;
    }
  }
}

export function App(props: { createWorker?: () => InferenceWorker }) {
  const [state, setState] = createSignal(initialState);
  const [instructions, setInstructions] =
    createSignal("この問い合わせを担当する部署は？");
  const [choices, setChoices] = createSignal("請求・返金\n技術サポート\n営業");
  const [questionType, setQuestionType] =
    createSignal<Question["type"]>("choice");
  const [scale, setScale] = createSignal("通常\n優先\n緊急");
  const [falseCriterion, setFalseCriterion] = createSignal("");
  const [trueCriterion, setTrueCriterion] = createSignal("");
  const [backend, setBackend] = createSignal<Request["backend"]>("auto");
  const [view, setView] = createSignal<View>({ phase: "idle" });
  const [result, setResult] = createSignal<Result>();
  const downloads = new Map<
    string | undefined,
    { loaded: number; total?: number }
  >();
  const busy = () => view().phase === "loading" || view().phase === "running";
  const progress = () => {
    const v = view();
    return v.phase === "loading" && v.progress?.phase === "download"
      ? v.progress
      : undefined;
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
      case "progress": {
        const progress = data.progress;
        if (progress.phase !== "download") {
          setView({ phase: "loading", progress });
          break;
        }
        downloads.set(progress.file, {
          loaded: progress.loaded ?? 0,
          total: progress.total,
        });
        const files = [...downloads.values()];
        setView({
          phase: "loading",
          progress: {
            phase: "download",
            loaded: files.reduce((sum, file) => sum + file.loaded, 0),
            total: files.every((file) => file.total !== undefined)
              ? files.reduce((sum, file) => sum + file.total!, 0)
              : undefined,
          },
        });
        break;
      }
      case "running":
        setView({ phase: "running" });
        break;
      case "result":
        batch(() => {
          setResult(data);
          setView({ phase: "result" });
        });
        break;
    }
  }
  function submit(event: SubmitEvent) {
    event.preventDefault();
    if (busy()) return;
    const type = questionType();
    let question: Question;
    if (type === "noul") {
      const criteria = {
        ...(falseCriterion().trim() ? { false: falseCriterion().trim() } : {}),
        ...(trueCriterion().trim() ? { true: trueCriterion().trim() } : {}),
      };
      question = {
        type,
        instructions: instructions(),
        ...(Object.keys(criteria).length ? { criteria } : {}),
      };
    } else {
      const criteria = (type === "choice" ? choices() : scale())
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      if (
        !criteria.length ||
        (type === "choice" && new Set(criteria).size !== criteria.length)
      ) {
        fail(
          type === "choice"
            ? "選択肢は重複しないように入力してください。"
            : "評価尺度を1行以上入力してください。",
        );
        return;
      }
      question = { type, instructions: instructions(), criteria };
    }
    downloads.clear();
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
          result: question,
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
        文章から、選択・評価・真偽判定。
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
        <label for="question-type">質問形式</label>
        <select
          id="question-type"
          value={questionType()}
          disabled={busy()}
          onChange={(e) =>
            setQuestionType(e.currentTarget.value as Question["type"])
          }
        >
          <option value="choice">choice · 選択</option>
          <option value="score">score · 段階評価</option>
          <option value="noul">noul · 真偽判定</option>
        </select>
        <label for="instructions">質問</label>
        <input
          id="instructions"
          required
          value={instructions()}
          onInput={(e) => setInstructions(e.currentTarget.value)}
          disabled={busy()}
        />
        <Show when={questionType() === "choice"}>
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
        </Show>
        <Show when={questionType() === "score"}>
          <label for="scale">
            評価尺度 <span class="hint">1行に1段階 · 上から0, 1, 2…</span>
          </label>
          <textarea
            id="scale"
            rows="3"
            required
            value={scale()}
            disabled={busy()}
            onInput={(e) => setScale(e.currentTarget.value)}
          />
        </Show>
        <Show when={questionType() === "noul"}>
          <p class="hint">質問に書いた命題が真である確率を返します。</p>
          <label for="false-criterion">
            falseの基準 <span class="hint">任意</span>
          </label>
          <input
            id="false-criterion"
            value={falseCriterion()}
            disabled={busy()}
            onInput={(e) => setFalseCriterion(e.currentTarget.value)}
          />
          <label for="true-criterion">
            trueの基準 <span class="hint">任意</span>
          </label>
          <input
            id="true-criterion"
            value={trueCriterion()}
            disabled={busy()}
            onInput={(e) => setTrueCriterion(e.currentTarget.value)}
          />
        </Show>
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
      <Show when={result()}>
        {(data) => (
          <ResultView data={data()} previous={view().phase !== "result"} />
        )}
      </Show>
      <footer>
        <a href="https://github.com/mizorewww/laya-mlx">Laya-MLX</a>{" "}
        をもとにしたブラウザ向け実装。
        <br />
        モデルの読み込み後、入力文をサーバーへ送信せずに推論します。
      </footer>
    </main>
  );
}

function ResultView(props: { data: Result; previous: boolean }) {
  const answer = () => props.data.result.answers.result;
  const title = () => {
    const a = answer();
    switch (a.type) {
      case "choice":
        return a.choice;
      case "score":
        return `スコア ${a.score}`;
      case "noul":
        return `真である確率 ${percent(a.noul)}`;
    }
  };
  const probabilities = () => {
    const a = answer();
    if (a.type === "noul") return { false: 1 - a.noul, true: a.noul };
    return a.probabilities;
  };
  const label = (key: string) => {
    const a = answer();
    return a.type === "score"
      ? `${key}: ${typeof a.legend[key] === "string" ? a.legend[key] : JSON.stringify(a.legend[key])}`
      : key;
  };
  return (
    <section id="result" aria-label="推論結果">
      <p class="result-version">
        {props.previous ? "前回の結果" : "今回の結果"}
      </p>
      <div class="result-heading">
        <h2 id="choice">{title()}</h2>
        <span id="timing">
          {props.data.backend === "wasm" ? "Wasm" : "WebGPU"} ·{" "}
          {(props.data.elapsed / 1000).toFixed(2)} s
        </span>
      </div>
      <p class="confidence">
        確信度 <strong>{percent(answer()?.confidence ?? 0)}</strong>
        <span>
          {answer().type === "noul"
            ? "trueとfalseのうち高い方の確率です。正答率ではありません。"
            : "確率の集中度を表す指標です。正答率ではありません。"}
        </span>
      </p>
      <div id="probabilities">
        <For each={Object.entries(probabilities())}>
          {([key, value]) => (
            <div class="probability">
              <div class="probability-label">
                <span>{label(key)}</span>
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
