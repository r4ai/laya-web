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
const initialState =
  "I was charged twice for my subscription this month. Please issue a refund.";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function status(view: View): string {
  switch (view.phase) {
    case "idle":
      return "First run downloads ~934 MB of model assets";
    case "running":
      return "Running inference…";
    case "error":
      return view.message;
    case "result":
      return "Completed · Processed entirely in this browser";
    case "loading": {
      const p = view.progress;
      if (!p) return "Preparing…";
      if (p.phase === "fallback")
        return "WebGPU unavailable; falling back to Wasm…";
      if (p.phase === "initialize") return "Initializing model…";
      const mb = (n: number) => (n / 1e6).toFixed(1);
      return `Loading · ${mb(p.loaded ?? 0)}${p.total ? ` / ${mb(p.total)}` : ""} MB`;
    }
  }
}

export function App(props: { createWorker?: () => InferenceWorker }) {
  const [state, setState] = createSignal(initialState);
  const [instructions, setInstructions] = createSignal(
    "Which support department should handle this request?",
  );
  const [choices, setChoices] = createSignal(
    "Billing & Refunds\nTechnical Support\nSales",
  );
  const [questionType, setQuestionType] =
    createSignal<Question["type"]>("choice");
  const [scale, setScale] = createSignal("Normal\nElevated\nImmediate");
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
            ? "Choices must be unique"
            : "Enter at least one evaluation scale level",
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
          fail(event.message || "Failed to start inference worker");
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
        Typed decisions directly from text: choice, scoring, and binary
        verification.
        <br />
        All inference runs locally without server requests.
      </p>
      <form onSubmit={submit} aria-busy={busy()}>
        <label for="state">Input Context</label>
        <textarea
          id="state"
          rows="4"
          required
          value={state()}
          onInput={(e) => setState(e.currentTarget.value)}
          disabled={busy()}
        />
        <label for="question-type">Decision Type</label>
        <select
          id="question-type"
          value={questionType()}
          disabled={busy()}
          onChange={(e) =>
            setQuestionType(e.currentTarget.value as Question["type"])
          }
        >
          <option value="choice">choice · Categorical choice</option>
          <option value="score">score · Ordinal score</option>
          <option value="noul">noul · Binary verification</option>
        </select>
        <label for="instructions">Instructions</label>
        <input
          id="instructions"
          required
          value={instructions()}
          onInput={(e) => setInstructions(e.currentTarget.value)}
          disabled={busy()}
        />
        <Show when={questionType() === "choice"}>
          <label for="choices">
            Choices <span class="hint">one per line</span>
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
            Evaluation Scale{" "}
            <span class="hint">one level per line · ordered 0, 1, 2…</span>
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
          <p class="hint">
            Computes the probability that the proposition is true
          </p>
          <label for="false-criterion">
            False Criterion <span class="hint">optional</span>
          </label>
          <input
            id="false-criterion"
            value={falseCriterion()}
            disabled={busy()}
            onInput={(e) => setFalseCriterion(e.currentTarget.value)}
          />
          <label for="true-criterion">
            True Criterion <span class="hint">optional</span>
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
            {busy() ? "Running…" : "Run"}
          </button>
          <label class="backend-label" for="backend">
            Execution Backend
          </label>
          <select
            id="backend"
            value={backend()}
            onChange={(e) =>
              setBackend(e.currentTarget.value as Request["backend"])
            }
            disabled={busy()}
          >
            <option value="auto">Auto · Prefer WebGPU</option>
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
            fallback={<progress aria-label="Loading model" />}
          >
            <progress
              aria-label="Loading model"
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
        In-browser implementation based on{" "}
        <a href="https://github.com/mizorewww/laya-mlx">Laya-MLX</a>.
        <br />
        After model download, inference runs locally without sending input to
        any server.
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
        return `Score ${a.score}`;
      case "noul":
        return `P(True) ${percent(a.noul)}`;
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
    <section id="result" aria-label="Inference Result">
      <p class="result-version">
        {props.previous ? "Previous Result" : "Latest Result"}
      </p>
      <div class="result-heading">
        <h2 id="choice">{title()}</h2>
        <span id="timing">
          {props.data.backend === "wasm" ? "Wasm" : "WebGPU"} ·{" "}
          {(props.data.elapsed / 1000).toFixed(2)} s
        </span>
      </div>
      <p class="confidence">
        Confidence <strong>{percent(answer()?.confidence ?? 0)}</strong>
        <span>
          {answer().type === "noul"
            ? "Higher probability between true and false; not an accuracy score"
            : "Probability concentration metric (normalized Shannon entropy); not an accuracy score"}
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
        <summary>API Output</summary>
        <pre id="json">{JSON.stringify(props.data.result, null, 2)}</pre>
      </details>
    </section>
  );
}
