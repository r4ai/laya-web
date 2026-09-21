import { For, Show } from "solid-js";
import { createStore } from "solid-js/store";
import type { Question } from "@r4ai/laya-web";
import {
  createInference,
  spawnWorker,
  type Download,
  type InferenceWorker,
  type Result,
  type View,
} from "./inference.js";
import { defaultDraft, toQuestion, type Draft } from "./question.js";
import type { Request } from "./protocol.js";

export type { InferenceWorker };

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);

/** Draft fields edited through a text control. */
type TextField = Exclude<keyof Draft, "type" | "backend">;

function status(view: View, download: Download | undefined): string {
  switch (view.phase) {
    case "idle":
      return "First run downloads ~934 MB of model assets";
    case "running":
      return "Running inference…";
    case "error":
      return view.message;
    case "result":
      return "Completed · Processed entirely in this browser";
    case "loading":
      if (download)
        return `Loading · ${megabytes(download.loaded)}${download.total ? ` / ${megabytes(download.total)}` : ""} MB`;
      switch (view.progress?.phase) {
        case "fallback":
          return "WebGPU unavailable; falling back to Wasm…";
        case "initialize":
          return "Initializing model…";
        default:
          return "Preparing…";
      }
  }
}

/** Labelled text control; `rows` renders a textarea instead of a single-line input. */
function Field(props: {
  id: string;
  label: string;
  hint?: string;
  rows?: number;
  required?: boolean;
  value: string;
  disabled: boolean;
  onInput: (value: string) => void;
}) {
  const control = {
    id: props.id,
    get required() {
      return props.required;
    },
    get value() {
      return props.value;
    },
    get disabled() {
      return props.disabled;
    },
    onInput: (event: { currentTarget: { value: string } }) =>
      props.onInput(event.currentTarget.value),
  };
  return (
    <>
      <label for={props.id}>
        {props.label}
        <Show when={props.hint}>
          {(hint) => (
            <>
              {" "}
              <span class="hint">{hint()}</span>
            </>
          )}
        </Show>
      </label>
      <Show when={props.rows} fallback={<input {...control} />}>
        {(rows) => <textarea {...control} rows={rows()} />}
      </Show>
    </>
  );
}

export function App(props: { createWorker?: () => InferenceWorker }) {
  const [draft, setDraft] = createStore({ ...defaultDraft });
  const { view, result, busy, download, fail, start } = createInference(() =>
    (props.createWorker ?? spawnWorker)(),
  );
  /** Binds a text control to its draft field. */
  const bind = (key: TextField) => ({
    get value() {
      return draft[key];
    },
    get disabled() {
      return busy();
    },
    onInput: (value: string) => setDraft(key, value),
  });
  function submit(event: SubmitEvent) {
    event.preventDefault();
    if (busy()) return;
    const question = toQuestion(draft);
    if (question instanceof Error) {
      fail(question.message);
      return;
    }
    start({
      state: draft.state,
      questions: { result: question },
      backend: draft.backend,
    });
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
        <Field
          id="state"
          label="Input Context"
          rows={4}
          required
          {...bind("state")}
        />
        <label for="question-type">Decision Type</label>
        <select
          id="question-type"
          value={draft.type}
          disabled={busy()}
          onChange={(e) =>
            setDraft("type", e.currentTarget.value as Question["type"])
          }
        >
          <option value="choice">choice · Categorical choice</option>
          <option value="score">score · Ordinal score</option>
          <option value="noul">noul · Binary verification</option>
        </select>
        <Field
          id="instructions"
          label="Instructions"
          required
          {...bind("instructions")}
        />
        <Show when={draft.type === "choice"}>
          <Field
            id="choices"
            label="Choices"
            hint="one per line"
            rows={3}
            required
            {...bind("choices")}
          />
        </Show>
        <Show when={draft.type === "score"}>
          <Field
            id="scale"
            label="Evaluation Scale"
            hint="one level per line · ordered 0, 1, 2…"
            rows={3}
            required
            {...bind("scale")}
          />
        </Show>
        <Show when={draft.type === "noul"}>
          <p class="hint">
            Computes the probability that the proposition is true
          </p>
          <Field
            id="false-criterion"
            label="False Criterion"
            hint="optional"
            {...bind("falseCriterion")}
          />
          <Field
            id="true-criterion"
            label="True Criterion"
            hint="optional"
            {...bind("trueCriterion")}
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
            value={draft.backend}
            disabled={busy()}
            onChange={(e) =>
              setDraft("backend", e.currentTarget.value as Request["backend"])
            }
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
        {status(view(), download())}
      </p>
      <Show when={download()}>
        {(bytes) => (
          <Show
            when={bytes().total}
            fallback={<progress aria-label="Loading model" />}
          >
            {(total) => (
              <progress
                aria-label="Loading model"
                max={total()}
                value={bytes().loaded}
              />
            )}
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
    if (a.type !== "score") return key;
    const description = a.legend[key];
    return `${key}: ${typeof description === "string" ? description : JSON.stringify(description)}`;
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
        Confidence <strong>{percent(answer().confidence)}</strong>
        <span>
          {answer().type === "noul"
            ? "Higher probability between true and false; not an accuracy score"
            : "Probability concentration metric (1 minus normalized Shannon entropy); not an accuracy score"}
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
