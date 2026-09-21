import { render } from "solid-js/web";
import { App, type InferenceWorker } from "../../examples/minimal/app.js";
import type { Request, Response } from "../../examples/minimal/protocol.js";
import "../../examples/minimal/style.css";

// Controlled responses let browser checks inspect every transition without timers.
class ControlledWorker implements InferenceWorker {
  onmessage: Worker["onmessage"] = null;
  onerror: Worker["onerror"] = null;
  request?: Request;
  postMessage(request: Request) {
    this.request = request;
    this.emit({ type: "running" });
  }
  terminate() {}
  emit(data: Response) {
    this.onmessage?.call(
      this as unknown as Worker,
      new MessageEvent("message", { data }),
    );
  }
  complete() {
    const question = this.request?.questions.result;
    if (
      !question ||
      question.type !== "choice" ||
      !Array.isArray(question.criteria)
    )
      return;
    const criteria = question.criteria;
    this.emit({
      type: "result",
      backend: "wasm",
      elapsed: 100,
      result: {
        model: "laya-rl-agent",
        usage: { input_tokens: 10, output_tokens: 0 },
        answers: {
          result: {
            type: "choice",
            choice: criteria[0],
            confidence: 0.5,
            probabilities: Object.fromEntries(
              criteria.map((label) => [label, 1 / criteria.length]),
            ),
            action: { act_probability: 0.9 },
          },
        },
      },
    });
  }
}
const worker = new ControlledWorker();
render(
  () => (
    <>
      <aside
        style={{
          position: "fixed",
          top: "0",
          right: "0",
          "z-index": 1,
          background: "white",
          padding: "4px",
          "font-size": "11px",
        }}
      >
        <button onClick={() => worker.complete()}>Success Response</button>
        <button
          onClick={() =>
            worker.emit({ type: "error", error: "Simulated load failure" })
          }
        >
          Failure Response
        </button>
        <output id="scroll-check" style={{ display: "block" }} />
      </aside>
      <App createWorker={() => worker} />
    </>
  ),
  document.querySelector("#root")!,
);

// Observe native clicks, including their pre-update scroll position and DOM identity.
document.addEventListener(
  "click",
  (event) => {
    if (!(event.target instanceof HTMLButtonElement)) return;
    const before = scrollY;
    const result = document.querySelector("#result");
    const details = result?.querySelector("details");
    requestAnimationFrame(() => {
      document.querySelector("#scroll-check")!.textContent = JSON.stringify({
        before,
        after: scrollY,
        sameResult: result === document.querySelector("#result"),
        sameDetails: details === document.querySelector("#result details"),
        expanded: details?.open,
      });
    });
  },
  true,
);
