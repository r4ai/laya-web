// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { afterEach, expect, it } from "vitest";
import { App, type InferenceWorker } from "../examples/minimal/app.js";
import type { Request, Response } from "../examples/minimal/protocol.js";

class FakeWorker implements InferenceWorker {
  requests: Request[] = [];
  terminated = false;
  failSend = false;
  onmessage: Worker["onmessage"] = null;
  onerror: Worker["onerror"] = null;
  postMessage(request: Request) {
    if (this.failSend) throw new Error("Cannot send");
    this.requests.push(structuredClone(request));
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: Response) {
    this.onmessage?.call(
      this as unknown as Worker,
      new MessageEvent("message", { data }),
    );
  }
  crash() {
    this.onerror?.call(
      this as unknown as Worker,
      new ErrorEvent("error", { message: "Worker crashed", cancelable: true }),
    );
  }
}
const result: Response = {
  type: "result",
  backend: "wasm",
  elapsed: 100,
  result: {
    model: "laya-rl-agent",
    usage: { input_tokens: 10, output_tokens: 0 },
    answers: {
      result: {
        type: "choice",
        choice: "Billing & Refunds",
        probabilities: {
          "Billing & Refunds": 0.8,
          "Technical Support": 0.1,
          Sales: 0.1,
        },
        confidence: 0.4183,
        action: { act_probability: 0.9 },
      },
    },
  },
};
function submit() {
  const event = new Event("submit", { bubbles: true, cancelable: true });
  document.querySelector("form")!.dispatchEvent(event);
  expect(event.defaultPrevented).toBe(true);
}
function setup(worker = new FakeWorker()) {
  const app = render(() => <App createWorker={() => worker} />);
  return { worker, ...app };
}
afterEach(cleanup);

it("prevents navigation and shows an error when Worker construction fails, then retries", () => {
  const worker = new FakeWorker();
  let blocked = true;
  render(() => (
    <App
      createWorker={() => {
        if (blocked) throw new Error("Worker blocked");
        return worker;
      }}
    />
  ));
  submit();
  expect(screen.getByRole("status").textContent).toContain("Worker blocked");
  expect(screen.getByRole<HTMLButtonElement>("button").disabled).toBe(false);
  blocked = false;
  submit();
  expect(worker.requests).toHaveLength(1);
});

it("submits current input, locks while busy, and renders confidence separately from probabilities", () => {
  const { worker } = setup();
  fireEvent.input(screen.getByLabelText("Input Context"), {
    target: { value: "I need a refund" },
  });
  fireEvent.change(screen.getByLabelText("Execution Backend"), {
    target: { value: "wasm" },
  });
  fireEvent.click(screen.getByRole("button"));
  expect(worker.requests[0]).toMatchObject({
    state: "I need a refund",
    backend: "wasm",
  });
  expect(screen.getByRole<HTMLButtonElement>("button").disabled).toBe(true);
  submit(); // Guard even if a second submission bypasses the disabled button.
  expect(worker.requests).toHaveLength(1);
  worker.emit({
    type: "progress",
    progress: { phase: "download", file: "model", loaded: 50, total: 100 },
  });
  expect(screen.getByRole<HTMLProgressElement>("progressbar").value).toBe(50);
  worker.emit({ type: "running" });
  expect(screen.queryByRole("progressbar")).toBeNull();
  expect(screen.getByRole("status").textContent).toContain("Running inference");
  worker.emit(result);
  expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(
    "Billing & Refunds",
  );
  expect(screen.getByText("41.8%")).toBeTruthy();
  expect(screen.getByText("80.0%")).toBeTruthy();
  expect(screen.getByText(/not an accuracy score/)).toBeTruthy();
  expect(screen.getByRole<HTMLButtonElement>("button").disabled).toBe(false);
  expect(
    screen.getByLabelText<HTMLTextAreaElement>("Input Context").value,
  ).toBe("I need a refund");
  submit();
  expect(screen.getByRole("region", { name: "Inference Result" })).toBeTruthy();
  expect(screen.getByText("Previous Result")).toBeTruthy();
  expect(worker.requests).toHaveLength(2);
});

it.each(["Duplicate\nDuplicate", " \n "])(
  "rejects invalid choices %j and preserves the previous result",
  (choices) => {
    const { worker } = setup();
    submit();
    worker.emit(result);
    fireEvent.input(screen.getByLabelText(/Choices/i), {
      target: { value: choices },
    });
    submit();
    expect(worker.requests).toHaveLength(1);
    expect(screen.getByRole("status").textContent).toContain("unique");
    expect(
      screen.getByRole("region", { name: "Inference Result" }),
    ).toBeTruthy();
    expect(screen.getByText("Previous Result")).toBeTruthy();
    fireEvent.input(screen.getByLabelText(/Choices/i), {
      target: { value: "Yes\nNo" },
    });
    submit();
    expect(worker.requests).toHaveLength(2);
  },
);

it("handles model errors without dropping the reusable Worker", () => {
  const { worker } = setup();
  submit();
  worker.emit({ type: "error", error: "Model missing" });
  expect(screen.getByRole("status").textContent).toBe("Model missing");
  expect(worker.terminated).toBe(false);
  submit();
  worker.emit(result);
  expect(screen.getByRole("region", { name: "Inference Result" })).toBeTruthy();
});

it.each(["crash", "send"])(
  "replaces a broken Worker after %s failure",
  (failure) => {
    const first = new FakeWorker(),
      second = new FakeWorker();
    const workers = [first, second];
    render(() => <App createWorker={() => workers.shift()!} />);
    submit();
    first.emit(result);
    const region = screen.getByRole("region", { name: "Inference Result" });
    first.failSend = failure === "send";
    submit();
    if (failure === "crash") first.crash();
    expect(screen.getByRole("region", { name: "Inference Result" })).toBe(
      region,
    );
    expect(screen.getByText("Previous Result")).toBeTruthy();
    expect(first.terminated).toBe(true);
    expect(first.onmessage).toBeNull();
    expect(screen.getByRole("status").textContent).toContain(
      failure === "crash" ? "Worker crashed" : "Cannot send",
    );
    submit();
    expect(second.requests).toHaveLength(1);
  },
);

it("shows indeterminate download, fallback and initialization states", () => {
  const { worker } = setup();
  submit();
  worker.emit({
    type: "progress",
    progress: { phase: "download", file: "model" },
  });
  expect(screen.getByRole("progressbar").hasAttribute("value")).toBe(false);
  worker.emit({ type: "progress", progress: { phase: "fallback" } });
  expect(screen.queryByRole("progressbar")).toBeNull();
  expect(screen.getByRole("status").textContent).toContain(
    "falling back to Wasm",
  );
  worker.emit({ type: "progress", progress: { phase: "initialize" } });
  expect(screen.getByRole("status").textContent).toContain("Initializing");
});

it("terminates the Worker and detaches handlers when unmounted during inference", () => {
  const { worker, unmount } = setup();
  submit();
  unmount();
  expect(worker.terminated).toBe(true);
  expect(worker.onmessage).toBeNull();
  expect(worker.onerror).toBeNull();
});

it("keeps the same result DOM and expanded details through progress, errors and success", () => {
  const { worker } = setup();
  submit();
  worker.emit(result);
  const region = screen.getByRole("region", { name: "Inference Result" });
  const details = region.querySelector("details")!;
  details.open = true;
  submit();
  for (const message of [
    {
      type: "progress",
      progress: { phase: "download", file: "model", loaded: 1, total: 2 },
    },
    { type: "running" },
    { type: "error", error: "Download failed" },
  ] satisfies Response[]) {
    worker.emit(message);
    expect(screen.getByRole("region", { name: "Inference Result" })).toBe(
      region,
    );
    expect(details.open).toBe(true);
    expect(screen.getByText("Previous Result")).toBeTruthy();
  }
  submit();
  const next = structuredClone(result);
  if (next.result.answers.result.type === "choice")
    next.result.answers.result.choice = "Sales";
  worker.emit(next);
  expect(screen.getByRole("region", { name: "Inference Result" })).toBe(region);
  expect(region.querySelector("details")).toBe(details);
  expect(details.open).toBe(true);
  expect(screen.getByRole("heading", { level: 2 }).textContent).toBe("Sales");
  expect(screen.queryByText("Previous Result")).toBeNull();
});

it("aggregates interleaved progress without double counting and resets on retry", () => {
  const { worker } = setup();
  submit();
  const progress = (file: string, loaded: number, total?: number) =>
    worker.emit({
      type: "progress",
      progress: { phase: "download", file, loaded, total },
    });
  progress("a", 0, 100);
  progress("b", 0, 200);
  progress("a", 20, 100);
  progress("b", 50, 200);
  progress("a", 40, 100);
  const bar = screen.getByRole<HTMLProgressElement>("progressbar");
  expect(bar.value).toBe(90);
  expect(bar.max).toBe(300);
  progress("unknown", 10);
  expect(screen.getByRole("progressbar").hasAttribute("value")).toBe(false);
  progress("unknown", 10, 10);
  expect(screen.getByRole<HTMLProgressElement>("progressbar").value).toBe(100);
  expect(screen.getByRole<HTMLProgressElement>("progressbar").max).toBe(310);
  worker.emit({ type: "error", error: "Failed" });
  submit();
  progress("a", 2, 100);
  expect(screen.getByRole<HTMLProgressElement>("progressbar").value).toBe(2);
  expect(screen.getByRole<HTMLProgressElement>("progressbar").max).toBe(100);
});

// State transition table:
// From       | Event                          | To       | Expected
// idle       | select choice/score/noul        | idle     | matching fields; drafts retained
// idle       | empty score / duplicate choice | error    | no Worker request
// idle/error | valid criteria / optional noul | loading  | typed request; controls locked
// loading    | result for score/noul           | result   | typed value and distribution
// result     | select another type             | result   | previous answer remains intact
// Existing cases above cover progress, retry, Worker failure and unmount.
it.each(["score", "noul"] as const)(
  "submits and renders %s questions",
  (type) => {
    const { worker } = setup();
    fireEvent.change(screen.getByLabelText("Decision Type"), {
      target: { value: type },
    });
    if (type === "score") {
      fireEvent.input(screen.getByLabelText(/Evaluation Scale/i), {
        target: { value: "Low\nHigh" },
      });
    } else {
      fireEvent.input(screen.getByLabelText(/True Criterion/i), {
        target: { value: "Refund required" },
      });
    }
    submit();
    expect(worker.requests[0].questions.result).toMatchObject({
      type,
      criteria:
        type === "score" ? ["Low", "High"] : { true: "Refund required" },
    });
    expect(
      screen.getByLabelText<HTMLSelectElement>("Decision Type").disabled,
    ).toBe(true);
    const response = structuredClone(result);
    response.result.answers.result = {
      confidence: 0.5,
      action: { act_probability: 0.9 },
      ...(type === "score"
        ? {
            type,
            score: 0.75,
            legend: { "0": "Low", "1": "High" },
            probabilities: { "0": 0.25, "1": 0.75 },
          }
        : { type, noul: 0.75 }),
    };
    worker.emit(response);
    expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(
      type === "score" ? "Score 0.75" : "P(True) 75.0%",
    );
    if (type === "score") expect(screen.getByText("1: High")).toBeTruthy();
    else
      expect(
        screen.getByText(/Higher probability between true and false/),
      ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Decision Type"), {
      target: { value: "choice" },
    });
    expect(
      screen.getByLabelText<HTMLTextAreaElement>(/Choices/i).value,
    ).toContain("Billing & Refunds");
  },
);

it("validates empty score scales, accepts one level and duplicate descriptions", () => {
  const { worker } = setup();
  fireEvent.change(screen.getByLabelText("Decision Type"), {
    target: { value: "score" },
  });
  fireEvent.input(screen.getByLabelText(/Evaluation Scale/i), {
    target: { value: "\n " },
  });
  submit();
  expect(worker.requests).toHaveLength(0);
  for (const value of ["Low", "Same\nSame"]) {
    fireEvent.input(screen.getByLabelText(/Evaluation Scale/i), {
      target: { value },
    });
    submit();
    expect(worker.requests.at(-1)?.questions.result).toMatchObject({
      type: "score",
      criteria: value.split("\n"),
    });
    worker.emit({ type: "error", error: "retry" });
  }
});

it("omits optional noul criteria when blank", () => {
  const { worker } = setup();
  fireEvent.change(screen.getByLabelText("Decision Type"), {
    target: { value: "noul" },
  });
  submit();
  expect(worker.requests[0].questions.result).toEqual({
    type: "noul",
    instructions: "Which support department should handle this request?",
  });
});

it.each([
  ["False description", "", { false: "False description" }],
  [
    "False description",
    "True description",
    { false: "False description", true: "True description" },
  ],
] as const)(
  "submits noul criteria %s / %s",
  (falseValue, trueValue, criteria) => {
    const { worker } = setup();
    fireEvent.change(screen.getByLabelText("Decision Type"), {
      target: { value: "noul" },
    });
    fireEvent.input(screen.getByLabelText(/False Criterion/i), {
      target: { value: falseValue },
    });
    fireEvent.input(screen.getByLabelText(/True Criterion/i), {
      target: { value: trueValue },
    });
    submit();
    expect(worker.requests[0].questions.result).toMatchObject({
      type: "noul",
      criteria,
    });
  },
);
