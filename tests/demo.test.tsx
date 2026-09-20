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
        choice: "請求・返金",
        probabilities: { "請求・返金": 0.8, 技術サポート: 0.1, 営業: 0.1 },
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
  fireEvent.input(screen.getByLabelText("文章"), {
    target: { value: "返金を希望" },
  });
  fireEvent.change(screen.getByLabelText("実行環境"), {
    target: { value: "wasm" },
  });
  fireEvent.click(screen.getByRole("button"));
  expect(worker.requests[0]).toMatchObject({
    state: "返金を希望",
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
  expect(screen.getByRole("status").textContent).toContain("分類しています");
  worker.emit(result);
  expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(
    "請求・返金",
  );
  expect(screen.getByText("41.8%")).toBeTruthy();
  expect(screen.getByText("80.0%")).toBeTruthy();
  expect(screen.getByText(/正答率ではありません/)).toBeTruthy();
  expect(screen.getByRole<HTMLButtonElement>("button").disabled).toBe(false);
  expect(screen.getByLabelText<HTMLTextAreaElement>("文章").value).toBe(
    "返金を希望",
  );
  submit();
  expect(screen.queryByRole("region", { name: "分類結果" })).toBeNull();
  expect(worker.requests).toHaveLength(2);
});

it.each(["重複\n重複", " \n "])(
  "rejects invalid choices %j and clears an older result",
  (choices) => {
    const { worker } = setup();
    submit();
    worker.emit(result);
    fireEvent.input(screen.getByLabelText(/選択肢/), {
      target: { value: choices },
    });
    submit();
    expect(worker.requests).toHaveLength(1);
    expect(screen.getByRole("status").textContent).toContain("重複しない");
    expect(screen.queryByRole("region", { name: "分類結果" })).toBeNull();
    fireEvent.input(screen.getByLabelText(/選択肢/), {
      target: { value: "はい\nいいえ" },
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
  expect(screen.getByRole("region", { name: "分類結果" })).toBeTruthy();
});

it.each(["crash", "send"])(
  "replaces a broken Worker after %s failure",
  (failure) => {
    const first = new FakeWorker(),
      second = new FakeWorker();
    const workers = [first, second];
    render(() => <App createWorker={() => workers.shift()!} />);
    first.failSend = failure === "send";
    submit();
    if (failure === "crash") first.crash();
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
  expect(screen.getByRole("status").textContent).toContain("Wasmで準備");
  worker.emit({ type: "progress", progress: { phase: "initialize" } });
  expect(screen.getByRole("status").textContent).toContain("初期化");
});

it("terminates the Worker and detaches handlers when unmounted during inference", () => {
  const { worker, unmount } = setup();
  submit();
  unmount();
  expect(worker.terminated).toBe(true);
  expect(worker.onmessage).toBeNull();
  expect(worker.onerror).toBeNull();
});
