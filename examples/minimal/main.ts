import "./style.css";
import type { Request, Response } from "./protocol.js";
const form = document.querySelector("form")!;
const state = document.querySelector<HTMLTextAreaElement>("#state")!;
const instructions = document.querySelector<HTMLInputElement>("#instructions")!;
const choices = document.querySelector<HTMLTextAreaElement>("#choices")!;
const backend = document.querySelector<HTMLSelectElement>("#backend")!;
const button = document.querySelector("button")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const progress = document.querySelector<HTMLProgressElement>("#progress")!;
const result = document.querySelector<HTMLElement>("#result")!;
const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
function busy(value: boolean) {
  button.disabled = value;
  backend.disabled = value;
  state.disabled = value;
  instructions.disabled = value;
  choices.disabled = value;
  button.textContent = value ? "実行中…" : "実行する";
  form.setAttribute("aria-busy", String(value));
}
function fail(message: string) {
  result.hidden = true;
  status.textContent = message;
  status.dataset.error = "";
  progress.hidden = true;
  busy(false);
}
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const criteria = choices.value
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!criteria.length || new Set(criteria).size !== criteria.length) {
    fail("選択肢は重複しないように入力してください。");
    return;
  }
  delete status.dataset.error;
  busy(true);
  result.hidden = true;
  status.textContent = "準備しています…";
  const request: Request = {
    state: state.value,
    questions: {
      result: { type: "choice", instructions: instructions.value, criteria },
    },
    backend: backend.value as Request["backend"],
    modelUrl: new URL("./models/laya/", location.href).href,
    wasmPaths: new URL("./ort/", location.href).href,
  };
  worker.postMessage(request);
});
worker.onerror = (event) =>
  fail(event.message || "推論用Workerを起動できませんでした。");
worker.onmessage = ({ data }: MessageEvent<Response>) => {
  if (data.type === "error") {
    fail(data.error);
    return;
  }
  if (data.type === "running") {
    status.textContent = "分類しています…";
    progress.hidden = true;
    return;
  }
  if (data.type === "progress") {
    const p = data.progress;
    progress.hidden = p.phase !== "download";
    if (p.total) {
      progress.max = p.total;
      progress.value = p.loaded ?? 0;
    } else {
      progress.removeAttribute("value");
    }
    const mb = (n: number) => (n / 1e6).toFixed(1);
    status.textContent =
      p.phase === "download"
        ? `読み込み中 · ${p.file} · ${mb(p.loaded ?? 0)}${p.total ? ` / ${mb(p.total)}` : ""} MB`
        : p.phase === "fallback"
          ? "WebGPUを利用できないため、Wasmで準備しています…"
          : "モデルを初期化しています…";
    return;
  }
  busy(false);
  progress.hidden = true;
  result.hidden = false;
  status.textContent = "完了 · 入力文はこのブラウザ内で処理しました。";
  document.querySelector("#timing")!.textContent =
    `${data.backend === "wasm" ? "Wasm" : "WebGPU"} · ${(data.elapsed / 1000).toFixed(2)} s`;
  document.querySelector("#json")!.textContent = JSON.stringify(
    data.result,
    null,
    2,
  );
  const answer = data.result.answers.result;
  if (answer.type !== "choice") return;
  document.querySelector("#choice")!.textContent = answer.choice;
  const rows = Object.entries(answer.probabilities).map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "probability";
    const heading = document.createElement("div");
    heading.className = "probability-label";
    const name = document.createElement("span");
    name.textContent = label;
    const percent = document.createElement("span");
    percent.textContent = `${(value * 100).toFixed(1)}%`;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${value * 100}%`;
    track.append(fill);
    heading.append(name, percent);
    row.append(heading, track);
    return row;
  });
  document.querySelector("#probabilities")!.replaceChildren(...rows);
};
