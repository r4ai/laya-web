import type {
  Backend,
  LoadProgress,
  Prediction,
  Questions,
  State,
} from "laya-web";
export type Request = {
  backend: Backend | "auto";
  modelUrl: string;
  wasmPaths: string;
  state: State;
  questions: Questions;
};
export type Response =
  | { type: "progress"; progress: LoadProgress }
  | { type: "running" }
  | { type: "result"; result: Prediction; backend: Backend; elapsed: number }
  | { type: "error"; error: string };
