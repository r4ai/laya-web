export type Json =
  | null
  | boolean
  | number
  | string
  | Json[]
  | { [key: string]: Json };
export type State = string | { [key: string]: Json } | Json[];
export type Question =
  | {
      type: "choice";
      instructions: string;
      criteria: string[] | Record<string, Json>;
    }
  | { type: "score"; instructions: string; criteria: Json[] }
  | {
      type: "noul";
      instructions: string;
      criteria?: { false?: Json; true?: Json };
    };
export type Questions = Record<string, Question>;
export type Backend = "webgpu" | "wasm";
export interface AnswerBase {
  confidence: number;
  action: { act_probability: number };
}
export type Answer = AnswerBase &
  (
    | { type: "choice"; choice: string; probabilities: Record<string, number> }
    | {
        type: "score";
        score: number;
        legend: Record<string, Json>;
        probabilities: Record<string, number>;
      }
    | { type: "noul"; noul: number }
  );
export interface Prediction {
  model: "laya-rl-agent";
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: 0 };
}
export interface LoadOptions {
  /** Directory produced by the exporter, served over HTTP(S). */
  modelUrl: string;
  /** Auto tries WebGPU initialization, then creates a WASM session if it fails. */
  backend?: Backend | "auto";
  /** Directory containing ONNX Runtime's matching .wasm and .mjs files. */
  wasmPaths?: string;
  signal?: AbortSignal;
  onProgress?: (event: LoadProgress) => void;
}
export interface LoadProgress {
  phase: "download" | "initialize" | "fallback";
  file?: string;
  loaded?: number;
  total?: number;
  message?: string;
}
export interface Tokenizer {
  cls: number;
  sep: number;
  mask: number;
  maskToken: string;
  encode(text: string): number[];
}
export interface ModelConfig {
  format: "laya-web-v1";
  maxLength: number;
  headMaxLength: number;
  hiddenSize: number;
  vocabSize: number;
  temperature: [number, number, number];
  temperatureByOptions: Record<string, number>;
  files?: Record<string, { bytes: number; sha256: string }>;
}
export interface PreparedQuestion {
  ids: number[];
  markers: number[];
  qtype: number;
  question: Question;
  labels: string[];
}
