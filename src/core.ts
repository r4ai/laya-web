import type {
  Answer,
  Json,
  ModelConfig,
  PreparedQuestion,
  Question,
  State,
  Tokenizer,
} from "./types.js";
const QTYPES = { choice: 0, score: 1, noul: 2 } as const;
const record = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

/** JSON formatting matches Python json.dumps(..., ensure_ascii=False). */
export function serialize(value: Json): string {
  if (Array.isArray(value)) return `[${value.map(serialize).join(", ")}]`;
  if (record(value))
    return `{${Object.entries(value)
      .map(([k, v]) => `${JSON.stringify(k)}: ${serialize(v as Json)}`)
      .join(", ")}}`;
  if (typeof value === "number" && !Number.isFinite(value))
    throw new TypeError("State must contain finite JSON numbers");
  const result = JSON.stringify(value);
  if (result === undefined) throw new TypeError("Expected a JSON value");
  return result;
}
const render = (v: Json) => (typeof v === "string" ? v : serialize(v));

export function validateQuestion(q: Question): void {
  if (
    !record(q) ||
    !Object.hasOwn(QTYPES, q.type) ||
    typeof q.instructions !== "string"
  )
    throw new TypeError("Expected question type and string instructions");
  if (q.type === "noul") {
    if (q.criteria !== undefined && !record(q.criteria))
      throw new TypeError("Noul criteria must be an object");
    return;
  }
  if (q.type === "score") {
    if (!Array.isArray(q.criteria) || !q.criteria.length)
      throw new TypeError("Score criteria must be a nonempty array");
    return;
  }
  if (Array.isArray(q.criteria)) {
    if (
      !q.criteria.length ||
      q.criteria.some((c) => typeof c !== "string") ||
      new Set(q.criteria).size !== q.criteria.length
    )
      throw new TypeError("Choice labels must be nonempty, unique strings");
  } else if (!record(q.criteria) || !Object.keys(q.criteria).length)
    throw new TypeError("Choice criteria must be a nonempty array or object");
}

function options(q: Question): { labels: string[]; texts: string[] } {
  if (q.type === "choice") {
    const entries = Array.isArray(q.criteria)
      ? q.criteria.map((k) => [k, null] as const)
      : Object.entries(q.criteria);
    return {
      labels: entries.map(([k]) => k),
      texts: entries.map(([k, v]) =>
        v === null || v === "" ? k : `${k}: ${render(v)}`,
      ),
    };
  }
  if (q.type === "score")
    return {
      labels: q.criteria.map((_, i) => String(i)),
      texts: q.criteria.map((v, i) => `level ${i}: ${render(v)}`),
    };
  return {
    labels: ["false", "true"],
    texts: (["false", "true"] as const).map((k) => {
      const v = q.criteria?.[k];
      const fallback =
        k === "true"
          ? "yes, the statement holds"
          : "no, the statement does not hold";
      return `${k}: ${v === undefined || v === null || v === "" ? fallback : render(v)}`;
    }),
  };
}

export function prepare(
  state: State,
  question: Question,
  tok: Tokenizer,
  cfg: ModelConfig,
): PreparedQuestion {
  validateQuestion(question);
  const { labels, texts } = options(question);
  const clean = (s: string) => s.replaceAll(tok.maskToken, " ");
  const head = tok.encode(
    `${question.type} question: ${clean(question.instructions)}`,
  );
  let opts = texts.map((s) => [
    tok.mask,
    ...tok.encode(` ${clean(s)}`).slice(0, 48),
  ]);
  let budget = cfg.headMaxLength - opts.reduce((n, ids) => n + ids.length, 0);
  if (budget < 16) {
    const per = Math.max(4, Math.floor((cfg.headMaxLength - 16) / opts.length));
    opts = opts.map((ids) => ids.slice(0, per));
    budget = cfg.headMaxLength - opts.reduce((n, ids) => n + ids.length, 0);
  }
  const ids = [tok.cls, ...head.slice(0, Math.max(8, budget)), tok.sep];
  const markers = opts.map((opt) => {
    const pos = ids.length;
    ids.push(...opt);
    return pos;
  });
  ids.push(tok.sep);
  // Never silently discard options or the final separator.
  if (ids.length + 1 > cfg.maxLength)
    throw new RangeError("Too many options for the model token budget");
  const stateIds = tok.encode(
    clean(typeof state === "string" ? state : serialize(state)),
  );
  ids.push(...stateIds.slice(0, cfg.maxLength - ids.length - 1), tok.sep);
  return { ids, markers, qtype: QTYPES[question.type], question, labels };
}

export function validateConfig(value: unknown): asserts value is ModelConfig {
  if (!record(value) || value.format !== "laya-web-v1")
    throw new TypeError("Unsupported model format; run the Laya Web exporter");
  for (const key of ["maxLength", "headMaxLength", "hiddenSize", "vocabSize"]) {
    if (!Number.isSafeInteger(value[key]) || (value[key] as number) <= 0)
      throw new TypeError(`Invalid ${key}`);
  }
  if (
    (value.headMaxLength as number) <= 4 ||
    (value.headMaxLength as number) >= (value.maxLength as number)
  )
    throw new TypeError("Invalid token budgets");
  if (
    !Array.isArray(value.temperature) ||
    value.temperature.length !== 3 ||
    !record(value.temperatureByOptions)
  )
    throw new TypeError("Invalid calibration");
  if (
    [...value.temperature, ...Object.values(value.temperatureByOptions)].some(
      (t) => typeof t !== "number" || !Number.isFinite(t) || t <= 0,
    )
  )
    throw new TypeError("Temperatures must be finite and positive");
}

export function softmax(values: readonly number[]): number[] {
  if (!values.length || values.some((v) => !Number.isFinite(v)))
    throw new Error("Expected finite model outputs");
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const total = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / total);
}
const round = (x: number) => Math.round(x * 10000) / 10000;

export function formatAnswer(
  p: PreparedQuestion,
  logits: readonly number[],
  action: readonly number[],
  cfg: ModelConfig,
): Answer {
  const k = p.labels.length;
  if (logits.length < k) throw new Error("Model returned too few logits");
  const bucket = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  const scale = Math.max(
    1e-3,
    cfg.temperatureByOptions[`${p.question.type}:${bucket}`] ??
      cfg.temperature[p.qtype],
  );
  const prob = softmax(logits.slice(0, k).map((v) => v / scale));
  const entropy = -prob.reduce(
    (sum, v) => sum + v * Math.log(Math.max(v, 1e-12)),
    0,
  );
  const base = {
    confidence: round(
      k < 2 ? 1 : Math.max(0, Math.min(1, 1 - entropy / Math.log(k))),
    ),
    action: { act_probability: round(softmax(action)[0]) },
  };
  if (p.question.type === "noul")
    return {
      ...base,
      type: "noul",
      noul: round(prob[1]),
      confidence: round(Math.max(prob[1], 1 - prob[1])),
    };
  const probabilities = Object.fromEntries(
    p.labels.map((label, i) => [label, round(prob[i])]),
  );
  if (p.question.type === "choice")
    return {
      ...base,
      type: "choice",
      choice: p.labels[prob.indexOf(Math.max(...prob))],
      probabilities,
    };
  return {
    ...base,
    type: "score",
    score: round(prob.reduce((s, v, i) => s + i * v, 0)),
    probabilities,
    legend: Object.fromEntries(
      p.question.criteria.map((v, i) => [String(i), v]),
    ),
  };
}

export function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return (
    sign *
    (exponent === 0
      ? fraction * 2 ** -24
      : (1 + fraction / 1024) * 2 ** (exponent - 15))
  );
}
