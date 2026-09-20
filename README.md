# Laya Web

[Laya-MLX](https://github.com/mizorewww/laya-mlx) の typed decision モデルを、ONNX Runtime Web の **WebGPU / Wasm** でブラウザ内実行する TypeScript ライブラリ。文章生成ではなく、`choice`（選択）、`score`（ルーブリック評価）、`noul`（命題の確率）を返します。

最小サンプルは Vanilla TypeScript + Vite。推論は Web Worker 内で行い、Wasm 実行中も画面を操作できます。推論サーバーや API キーは不要です。

## まず動かす

Node.js 22.12+、pnpm、[uv](https://docs.astral.sh/uv/) が必要です。Python は uv が管理します。

```sh
pnpm install --frozen-lockfile
pnpm model:export
pnpm dev
```

表示された localhost URL を開き、**実行する**を押します。日本語対応の `convaiinnovations/laya-multilingual` を既定で使います。初回変換は元のチェックポイントを約644 MBダウンロードします。ブラウザ用ファイルは合計約934 MBです。重みはGit管理しません。

変換先が既に存在すると上書きせず失敗します。再変換する場合は既存フォルダーを移動するか、別の出力先を指定してください。

```sh
uv run python scripts/export_model.py --output /path/to/new-model
```

## ライブラリとして使う

まだ npm には公開していません。ローカルで `pnpm pack` し、利用側で生成された `laya-web-0.1.0.tgz` をインストールできます。ESM / TypeScript型定義を含み、フレームワークに依存しません。

```ts
import { load } from "laya-web";

const agent = await load({
  modelUrl: "/models/laya/",
  backend: "auto", // 'webgpu' | 'wasm' | 'auto'
  wasmPaths: "/ort/",
  onProgress: (progress) => console.log(progress),
});

try {
  const result = await agent.predict(
    "料金が二重に請求されています。返金してください。",
    {
      department: {
        type: "choice",
        instructions: "この問い合わせを担当する部署は？",
        criteria: ["請求・返金", "技術サポート", "営業"],
      },
      urgency: {
        type: "score",
        instructions: "どの程度急いで対応すべきですか？",
        criteria: ["通常", "早め", "緊急"],
      },
      refund: {
        type: "noul",
        instructions: "顧客は返金を求めていますか？",
      },
    },
  );
  console.log(agent.backend, result.answers, result.usage);
} finally {
  await agent.dispose();
}
```

- `state`: 文字列、JSONオブジェクト、配列。質問は ID をキーにしたオブジェクト。
- `choice.criteria`: 重複しないラベル配列、または `{ ラベル: 説明 }`。オブジェクトはJavaScriptの列挙順を使用。
- `score.criteria`: 空でない配列。`score` は **0始まり** のレベルの期待値。
- `noul`: `noul` が P(true)。任意で `criteria: { false: '説明', true: '説明' }` を指定。
- 戻り値は upstream 同様の `model`、`answers`、`usage`。確率は小数4桁。`output_tokens` は0。
- `backend: 'auto'` はWebGPU初期化失敗時にWasmへ移行し、進捗イベントで通知。明示的な `'webgpu'` 指定時と、推論中の失敗はそのままエラー。
- WebGPUでも非対応演算はWasmに割り当てられます。`agent.backend` は選択した実行プロバイダー構成を表し、全演算がGPUで動いたことを保証しません。
- 質問は1問ずつ処理し、同じagentへの呼び出しを直列化。`dispose()` は受付済み処理の完了後に解放し、以後の呼び出しを拒否します。
- `load({ signal })` と `predict(state, questions, { signal })` で中断可能。実行済みのONNX演算自体は中断できず、完了後に結果を破棄します。

UIを止めないため、実アプリでもWorkerから呼び出してください。最小例は [`examples/minimal/worker.ts`](examples/minimal/worker.ts) です。

## モデルと配信

モデルは元のMLX重み形式のままでは使えません。エクスポーターが次の構成に変換します。

```text
models/laya/
  config.json
  model.onnx
  model.onnx.data
  embeddings.f16.bin
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

多言語モデルの巨大な語彙行列がWebGPUの単一バッファ制限を超えるため、FP16の埋め込み表から必要な行だけをCPUで取り出し、FP32入力としてONNXへ渡します。エンコーダー、decision head、action headはONNXで実行。独自の量子化や再学習はしていません。

- 既定モデルのリビジョン: `052592a15d198d9ad47da779604259b10b47b7aa`。
- コンテキストは質問・選択肢・本文を合わせて1,024トークン。本文の末尾を切り詰め、選択肢が収まらない場合はエラー。
- 配布ファイル約934 MBに加え、実行時に数GBのメモリが必要です。初版はデスクトップ向け。スマートフォンの実推論は未検証。
- `model.onnx.data` と埋め込み表を含むフォルダー全体をHTTP(S)配信。別ドメインならCORSを許可。
- `wasmPaths` にはインストールした **同じバージョン** の `onnxruntime-web/dist/ort-wasm*.{mjs,wasm}` を配置。サンプルのVite設定が開発・ビルド時に用意します。
- WebGPUにはHTTPSまたはlocalhostが必要。WasmはSIMD対応ブラウザで1スレッド実行し、COOP/COEPヘッダーは不要。
- ORTの環境設定は同じJSコンテキスト内で共有されます。`wasmPaths` は最初のロードから統一してください。
- ロード時はサイズ整合性を検証。SHA-256はエクスポート時に記録し、`model:verify` で検証。ブラウザのロード時に署名・ハッシュ検証は行いません。
- 初回はすべてダウンロードします。以後のキャッシュはHTTPのキャッシュ設定とブラウザ容量に依存し、永続オフラインキャッシュは提供しません。

他のModernBERT系Layaチェックポイントは `--source` / `--revision` で指定可能ですが、実モデル検証済みなのは既定の多言語版だけです。Router・学習・バッチ並列推論は対象外です。

## ビルドと検証

```sh
pnpm typecheck
pnpm test                       # 単体・HTTP結合テストとV8カバレッジ
uv run pytest                   # 小型モデルのPyTorch / ONNX比較
pnpm build                      # dist/ にライブラリと型定義
pnpm build:demo                 # examples/minimal/dist/ に静的サイト
```

実モデルの参照検証はApple Silicon上で行います。MLXは検証用で、変換やブラウザ実行には不要です。

```sh
pnpm model:verify               # 固定リビジョンのMLX FP32 CPUとの比較
pnpm test:browser               # 表示URLでWebGPU / Wasmの検証を実行
```

ブラウザ検証はモックを使わず、トークン列・マーカー位置・選択結果・確率・繰り返し実行・解放後の呼び出しを検査します。状態遷移と検証範囲は [`docs/validation.md`](docs/validation.md) を参照してください。

## 出典

Apache-2.0。派生元と変更点は [NOTICE](NOTICE) を参照。

- [Laya-MLX](https://github.com/mizorewww/laya-mlx)
- [元の多言語モデル](https://huggingface.co/convaiinnovations/laya-multilingual)
- [ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime配信要件](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
