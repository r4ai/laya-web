# 検証契約

## 状態遷移とテスト層

| 対象       | 開始状態・入力                                 | 期待結果                                      | 検証                            |
| ---------- | ---------------------------------------------- | --------------------------------------------- | ------------------------------- |
| 質問       | choice / score / noul                          | 互換プレフィックス、マーカー、型ID            | core単体 + 実モデル             |
| 質問       | 空の選択肢、重複ラベル、不明な型、説明欠落     | 推論前に拒否                                  | core / agent単体                |
| 文脈       | 長文                                           | 本文末尾だけ切り詰め、選択肢と終端を保持      | core単体 + 1024トークン実モデル |
| 文脈       | 選択肢が収まらない                             | 明示的エラー                                  | core単体                        |
| トークン化 | 日本語、英語、構造化JSON、mask文字列、連続空白 | Rust版と同じID列とマーカー位置                | 回帰単体 + 参照比較             |
| 出力       | 温度バケット、1選択肢、大きなlogit             | 安定softmax、ラベル対応、期待値               | core単体                        |
| 出力       | NaN / 不正温度                                 | エラー                                        | core単体                        |
| ロード     | 正常なHTTPストリーム                           | サイズ検証、実測進捗                          | ローカルHTTP結合                |
| ロード     | 404 / 過大 / 不足 / 中断                       | 不完全なデータを返さない                      | ローカルHTTP結合                |
| Agent      | ready → 2件受付                                | 直列実行                                      | 状態を持つFakeDriverによる単体  |
| Agent      | 推論失敗 → 次のリクエスト                      | 後続処理は成功可能                            | agent単体                       |
| Agent      | 中断済みリクエスト                             | 推論しない                                    | agent単体                       |
| Agent      | 実行中 → dispose                               | 受付済み処理を完了して解放                    | agent単体                       |
| Agent      | disposed → predict / dispose                   | predict拒否、二重解放なし                     | agent単体 + ブラウザ            |
| ONNX       | 可変長8/19/64、選択肢2/3/5、全3型              | PyTorchとlogit一致                            | 小型実モデル、27組合せ          |
| ブラウザ   | WebGPU / Wasm                                  | 実checkpointのトークン・選択・確率・usage一致 | ブラウザ内テストハーネス        |
| サンプル   | idle → loading → running → result              | 状態表示、入力を実行中ロック、結果表示        | ブラウザ操作                    |
| サンプル   | result → 重複選択肢で送信                      | エラー、古い結果を隠す、再試行可能            | ブラウザ操作                    |

## 再現コマンド

通常の開発環境では `pnpm typecheck`、`pnpm test`、`uv run pytest`、`pnpm build:demo`。
実モデルは `pnpm model:export` → `pnpm model:verify` → `pnpm test:browser`。
検証ページのBackendを選択してRun parity suiteを押します。

- 通常テスト: 38件。参照fixtureがあると追加のトークナイザー比較1件を実行。ない場合、その1件はskipと表示。
- Python: 小型モデル1テスト内で27組合せを検証。CPUで実行可能。
- 実モデル: 9問（日本語2、英語3、構造化state3、1024トークン長文1）。
- ブラウザ: 実モデル9問に加え、3問の同一リクエストを2回繰り返し、dispose後の拒否まで検証。
- Pythonで生成する `parity.json` と重みはGit管理しない。固定revisionとコマンドから再生成する。

## 初回実測（2026-09-21 JST）

Apple M5 / 16 GB、Codex内蔵Chromium、ONNX Runtime Web 1.30.0。
固定されたLaya-MLXのFP32 CPUを独立参照として使用。

| 検査                   | 結果                                                          |
| ---------------------- | ------------------------------------------------------------- |
| Python ONNX vs MLX CPU | 9/9 argmax一致、最大絶対logit差0.0018311（action headを含む） |
| ブラウザWebGPU         | 9/9、ID列・マーカー位置完全一致、小数4桁API出力の差0          |
| ブラウザWasm           | 9/9、ID列・マーカー位置完全一致、小数4桁API出力の差0          |
| 連続実行 / dispose     | 両backend成功                                                 |
| 単体・HTTP結合         | 39件成功、skipなし（参照fixtureあり）                         |
| V8カバレッジ           | statements 68.99%、branches 71.34%、lines 70.48%              |

V8の値はNodeでの単体・結合テストのみ。`runtime.ts` の実推論は別途ブラウザで検証しており、V8合算に含めていません。カバレッジ100%や任意入力での精度を意味しません。

実測の途中で、次の互換性問題を切り分けました。

1. MLXの既定GPU FP32では、このM5上の行列乗算に低精度経路が使われる。単一Linear演算をfloat64計算と照合し、`MLX_ENABLE_TF32=0` で差が通常の浮動小数点誤差まで下がることを確認。検証基準にはCPUのFP32を使用しています。[MLXの精度仕様](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
2. `@huggingface/tokenizers@0.2.0` がMetaspaceの `split: true` を無視し、正規化後の文字列にも `normalized: false` のadded tokenを適用する。連続空白で再現テストをRedにして、依存パッチで修正。ビルド成果物に修正版を同梱するので利用側のpackage managerには依存しません。

## 検証範囲の限界

Safari / Firefox、Android / iOSの実端末、WebGPUを持たない端末でのautoフォールバック、GPU device loss、メモリ不足からの復旧は未検証。端末間性能比較や精度評価用のベンチマークではありません。多数の入力に対するモデル自体の分類精度も未評価です。

Python exporterは固定したPyTorchのlegacy ONNX exporterを使用するため非推奨警告が出ます。固定依存での実行とONNX checker、実モデル一致を検証しています。PyTorchの更新時はこの検証を再実行してください。

サンプルの配布用ビルドは `http://127.0.0.1:4174/` で検証しました。
1200×900 / 390×844の表示で横はみ出しなし。ページ名、非空画面、エラーオーバーレイなし、console error/warnなしを確認。
日本語入力→実行→確率表示、重複選択肢→エラーと旧結果非表示→修正後の再実行、API出力の開閉とEnter操作を確認しました。
パッケージは `pnpm pack` 後に別ディレクトリへnpmインストールし、公開 `load` のimportが成功しています。
