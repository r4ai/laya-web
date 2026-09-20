# 開発ガイド (Development Guide)

本ドキュメントでは、`laya-web` (リポジトリ名: `hello-jev`) の開発環境構築、モデルエクスポート、ビルド、テスト、および CI/CD デプロイに関する詳細な手順を解説します。

---

## 🛠 開発環境構築

### 必要なツール
- **Node.js**: `v22.12.0` 以上
- **パッケージマネージャー**: `pnpm` (v11.x)
- **Python 環境**: [`uv`](https://docs.astral.sh/uv/) （Python 本体や依存ライブラリは `uv` が自動管理します）

### セットアップ手順

```sh
# 1. 依存関係のインストール
pnpm install --frozen-lockfile

# 2. モデルのエクスポート (初回必須)
pnpm model:export

# 3. 開発サーバーの起動
pnpm dev
```

ブラウザで表示された URL（標準: `http://127.0.0.1:5173/`）を開き、動作を確認します。

---

## 📦 モデルのエクスポートと管理

`laya-web` では、Hugging Face 上の MLX チェックポイントを ONNX Runtime Web 用のフォーマットに変換して使用します。

### 既定モデル
- **モデル名**: `convaiinnovations/laya-multilingual`
- **リビジョン**: `052592a15d198d9ad47da779604259b10b47b7aa`
- **ダウンロードサイズ**: 約 644 MB
- **変換後出力サイズ**: 約 934 MB (`examples/minimal/public/models/laya/`)

### 再変換・別フォルダへの出力
変換先が既に存在する場合、上書きされずにエラーとなります。出力先を変更して再変換する場合は以下を実行します：

```sh
uv run python scripts/export_model.py --output /path/to/new-model
```

別の ModernBERT 系 Laya チェックポイントを指定する場合は `--source` および `--revision` オプションを使用できます（※検証済みは既定の多言語版のみです）。

---

## 🛠 コマンドリファレンス

### 開発・検証コマンド

| コマンド | 内容 |
| :--- | :--- |
| `pnpm typecheck` | TypeScript の型チェック |
| `pnpm test` | Vitest による単体・結合テスト（V8 カバレッジ付き） |
| `uv run pytest` | PyTorch / ONNX 間のロジット一致検証 |
| `pnpm model:verify` | MLX FP32 CPU 参照値とエクスポートモデルの出力照合 |
| `pnpm test:browser` | ブラウザ（WebGPU / Wasm）実機検証テストハブの起動 |

### ビルドコマンド

| コマンド | 内容 |
| :--- | :--- |
| `pnpm build` | ライブラリ本体のビルド (`dist/` に TypeScript 型定義と ESM を出力) |
| `pnpm build:demo` | デモアプリのビルド (`examples/minimal/dist/`) |
| `pnpm build:pages` | GitHub Pages 公開用の検証付きビルド |
| `pnpm pack` | ローカル検証用の npm パッケージ (`.tgz`) の作成 |

---

## 🌐 GitHub Pages デプロイと CI/CD

本リポジトリは GitHub Actions を用いて GitHub Pages に自動デプロイされます (`.github/workflows/pages.yml`)。

### CI/CD パイプラインの流れ
1. **キャッシュ確認**: 変換スクリプトと Python 依存ハッシュをキーに、変換済みモデルをキャッシュ。
2. **モデル変換**: キャッシュ未ヒット時のみ Hugging Face からダウンロードしてエクスポート。
3. **品質検証**: `pnpm typecheck`、`pnpm test` (カバレッジ計測)、および `pnpm build:pages` を実行。
4. **配信物検査 (`validate-pages.mjs`)**:
   - 必須ファイル (`index.html`, `model.onnx`, `embeddings.f16.bin` 等) の存在チェック。
   - モデルの SHA-256 ハッシュ照合。
   - 全体サイズが 1,000,000,000 bytes (1 GB) 以下であることを検証。
   - ONNX Runtime Web (Asyncify 版) モジュールの依存検証。
5. **デプロイ**: 上記検証をすべてパスした場合のみ GitHub Pages へ自動公開。

### ローカルでの Pages ビルド検証
```sh
pnpm build:pages
```

---

## 📚 関連ドキュメント

- **[README.md](../README.md)**: プロジェクト概要とライブラリの使用方法
- **[docs/validation.md](validation.md)**: 検証仕様と品質保証（QA）ガイド
