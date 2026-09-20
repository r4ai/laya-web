# 開発ガイド

`laya-web` (リポジトリ名: `hello-jev`) の開発環境構築、モデルエクスポート、ビルド、テスト、および CI/CD デプロイに関する手順である。

## 開発環境構築

### 必要なツール

- **Node.js**: `v22.12.0` 以上
- **パッケージマネージャー**: `pnpm` (v11.x)
- **Python 環境**: [`uv`](https://docs.astral.sh/uv/)（Python 本体および依存ライブラリの自動管理）

### セットアップ手順

```sh
# 1. 依存関係のインストール
pnpm install --frozen-lockfile

# 2. モデルのエクスポート (初回必須)
pnpm model:export

# 3. 開発サーバーの起動
pnpm dev
```

ブラウザで表示された URL（標準: `http://127.0.0.1:5173/`）にアクセスし動作を確認する。

## モデルのエクスポートと管理

Hugging Face 上の MLX チェックポイントを ONNX Runtime Web 用フォーマットに変換して使用する。

### 既定モデル仕様

- **モデル名**: `convaiinnovations/laya-multilingual`
- **リビジョン**: `052592a15d198d9ad47da779604259b10b47b7aa`
- **ダウンロードサイズ**: 約 644 MB
- **変換後出力サイズ**: 約 934 MB (`examples/minimal/public/models/laya/`)

### 出力先の変更と再変換

出力先が既に存在する場合は上書きを防止するためエラーが発生する。
別フォルダへの出力は以下のコマンドで実行する。

```sh
uv run python scripts/export_model.py --output /path/to/new-model
```

別の ModernBERT 系 Laya チェックポイントを指定する場合は `--source` および `--revision` オプションを使用する（検証済みは既定の多言語版のみ）。

## コマンドリファレンス

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
| `pnpm build` | ライブラリ本体のビルド（`dist/` へ TypeScript 型定義と ESM を出力） |
| `pnpm build:demo` | デモアプリのビルド（`examples/minimal/dist/`） |
| `pnpm build:pages` | GitHub Pages 公開用の検証付きビルド |
| `pnpm pack` | ローカル検証用の npm パッケージ（`.tgz`）作成 |

## GitHub Pages デプロイと CI/CD

GitHub Actions により GitHub Pages へ自動デプロイされる（`.github/workflows/pages.yml`）。

### CI/CD パイプラインの処理手順

1. **キャッシュ確認**
   - 変換スクリプトと Python 依存ハッシュをキーに変換済みモデルを取得
2. **モデル変換**
   - キャッシュ未ヒット時のみ Hugging Face からダウンロードしてエクスポート
3. **品質検証**
   - `pnpm typecheck`、`pnpm test`（カバレッジ計測）、`pnpm build:pages` を実行
4. **配信物検査（`validate-pages.mjs`）**
   - 必須ファイル（`index.html`, `model.onnx`, `embeddings.f16.bin` 等）の存在確認
   - モデルの SHA-256 ハッシュ照合
   - 全体サイズの上限検証（1 GB / 1,000,000,000 bytes 以下）
   - ONNX Runtime Web モジュールの依存検証
5. **デプロイ**
   - 全検証パス時のみ GitHub Pages へ自動公開

### ローカルでの Pages ビルド検証

```sh
pnpm build:pages
```

## 関連ドキュメント

- [README.md](../README.md): プロジェクト概要とライブラリの使用方法
- [docs/validation.md](validation.md): 検証仕様と品質保証ガイド
