# Development Guide

Engineering guide for developing, building, exporting models, testing, and deploying `@r4ai/laya-web`.

## Prerequisites

- **Node.js**: `v22.12.0` or higher
- **Package Manager**: `pnpm` (v11.x)
- **Python Environment**: [`uv`](https://docs.astral.sh/uv/) for Python and virtual environment management

> [!NOTE]
> Running `pnpm model:verify` requires an Apple Silicon Mac to execute the MLX reference implementation. Model export via `pnpm model:export` and PyTorch tests via `uv run pytest` run on standard CPU architectures.

## Quickstart

```sh
# 1. Clone repository and install dependencies
git clone https://github.com/r4ai/laya-web.git
cd laya-web
pnpm install --frozen-lockfile

# 2. Export base model checkpoint (required on initial setup)
pnpm model:export

# 3. Start local development server
pnpm dev
```

Navigate to `http://127.0.0.1:5173/` in your browser to inspect the application.

## Model Export & Management

The exporter converts Hugging Face Laya checkpoints into partitioned assets optimized for browser streaming and execution.

### Default Checkpoint Specification

- **Model ID**: `convaiinnovations/laya-multilingual`
- **Revision**: `052592a15d198d9ad47da779604259b10b47b7aa`
- **Download Size**: ~644 MB
- **Exported Output Size**: ~934 MB (`examples/minimal/public/models/laya/`)

### Export Customization

To prevent accidental overwrites, the export script aborts if the destination directory already exists.

To export to an alternate directory:

```sh
uv run python scripts/export_model.py --output /path/to/custom-model
```

Use `--source` and `--revision` to specify alternative ModernBERT-based Laya checkpoints. Official validation targets the default multilingual checkpoint.

## Command Reference

### Development and Quality Assurance

| Command | Description |
| :--- | :--- |
| `pnpm typecheck` | Run TypeScript compiler checks without emitting files |
| `pnpm test` | Run Vitest unit and integration suites with V8 coverage |
| `uv run pytest` | Validate logit parity between PyTorch and ONNX models |
| `pnpm model:verify` | Compare exported model outputs against MLX FP32 CPU reference values |
| `pnpm test:browser` | Launch browser test harness for WebGPU and Wasm verification |

### Build and Distribution

| Command | Description |
| :--- | :--- |
| `pnpm build` | Compile library source to `dist/` with ESM bundles and TypeScript definitions |
| `pnpm build:demo` | Build the SolidJS demo application in `examples/minimal/dist/` |
| `pnpm build:pages` | Build the GitHub Pages distribution with automated asset validation |
| `pnpm pack` | Package library into a `.tgz` archive for local npm verification |

## CI/CD Pipeline & GitHub Pages

Automated workflows test, validate, and deploy the demo application to GitHub Pages on pushes to `main` (`.github/workflows/pages.yml`).

### Workflow Steps

1. **Cache Verification**
   - Restores converted model artifacts using cache keys derived from Python dependencies and export scripts
2. **Model Conversion**
   - Downloads checkpoint from Hugging Face and exports ONNX assets only on cache miss
3. **Quality Validation**
   - Runs `uv run --frozen pytest`, `pnpm typecheck`, and `pnpm test` with code coverage thresholds
4. **Distribution Asset Inspection (`scripts/validate-pages.mjs`)**
   - Validates existence of required entrypoints (`index.html`, `model.onnx`, `embeddings.f16.bin`)
   - Verifies SHA-256 integrity hashes against `config.json` manifests
   - Enforces overall distribution size limit of 1,000,000,000 bytes (1 GB)
   - Verifies presence of all ONNX Runtime Web helper binaries required by the application bundle
5. **Deployment**
   - Publishes verified assets to GitHub Pages

### Validating Pages Build Locally

Run the complete build and verification pipeline locally:

```sh
pnpm build:pages
```

## Related Documents

- [README.md](../README.md): Project overview and client API guide
- [docs/validation.md](validation.md): Verification specifications and test matrices
