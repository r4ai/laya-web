# Development Guide

Engineering guide for developing, building, exporting models, testing, and deploying `@r4ai/laya-web`.

## Prerequisites

- **Node.js**: `24.x` (the CI runtime)
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

| Command             | Description                                                          |
| :------------------ | :------------------------------------------------------------------- |
| `pnpm lint`         | Check JavaScript and TypeScript with oxlint; warnings fail CI        |
| `pnpm lint:fix`     | Apply safe oxlint fixes                                              |
| `pnpm format:check` | Check formatting with oxfmt                                          |
| `pnpm format`       | Format supported source, configuration, and documentation files      |
| `pnpm typecheck`    | Run TypeScript compiler checks without emitting files                |
| `pnpm test`         | Run Vitest unit and integration suites with V8 coverage              |
| `uv run pytest`     | Validate logit parity between PyTorch and ONNX models                |
| `pnpm model:verify` | Compare exported model outputs against MLX FP32 CPU reference values |
| `pnpm test:browser` | Launch browser test harness for WebGPU and Wasm verification         |

### Build and Distribution

| Command            | Description                                                                   |
| :----------------- | :---------------------------------------------------------------------------- |
| `pnpm build`       | Compile library source to `dist/` with ESM bundles and TypeScript definitions |
| `pnpm build:demo`  | Build the SolidJS demo application in `examples/minimal/dist/`                |
| `pnpm build:pages` | Build the GitHub Pages distribution with automated asset validation           |
| `pnpm pack`        | Package library into a `.tgz` archive for local npm verification              |

## Continuous Integration

`.github/workflows/ci.yml` runs on pull requests and manual dispatches. The release
workflow also calls it for every `main` push before any versioning or publication.
It checks oxlint, oxfmt, TypeScript, Vitest with V8 coverage, the library build,
package contents, and Python's synthetic PyTorch/ONNX parity test. Dependencies
are installed from frozen pnpm and uv lockfiles.

The model-dependent tokenizer parity test is skipped when exported fixtures are
absent, as on clean PR runners. The Pages workflow exports the pinned model first
and runs that test with real fixtures. Browser/WebGPU and MLX reference checks
remain separate manual acceptance steps. V8 coverage is reported; this setup does
not impose a new percentage threshold.

Generated models, build output, coverage, local learning artifacts, and dependency
patches are excluded from formatting. pnpm owns the lockfile's formatting.

## npm Releases with Changesets and OIDC

For a publishable change, run `pnpm changeset`, choose `@r4ai/laya-web` and a bump
type, and commit the generated Markdown file with the implementation. Tooling-only
changes do not require a changeset.

`.github/workflows/release.yml` runs the checks, then selects one transition:

| State on `main`                                   | Result                                                                     |
| :------------------------------------------------ | :------------------------------------------------------------------------- |
| Any required check fails                          | No version PR, package upload, or release                                  |
| Pending changesets                                | Create/update `chore: release packages` with versions and changelog        |
| No changesets and an unpublished version          | Build and pack, then publish the tarball and create the GitHub release/tag |
| No changesets and every version already published | No new package publication                                                 |
| Manual dispatch on another branch                 | Run checks only                                                            |

Merge the version PR to publish. Versioning refreshes the lockfile and formats
generated files. Build/pack jobs have read access; only the publish job receives
`id-token: write`, with the `npm` environment. Changesets v3 invokes the pinned
pnpm 11 CLI, which supports npm trusted publishing natively. No `NPM_TOKEN` or
`NODE_AUTH_TOKEN` secret is used. Concurrent releases are serialized.

### One-time Registry and Repository Setup

1. The npm package must exist before its trusted publisher can be configured.
   If `@r4ai/laya-web` is still unpublished, a maintainer must bootstrap the current
   `0.1.0` using interactive npm authentication and 2FA: run `npm login`, then
   `pnpm install --frozen-lockfile`, the local quality checks, and
   `npm publish --access public`. The `prepack` hook builds the library. This is a
   one-time account operation; subsequent releases use OIDC. The included initial
   changeset prepares `0.1.1` through the version PR.
2. In npm package settings, add a GitHub Actions trusted publisher with these
   exact values. Enable direct publishing (`npm publish`); stage-only permission
   does not work with this Changesets flow.

   | Field                | Value         |
   | :------------------- | :------------ |
   | Organization or user | `r4ai`        |
   | Repository           | `laya-web`    |
   | Workflow filename    | `release.yml` |
   | Environment          | `npm`         |

3. In GitHub repository **Settings → Actions → General**, enable **Allow GitHub
   Actions to create and approve pull requests**. The version job requests its
   own `contents: write` and `pull-requests: write` permissions.
4. Create the GitHub environment `npm` and restrict its deployment branch to
   `main`. Keep the same environment name in npm's trusted publisher settings.

PRs created with `GITHUB_TOKEN` do not automatically trigger PR workflows. Close
and reopen the generated version PR as a maintainer to run PR checks if required
by branch protection. Every merge to `main` is independently checked again before
publishing.

See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/) and
[Changesets automation](https://changesets.dev/guide/automating) for provider setup.

## GitHub Pages

Automated workflows test, validate, and deploy the demo application to GitHub Pages on pushes to `main` (`.github/workflows/pages.yml`).

### Workflow Steps

1. **Cache Verification**
   - Restores converted model artifacts using cache keys derived from Python dependencies and export scripts
2. **Model Conversion**
   - Downloads checkpoint from Hugging Face and exports ONNX assets only on cache miss
3. **Quality Validation**
   - Runs `pnpm lint`, `pnpm format:check`, `uv run --frozen pytest`, `pnpm typecheck`, and `pnpm test` with V8 coverage reporting
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
