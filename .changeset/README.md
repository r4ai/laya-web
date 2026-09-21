# Changesets

Use [Changesets](https://github.com/changesets/changesets) to record package version bumps and changelog entries.

## When to Add a Changeset

Run `pnpm changeset` whenever you make changes intended for public npm distribution. Select `@r4ai/laya-web`, specify the semver bump type (`major`, `minor`, or `patch`), and write a clear, concise summary of the change.

Commit the generated markdown file alongside your implementation PR.

> Documentation, tooling, and test-only updates do not require a changeset.

## Release Process

1. Merging a PR with changesets into `main` causes the release workflow to create or update a **Version Packages** PR.
2. Merging the Version Packages PR automatically triggers build, packaging, and publishing to npm using GitHub Actions OIDC trusted publishing.

For initial OIDC setup and maintainer workflows, refer to the [Development Guide](../docs/development.md).
