# Changesets

Guidance for recording package version bumps and changelog entries via [Changesets](https://github.com/changesets/changesets).

## When to Add a Changeset

Run `pnpm changeset` for any user-facing change intended for npm distribution:

1. Select `@r4ai/laya-web`
2. Specify the semver bump type (`major`, `minor`, or `patch`)
3. Write a concise summary of the change
4. Commit the generated `.changeset/*.md` file with your pull request

> [!NOTE]
> Tooling, test, and documentation-only updates do not require a changeset.

## Release Lifecycle

1. Merging a pull request with changesets into `main` creates or updates a **Version Packages** PR
2. Merging the Version Packages PR triggers automated build, packaging, and publishing to npm via GitHub Actions OIDC trusted publishing

For initial OIDC configuration and maintainer steps, refer to the [Development Guide](../docs/development.md).
