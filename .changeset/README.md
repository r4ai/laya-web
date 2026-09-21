# Changesets

Run `pnpm changeset` for changes that should ship to npm, select
`@r4ai/laya-web`, and describe the user-visible change.

After merging to `main`, the release workflow opens or updates a version PR.
Merge that PR to publish with npm trusted publishing (OIDC).

Tooling-only changes do not need a version bump. See the
[development guide](../docs/development.md) for initial setup and validation.
