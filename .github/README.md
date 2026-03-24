# CI Overview

The workflow files are the source of truth for exact job definitions. This file
only documents the parts that are easy to miss when reading the YAML in
isolation: workflow coupling, stable check names, and tag protection.

## Workflows

- `tests.yml`: normal source validation
- `docs-check.yml`: PR docs validation
- `gh-pages.yml`: docs rebuild + Pages deploy from tested `main` commits
- `release.yml`: PyPI publishing

Docs building itself is centralized in `.github/actions/build-docs/` and reused
by `docs-check.yml` and `gh-pages.yml`.

## Stable Checks

Two check names are intentionally treated as stable interfaces for repository
rules:

- `tests-pass`
- `docs-pass`

`tests-pass` comes from `tests.yml`.

`docs-pass` is intentionally produced by two workflows:

- `docs-check.yml` on pull requests
- `gh-pages.yml` on tested commits on `main`

That duplication is deliberate: PRs need a docs gate before merge, while tag
protection needs the same check name on the merged commit itself.

## How The Workflows Connect

### Pull requests

For a PR to `main`:

1. `tests.yml` runs and emits `tests-pass`.
2. `docs-check.yml` runs and emits `docs-pass`.

### Commits on `main`

After a commit lands on `main`:

1. `tests.yml` runs.
2. If it succeeds, `gh-pages.yml` is triggered via `workflow_run`.
3. `gh-pages.yml` checks out the tested SHA, rebuilds docs, emits `docs-pass`,
   and deploys GitHub Pages.

### Releases

Workflow:

1. Prepare a release locally, e.g. with `bumpversion`.
2. Push the version-bump commit to `main`.
3. Wait for `tests-pass` and `docs-pass` on that commit.
4. Push the release tag (`git push --tags`)
5. Publish a GitHub Release in the GitHub UI.
6. `release.yml` publishes to PyPI.

`release.yml` is intentionally not a full source-validation workflow. It assumes
that the tagged commit was already validated by normal CI and by the tag
rulesets.

## Tag Rulesets

The repository currently has two active tag rulesets for tags matching
`refs/tags/*.*.*`:

- one requires `tests-pass`
- one requires `docs-pass`

Both also forbid:

- deletion
- non-fast-forward updates

Practical effect:

- a release tag can only be pushed if the pointed-to commit already has both
  stable checks: `tests-pass` and `docs-pass`

## Repo Settings That Matter

- GitHub Pages must use `Source = GitHub Actions`, not branch-based publishing
- the `github-pages` environment must allow deployment from `main`
- the `pypi` environment must exist
- PyPI Trusted Publishing must be configured for this repo and `release.yml`

## Maintenance Notes

- Treat `tests-pass` and `docs-pass` as stable API-like names.
- If either name changes, update the rulesets too.
- If docs deploy starts failing with a complaint about `gh-pages`, check the
  repository Pages source setting first.
- If PyPI publishing fails with an OIDC / Trusted Publisher error, verify the
  PyPI-side Trusted Publisher settings for `release.yml` and environment
  `pypi`.
