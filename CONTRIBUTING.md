# Contributing

Thanks for improving the project. Keep changes focused, reproducible, and
explicit about whether evidence comes from a development sample or a full-data
run.

## Development setup

Python 3.11 is required. From the repository root:

```bash
make setup
make check-fast
```

Use `CONFIG=configs/full.toml` only when intentionally running the full-data
profile. Generated datasets, model binaries, indexes, reports, and secrets must
not be committed.

## Before opening a pull request

```bash
make lint
make test-unit
make test-integration
```

Infrastructure tests are opt-in because they need live services:

```bash
make test-redis
make test-triton
```

Document new configuration values and externally visible behavior. Add or
update tests for bug fixes and keep model/evaluation claims tied to a generated
report or a reproducible command.

When `pyproject.toml` dependencies change, regenerate both pinned environments:

```bash
make lock
```

## Commit and pull-request scope

- Prefer one logical change per commit.
- Do not hide negative or inconclusive experiment results.
- Call out compatibility, data-contract, or metric changes explicitly.
- Never include MovieLens data, credentials, or generated model artifacts.
