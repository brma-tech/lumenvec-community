# Contributing

## Contribution Flow

- Do not open pull requests directly from `main`.
- Start from an up-to-date local `main`, create a feature branch, and work from that branch.
- Open the PR from your feature branch into `main`.

Recommended flow:

```bash
git checkout main
git pull origin main
git checkout -b feature/short-description
```

## Development Setup

1. Install Go `1.23+`.
2. Clone the repository.
3. Run:

```bash
go mod tidy
go test ./...
```

## Project Conventions

- Keep changes focused and small.
- Prefer preserving the current API unless the change explicitly targets API evolution.
- Update `README.md` when behavior, configuration, or public endpoints change.
- Update example configs when new config fields or security controls are introduced.
- Add or update tests for behavior changes.

## Before Opening a PR

Run:

```bash
go test ./...
go vet ./...
go run ./tools/checkcoverage
```

If you change core search or ingest behavior, also run:

```bash
go test ./internal/core -bench . -benchmem
```

If you change transport, security, or persistence behavior, also run:

```bash
go test -race ./internal/core ./internal/api ./pkg/client
golangci-lint run --timeout=5m
```

## Coverage Policy

Production packages must keep at least `90%` statement coverage.

The enforced package set is checked by:

```bash
go run ./tools/checkcoverage
```

Examples and integration-only packages are not part of this threshold.

## Pull Requests

- Branch from a feature branch, not directly from `main`.
- Describe the problem and the chosen approach.
- Mention any API, config, persistence, or benchmark impact.
- Mention any security impact and whether the change affects `development` and `production` profiles differently.
- Keep unrelated refactors out of the same PR.
