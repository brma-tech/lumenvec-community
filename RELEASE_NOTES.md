# LumenVec Community v0.3.0-rc.1

Release candidate prepared on 2026-09-28. This is a prerelease for evaluation.

- Hierarchical HNSW backend, graph construction and search instrumentation.
- Persistence synchronization, cancellation, deleted-vector filtering and ANN metrics fixes.
- Streaming rebuild and query overhead improvements.
- Updated official logo and corrected standalone Community Docker packaging.
- gRPC 1.83.2 and updated x/net, x/sys and x/text dependencies to address the reachable vulnerabilities found during release preparation.

Validation: isolated Community Go tests and go vet passed. Business-only integration tests are kept in the private edition. No proprietary runtime packages are included in the Community dependency graph. Dependency versions already merged upstream are preserved.

Known limits: recall and memory remain workload-dependent; no claim of parity with pgvector is made. Kubernetes, Docker runtime and upgrade/rollback compatibility are not certified by the unit suite. Back up snapshot/WAL and configuration before evaluating this candidate. Rollback using the previous binary and a pre-upgrade copy of the data.

Assets contain the standalone server for Windows and Linux amd64. Configure through VECTOR_DB_* environment variables or an explicit YAML configuration. HTTP defaults to port 19190. Set an API key and TLS before exposing it beyond a trusted test environment.
