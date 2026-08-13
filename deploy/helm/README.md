# Helm deployment

Install with `helm upgrade --install lumenvec ./deploy/helm/lumenvec`.
The chart uses a non-root StatefulSet, persistent volume, probes, and a
Secret-backed API key. Set `security.apiKey` through an external values file
or secret manager for production deployments.

Set `profiling.addr` only on an isolated diagnostic deployment; it is empty
by default and must not be exposed through a public Service.

Run the private end-to-end installation check against the current Kubernetes
context with:

```powershell
pwsh -File scripts/run-community-helm-smoke.ps1
```

The check installs into a temporary namespace, waits for the StatefulSet,
verifies health, inserts and fetches a vector, validates search, and removes
the release and namespace. For a `kind` context, it loads the local image into
the selected cluster automatically.
