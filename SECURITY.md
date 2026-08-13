# Security Policy

## Supported Scope

This repository is an early-stage project. Security fixes are handled on a best-effort basis.

Current priority areas:
- authentication bypasses
- data corruption or unsafe persistence behavior
- denial-of-service vectors in public HTTP endpoints

## Reporting

If you discover a security issue, do not open a public issue with exploit details.

Report privately to the maintainers through the channel you use to manage this project, including:
- affected version or commit
- reproduction steps
- impact assessment
- suggested mitigation if available

## Hardening Notes

For public deployments:
- enable `server.api_key`
- place the service behind a reverse proxy
- restrict network exposure when possible
- persist `data/` on durable storage
- monitor `/metrics` and HTTP error rates

