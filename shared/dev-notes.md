# Dev Notes (Phase 0)

## Language defaults

- **Client**: TypeScript (Electron app)
- **Server**: Python **3.11+** (or latest stable available in the environment)

## Dependency management defaults

### Python

**uv** is the default for the server environment and dependency management.

```bash
# Install uv (see: https://docs.astral.sh/uv/)
# From repo root:
uv sync --project server
uv run --project server python -m server.main
```

### Node / Electron

**pnpm** is the default for this monorepo.

Monorepo tooling:

- **pnpm workspaces** (`pnpm-workspace.yaml`)
- **Turborepo** (`turbo.json`) for running tasks across packages

Minimal setup (once Node is installed):

```bash
pnpm install
```

## Deployment note

Planned deployment connectivity uses **Tailscale** between client and server. The protocol remains plain WebSockets so transport/security can be layered independently (Tailscale, TLS, auth).
