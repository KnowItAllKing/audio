# Tailscale test procedure (Phase 2)

This phase verifies that the **same WebSocket pipeline** works over a Tailscale network with **no code changes** (only configuration).

## Assumptions

- Tailscale is installed on both the **server machine** and the **client machine**
- You will rely on **Tailscale network security** (no app-level auth yet)

## Steps

1. **Install & authenticate Tailscale** on both machines.
2. **Get the server’s Tailscale hostname or IP** (from the Tailscale admin console or `tailscale status`).
3. On the **server machine**, start the WebSocket server:

```bash
cd /path/to/audio
uv sync --project server
WS_PORT=8765 PYTHONPATH=.. uv run --project server python -m server.main
```

4. On the **client machine**, run the test client pointing at the server’s Tailscale hostname:

```bash
cd /path/to/audio/client
pnpm install
WS_URL=ws://<server-tailnet-hostname>:8765 pnpm dev:test-client
```

5. **Verify** you see periodic `transcript_update` logs on the client.

