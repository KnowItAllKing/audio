# Security Policy

This project handles live meeting audio and transcripts, so dependency and
credential hygiene are part of the product design, not cleanup work.

## Dependency Rules

- npm dependencies are managed with `pnpm@11.7.0`.
- npm dependency resolution must enforce a minimum package age of 7 days
  (`minimumReleaseAge: 10080`) for direct and transitive dependencies.
- npm resolution must fail when registry publish times are missing
  (`minimumReleaseAgeIgnoreMissingTime: false`) or no age-compliant package
  version exists (`minimumReleaseAgeStrict: true`).
- npm installs must keep lockfile supply-chain age verification enabled
  (`trustLockfile: false`). pnpm's provenance `trustPolicy` is not enabled by
  default because current common packages can fail it for old transitive
  dependencies; prefer audits, age gating, no install scripts, and exact pins
  before adding provenance exceptions.
- npm peer dependencies must be declared intentionally
  (`autoInstallPeers: false`).
- npm dependency lifecycle/build scripts must not run during install
  (`ignoreScripts: true`). Packages that require install-time code should be
  avoided, replaced, or handled with an explicit manual binary/runtime setup.
- Electron runtime binaries must be treated as explicit local tooling. Do not
  re-enable Electron's dependency install script; use checksum-verified manual
  setup or a VM-local `ELECTRON_OVERRIDE_DIST_PATH` when a downloaded app bundle
  cannot run from a shared Tart volume.
- Python dependencies are managed with `uv`; PyPI resolution uses
  `exclude-newer = "7 days"` and the default `first-index` strategy to reduce
  dependency-confusion risk if additional indexes are introduced.
- Python source builds are currently allowed because Whisper/WebRTC VAD related
  packages need them. New source-build-only packages require a note in the PR or
  ticket explaining the package, build path, and local-only risk.

Security hotfix exception: a dependency can bypass the 7-day age rule only when
it fixes a known vulnerability or operational blocker, is pinned intentionally,
and the exception is documented with the package name, version, reason, and
planned removal date. This exception does not allow dependency install scripts.

## Verification

Run before dependency or security-sensitive changes:

```bash
make server-test
make security-audit
```

Run before frontend changes once the client typecheck baseline is clean:

```bash
make client-typecheck
```

## Secrets And Zoom Credentials

- Never commit `.env`, OAuth client secrets, access tokens, refresh tokens,
  private keys, Zoom webhook secrets, or RTMS secrets.
- Store Zoom credentials only in local environment variables or an OS secret
  store.
- Request the smallest Zoom scopes needed for the selected integration path.
- Prefer private/development Zoom apps for personal/local use. Public or
  cross-account distribution should go through Zoom review.
- Log token presence only as booleans; never log token values.

## Audio And Transcript Privacy

- Default to local processing and local transcript storage.
- Do not persist raw meeting audio unless the user explicitly enables it.
- Treat saved transcripts as sensitive local files.
- If a Zoom integration is added, make participant names and meeting IDs
  opt-in for persistence and avoid logging them at info level.

## Network Defaults

- Local development should bind to localhost when possible.
- Cross-device use should rely on Tailscale or TLS plus explicit authentication.
- Do not expose the websocket server directly to the public internet without
  auth, TLS, rate limits, and request-size limits.
