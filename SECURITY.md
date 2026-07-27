# Security policy

## Supported versions

Powerhouse AI is an experimental `0.x` project. Security fixes are applied to the latest release
only.

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private vulnerability reporting for
this repository. Include the affected version, reproduction steps, impact, and any suggested
mitigation. Please allow a reasonable remediation window before public disclosure.

## Security boundary

- The built-in API binds only to `127.0.0.1` and has no authentication. Do not expose it through a
  proxy, port-forward, tunnel, or custom ASGI launch without adding authentication and transport
  security.
- Project messages, memory, indexed document text, and local source paths are stored unencrypted in
  SQLite. Protect the workstation account and disk accordingly.
- Python execution is disabled by default. Enabling it permits generated trusted code to access the
  current user's files and network. The timeout and Python isolation flags are not a hostile-code
  sandbox.
- Retrieved documents and project memory are treated as untrusted data in prompts, but model-level
  prompt-injection defenses are not a complete security boundary.
- Runtime and model installers verify pinned SHA-256 digests before use.

See [the architecture document](docs/ARCHITECTURE.md) for the implemented runtime boundary.
