# archive/python

The original Python implementation of proxycache (FastAPI/uvicorn). **Reference only** — not maintained, not run in production.

The Go implementation at the repo root is canonical. This copy is preserved to document intended behavior while porting decisions diverged (e.g., recompute threshold: Go uses 0.7, Python used 0.92).

If you need to run it: `uv sync && uv run python proxycache.py` with the Windows Python toolchain. Note `test_smoke.py` uses unsanitized backend keys (with colons); the Go tests use `SanitizeBackendDir()` output.
