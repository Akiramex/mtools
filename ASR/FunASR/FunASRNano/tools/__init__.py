"""Compatibility shim for FunASR Nano.

Some FunASR releases import submodules via `from tools...` (non-package import).
Providing a local `tools` package makes those imports work when running from
this repo root.
"""
