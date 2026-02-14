"""Compatibility shim for FunASR Nano.

Some FunASR releases import `CTC` via `from ctc import CTC` (non-package import).
When running from this repo root, providing this module makes that import work.
"""

from funasr.models.fun_asr_nano.ctc import CTC

__all__ = ["CTC"]
