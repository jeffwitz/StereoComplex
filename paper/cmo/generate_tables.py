#!/usr/bin/env python3
"""Compatibility entry point for the current diagnostic article's assets."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent/'analysis/build_diagnostic_assets.py'),run_name='__main__')
