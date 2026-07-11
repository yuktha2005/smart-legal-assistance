import os
import sys

# Ensure repository modules can be imported regardless of how pytest is launched.
# Tests expect imports like: `from src...` and `from config.config...`
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Smart_Legal_Assistance/
REPO_ROOT = os.path.dirname(BASE_DIR)  # d:/Skylena/SLA

# Put repo root first so top-level compatibility package `config/` wins.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Also add Smart_Legal_Assistance so `import src` and `from config.config ...` work.
# NOTE: Do NOT add config/ or src/ subdirs directly — adding config/ causes
# config.py to shadow the config *package*, breaking `from config.config import`.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Final guard: if something already imported `config` as a non-package, reset it.
mod = sys.modules.get("config")
if mod is not None and not hasattr(mod, "__path__"):
    sys.modules.pop("config", None)



