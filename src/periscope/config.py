"""Re-export config from project root.

Config lives at project root (config.py next to pyproject.toml).
This stub adds the project root to sys.path and re-exports so
`from periscope.config import ...` and `from periscope import config`
continue to work.
"""

import sys
from pathlib import Path

# Project root = parent of src (parent.parent of this file)
_ROOT = Path(__file__).resolve().parent.parent.parent
if _ROOT not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import all names from root config (no periscope.* deps in root config)
from config import *  # noqa: F401, F403

