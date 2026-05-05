"""Module entry for `python -m src.batch ...`.

Loads `.env` BEFORE importing the CLI (which transitively imports
`transformers` / `huggingface_hub`). Those libraries snapshot env vars
like `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` at import time — the
same reason `main.py` calls `_bootstrap_env()` at the top.
"""
import sys
from pathlib import Path

# Make the project root importable so we can re-use the bootstrap helper
# from main.py without duplicating it.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import _bootstrap_env  # noqa: E402

_bootstrap_env()

# Safe now — env vars are populated.
from src.batch.cli import main  # noqa: E402

sys.exit(main())
