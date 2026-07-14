"""Shared low-level capture/data utilities for the standalone sparse-KV experiments.

This experiment is independent of ``query_forecast``.  The low-level Qwen3
adapter is shared with the sibling pre-experiment so fixes stay consistent.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from future_query_union_frontier.common import *  # noqa: F401,F403,E402
