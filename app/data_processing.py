"""Backward-compatible imports for legacy tests and scripts.

The implementation lives in app.data.data_processing. New code should import
from that module directly.
"""

from app.data.data_processing import *  # noqa: F401,F403
