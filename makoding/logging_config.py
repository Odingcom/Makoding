"""Application-wide logging configuration.

Streamlit apps are easy to leave un-instrumented; this gives DataLab Pro a
single, predictable place to control log verbosity (useful once deployed
behind a paywall or in a customer's environment, where you'll need logs to
debug support tickets without reproducing on your own machine).
"""
from __future__ import annotations

import logging
import sys

from makoding.config import LOG_LEVEL

_CONFIGURED = False


def setup_logging() -> logging.Logger:
    """Idempotently configure and return the DataLab Pro root logger."""
    global _CONFIGURED
    logger = logging.getLogger("datalab")

    if not _CONFIGURED:
        logger.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True

    return logger