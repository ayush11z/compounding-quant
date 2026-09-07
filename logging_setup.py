"""Single module for all logging configuration call sites (per CLAUDE.md tech constraints).

Named logging_setup.py (not logging.py or logging/) deliberately -- a top-level module or
package named `logging` shadows the stdlib module for every other import in this project.
"""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
