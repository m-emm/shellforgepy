"""Entry point for running shellforgepy as a module."""

import logging
import sys

from shellforgepy.builder.errors import BuilderError
from shellforgepy.workflow.workflow import WorkflowError, main

_logger = logging.getLogger(__name__)


def _main() -> int:
    try:
        return main()
    except (BuilderError, WorkflowError) as exc:
        _logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
