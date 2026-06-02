"""Entry point for running shellforgepy as a module."""

import sys

from shellforgepy.builder.errors import BuilderError
from shellforgepy.workflow.workflow import WorkflowError, main


def _main() -> int:
    try:
        return main()
    except (BuilderError, WorkflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
