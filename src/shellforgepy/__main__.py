"""Entry point for running shellforgepy as a module."""

import sys

from shellforgepy.workflow.workflow import main

if __name__ == "__main__":
    sys.exit(main())
