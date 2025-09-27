#!/bin/bash
# 
# Generic FreeCAD Python runner script
#
# This script runs Python commands within the FreeCAD environment,
# providing access to FreeCAD modules while supporting standard Python patterns.
#
# Usage examples:
#   ./freecad_python.sh                                    # Interactive REPL
#   ./freecad_python.sh -m pytest tests/unit/ -v          # Run pytest
#   ./freecad_python.sh -c "import FreeCAD; print('OK')"  # Execute code
#   ./freecad_python.sh my_script.py arg1 arg2            # Run script
#   ./freecad_python.sh -m shellforgepy.simple             # Run module
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Join all arguments into a single string for the environment variable
if [ $# -eq 0 ]; then
    # No arguments - will start interactive REPL
    PYTHON_ARGS=""
    echo "Starting FreeCAD interactive Python REPL..."
else
    # Join all arguments with proper shell quoting
    PYTHON_ARGS=""
    for arg in "$@"; do
        if [ -n "$PYTHON_ARGS" ]; then
            PYTHON_ARGS="$PYTHON_ARGS "
        fi
        # Use printf %q for proper shell quoting
        PYTHON_ARGS="$PYTHON_ARGS$(printf '%q' "$arg")"
    done
    
    echo "Running FreeCAD Python with: $*"
fi

echo "=================================="

# Set environment variable for the FCMacro script
export FREECAD_PYTHON_ARGS="$PYTHON_ARGS"

# Run the FCMacro using the FreeCAD command line
# Note: We use exit() to ensure clean shutdown
echo "exit()" | /Applications/FreeCAD.app/Contents/Resources/bin/freecad -c "$SCRIPT_DIR/docker/freecad_python.FCMacro"

echo "=================================="
echo "FreeCAD Python session completed"