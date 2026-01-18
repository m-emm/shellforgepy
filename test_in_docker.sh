#!/bin/bash
# Test shellforgepy in a clean Docker container with CadQuery
# Uses x86_64 platform because CadQuery only has wheels for that architecture

set -euo pipefail
trap 'echo "Script $0 failed at line $LINENO" >&2' ERR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_FULLPATH="$(realpath "$0")"

echo "Running ${SCRIPT_FULLPATH} with args: $@"

# Docker image - standard Python, not Lambda (need system libs for CadQuery/OCP)
DOCKER_IMAGE="python:3.12-slim"

# Test file pattern - default to all cadquery adapter tests
TEST_PATTERN="${1:-tests/unit/adapters/cadquery/}"

# Create temp directory under $HOME (required for macOS Lima docker mounts)
TEMP_DIR="$HOME/tmp/shellforgepy_docker_test_$$"
mkdir -p "$TEMP_DIR"

# Create the test script that will run inside the container
cat <<'EOF' > "$TEMP_DIR/run_tests.sh"
#!/bin/bash
set -e

echo "=== Installing system dependencies for CadQuery/OCP ==="
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0 libxrender1 libxmu6 libxi6 > /dev/null 2>&1

echo "=== Installing Python packages ==="
cd /mnt/shellforgepy
pip install -q pytest -e .[cadquery]

echo "=== Verifying CadQuery import ==="
python -c "import cadquery; print(f'CadQuery {cadquery.__version__} loaded successfully')"

echo "=== Running tests ==="
python -m pytest "$@" -v
EOF

chmod +x "$TEMP_DIR/run_tests.sh"

echo "=== Starting Docker container (platform: linux/amd64) ==="
docker run --platform linux/amd64 \
    --rm \
    -e DISABLE_SETUPTOOLS_SCM=1 \
    -v "${SCRIPT_DIR}:/mnt/shellforgepy" \
    -v "${TEMP_DIR}:/mnt/scripts" \
    "$DOCKER_IMAGE" \
    /bin/bash /mnt/scripts/run_tests.sh "$TEST_PATTERN"

# Cleanup
rm -rf "$TEMP_DIR"

echo "=== Docker test completed successfully ==="
