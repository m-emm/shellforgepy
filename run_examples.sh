#!/bin/bash

# Run all examples to ensure they work
# This script runs each example and reports success/failure

set -e  # Exit on any error

echo "🚀 Running all ShellForgePy examples..."
echo "========================================="

# Change to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Track success/failure
TOTAL=0
PASSED=0
FAILED=0

# Function to run an example
run_example() {
    local example_file="$1"
    local example_name=$(basename "$example_file" .py)
    
    echo ""
    echo "📝 Running: $example_name"
    echo "----------------------------"
    
    TOTAL=$((TOTAL + 1))
    
    if python "examples/$example_file"; then
        echo "✅ $example_name - SUCCESS"
        PASSED=$((PASSED + 1))
    else
        echo "❌ $example_name - FAILED"
        FAILED=$((FAILED + 1))
    fi
}

# Run each example in order (beginner to advanced)
echo ""
echo "🔰 Beginner Examples:"
run_example "filleted_boxes_example.py"
run_example "create_cylinder_stl.py"

echo ""
echo "🔥 Path-Following Examples:"
run_example "straight_snake.py"
run_example "curved_snake.py"
run_example "cylindrical_coil.py"
run_example "conical_coil.py"
run_example "mobius_strip.py"

echo ""
echo "🧠 Advanced Examples:"
run_example "bottle_cap_example.py"
run_example "create_face_stl.py"

# Summary
echo ""
echo "========================================="
echo "📊 SUMMARY"
echo "========================================="
echo "Total examples: $TOTAL"
echo "Passed: $PASSED ✅"
echo "Failed: $FAILED ❌"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🎉 All examples ran successfully!"
    exit 0
else
    echo ""
    echo "💥 Some examples failed. Please check the output above."
    exit 1
fi