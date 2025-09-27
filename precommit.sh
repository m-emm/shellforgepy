#! /bin/bash

# Format code
isort  $(find src -name '*.py' )  ; black  $(find src  -name '*.py' )  ; isort  $(find tests -name '*.py') ; black  $(find tests -name '*.py')

# # Format workflow files
# npx prettier --write  .github/workflows/*.yml

# Run linting (same as GitHub Actions)
echo "Running flake8 linting (syntax errors and undefined names only)..."
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --ignore=F824 --show-source --statistics


echo "Checking for unused imports..."
flake8 --select=F401 --exclude="src/*/simple.py" --exclude "build/*" --exclude "lambda_runtimes/*" src/
if [ $? -ne 0 ]; then
    echo "❌ Found unused imports! Please remove them before committing."
    exit 1
fi
echo "✅ No unused imports found."

