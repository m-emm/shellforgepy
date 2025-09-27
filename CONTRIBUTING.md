# Contributing

Contributions are welcome! Here's how you can help:

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone <your-fork-url>
   cd shellforgepy
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[testing]"
   ```

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest
   ```

4. Format your code:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

5. Commit and push your changes

6. Create a Pull Request

## Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Follow PEP 8 guidelines
- Write tests for new functionality

## Testing

Run the test suite with:

```bash
pytest
```

For coverage reporting:

```bash
pytest --cov=shellforgepy --cov-report=html
```
