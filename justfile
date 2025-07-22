tidy:
    # uses the pyproject.toml settings
    uv run ruff check --preview . --fix
    uv run ruff check --preview --fix --select=ANN001,UP035,F401,D213,I001 src/
    uv run ruff format src/ tests/

test:
    uv run pytest tests/ -v -x

test-all:
    uv run pytest tests/ -v

clean:
    # Remove Numba compilation artifacts and Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".numba_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    rm -rf .pytest_cache 2>/dev/null || true
    echo "Cleaned Numba and Python cache files"
