# OpenCode.md for CodeRunner

## Build, Lint, and Test Commands
- **Install dependencies:** `pip install -r examples/requirements.txt`
- **Install main requirements:** `pip install -r requirements.txt`
- **Run main server:** `python server.py`
- **No test suite detected** (no pytest/unittest found; add tests in `tests/` or `test_*.py`)

## Code Style Guide
- Follow [PEP8](https://peps.python.org/pep-0008/) for Python code.
- Use meaningful, descriptive names for variables, functions, and classes.
- Prefer absolute imports; group stdlib, third-party, then local imports.
- Use type annotations for all public functions.
- Use async/await for async flows where possible.
- Log errors using `logging` (not print), unless interactive CLI.
- Include basic docstrings for all public functions/classes.
- Add comments for complex logic, but avoid redundant comments.
- Write atomic, focused commits with clear messages.
- Add tests for new features (see CONTRIBUTING.md).
- Prefer pathlib for filesystem paths.
- File uploads/storage: use `/app/uploads` as shared dir.
- Respect containerization: never assume global system state.
- Sensitive data should never be hardcoded or logged.
- See CONTRIBUTING.md for workflow & quality rules.