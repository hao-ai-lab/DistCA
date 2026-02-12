# Development: CI & pre-commit

The project uses **pre-commit** for lint and format. **CI:** [`.github/workflows/lint.yml`](../.github/workflows/lint.yml) runs the same checks on push/PR to `main`/`master`.

## Using pre-commit

**Install the git hook (recommended):**

```bash
pip install pre-commit
pre-commit install
```

Then each commit runs Ruff (Python), ShellCheck (shell scripts), clang-format (C++/CUDA in `csrc/` and `baseline/`), and generic hooks (YAML, trailing whitespace, etc.). Submodules (`Megatron-LM`, `TransformerEngine`, `.build`) are excluded.

**Run manually on all files:**

```bash
pre-commit run --all-files
```

Optional: a [`.clang-format`](https://clang.llvm.org/docs/ClangFormat.html) file at the repo root enforces C++/CUDA style; otherwise the default is used.

---

## Verifying that pre-commit did not change behavior

This section explains **which hooks modify files** and **how to confirm** nothing functional changed.

### What pre-commit can change

| Hook | Modifies files? | Can change behavior? |
|------|-----------------|----------------------|
| check-yaml, check-merge-conflict, check-added-large-files | No | No |
| **end-of-file-fixer** | Yes (adds final newline) | No |
| **trailing-whitespace** | Yes (removes trailing spaces) | No |
| **ruff-check** (with `--fix`) | Yes (e.g. unused imports, rename to `_`) | Rare; only safe fixes are enabled |
| **ruff-format** | Yes (whitespace, line length, quotes) | No |
| shellcheck | No (fails only; no auto-fix) | No |
| **clang-format** | Yes (C++/CUDA formatting only) | No |

So in practice, **only formatting and trivial fixes** are applied. The one hook that could theoretically change behavior is **ruff-check --fix**; we use conservative rules to avoid that.

### How to verify

**1. Review the diff after pre-commit**

After `pre-commit run --all-files`, run `git diff`. You should see only final newlines, trailing space removal, and reformatting. If you see **logic changes**, review them before committing.

**2. Run tests (recommended)**

Tests are scripts under `tests/` (e.g. `test_planner.py`, `test_rope.py`); many require GPU. They import the `distca` package, so run from the **repo root** with the package on `PYTHONPATH`, or install it first:

```bash
# From repo root — Option A: set PYTHONPATH
PYTHONPATH=. python3 tests/test_planner.py

# Option B: install package in editable mode (once)
pip install -e .
python3 tests/test_planner.py
```

Install dependencies first if needed (use a **virtual environment** on systems where system Python is externally managed, e.g. Ubuntu 24.04 / PEP 668):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Then run tests with `python3 tests/test_planner.py` (or `PYTHONPATH=. python3 tests/...` if not using `pip install -e .`).

CI does not run the full test suite; local test run is the authority for “no behavior change.”