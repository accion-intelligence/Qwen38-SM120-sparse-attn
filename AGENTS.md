# Repository Guidelines

## Project Structure & Module Organization

`sm120_attn/` contains the Python package: `sparse_prefill.py` selects attention blocks and integrates with vLLM, while `sm120_attn/kernels/` holds the BF16 and FP8 CuTe DSL kernels. `tests/` contains GPU correctness and benchmark scripts. `scripts/` builds prompts and runs generation, scoring, and latency evaluations; `docs/` explains deployment and records validation results. `tools/patch_vllm.py` applies the integration hook, with notes in `patches/`. `extras/` contains optional experiments. `vllm/` is a submodule with its own contributor instructions; follow those when editing it.

## Build, Test, and Development Commands

- `pip install -e .` installs the package for local development. Kernels compile on first use; there is no separate C++ build.
- `pip install -e '.[eval]'` adds dependencies for evaluation scripts.
- `python tools/patch_vllm.py --target vllm/vllm --check` checks the vLLM hook without changing files; omit `--check` to apply it.
- `python tests/test_fa_bf16.py` runs the BF16 kernel check. Run `python tests/test_fa_fp8.py --big` and `python tests/test_fa_fp8_tma.py --big` for extended FP8 checks. See `README.md` for serving and evaluation commands.

## Coding Style & Naming Conventions

Use four spaces for Python indentation. Follow the existing module style: `snake_case` for files, functions, and variables; `PascalCase` for classes; uppercase names for constants and environment variables. Keep tensor shapes, dtypes, and layout assumptions explicit in docstrings or nearby comments. No formatter or linter is configured in `pyproject.toml`; avoid unrelated formatting changes.

## Testing Guidelines

Tests are standalone Python scripts named `test_*.py`, with some additional `test_*` functions. Run affected scripts directly on a compatible SM120 GPU with the CUDA and package versions listed in `README.md`. For sparse-path changes, also run `python tests/test_sparse_prefill.py` and `python tests/test_sparse_prefill_fp8.py`; compare numerical error and retained-block output against their references. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use short, plain-English subjects such as `update docs`; there is no enforced prefix convention. Write a specific imperative subject describing the change. In pull requests, describe the affected kernel or integration path, list commands and GPU used for validation, and include relevant error or latency results. Link related issues when available and update documentation for changes to flags, defaults, or deployment steps.
