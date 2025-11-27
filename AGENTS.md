# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `vlmeval/`: `dataset/` (benchmark loaders and metrics), `vlm/` (model wrappers), `api/` (cloud APIs), `utils/` (shared helpers), `smp/` (logging, misc utilities). Entry script `run.py` drives inference/evaluation.  
- `docs/` contains English/Chinese/Japanese guides (see `docs/en/Quickstart.md` and `docs/en/Development.md`).  
- `assets/` provides sample images; `outputs/` is the default working directory for generated predictions and scores.  
- Add new datasets under `vlmeval/dataset/` and new models under `vlmeval/vlm/` with config entries in `vlmeval/config.py`.

## Build, Test, and Development Commands
- Install in editable mode: `pip install -e .` (requires Python 3.8+, PyTorch/Transformers per model notes).  
- Evaluate image benchmarks: `python run.py --data MMBench_DEV_EN --model idefics_80b_instruct --verbose`.  
- Evaluate video benchmarks: `torchrun --nproc-per-node=8 run.py --data MMBench_Video_8frame_nopack --model idefics2_8`.  
- Model sanity check: `vlmutil check {MODEL_NAME}` after adding configs.  
- Code hygiene: `pre-commit run --all-files` (flake8, yapf, EOL/whitespace checks).  
- Need help? `python run.py --help` and `docs/en/ConfigSystem.md` list all flags.

## Coding Style & Naming Conventions
- Python: 4-space indentation, line length 120 (yapf config); run yapf/flake8 via pre-commit.  
- Prefer snake_case for files/functions, CamelCase for classes, UPPER_SNAKE for constants.  
- Keep model/dataset IDs aligned with `supported_VLM` keys and dataset TYPE values; mirror existing prompt-building patterns when extending.

## Testing Guidelines
- No standalone unit suite; validate changes by running `run.py --mode infer` on a small dataset and inspecting `{model}_{dataset}.xlsx` under `outputs/`.  
- For new datasets, ensure `evaluate()` returns expected metrics and handles judge LLM arguments.  
- For new models, verify `generate_inner` handles interleaved image/text lists and respects `use_custom_prompt` if overridden.

## Commit & Pull Request Guidelines
- Follow short, descriptive commit titles seen in history (e.g., "Add XXX", "Improve YYY").  
- PRs should summarize the feature/fix, list datasets/models touched, include sample commands/logs, and link issues or papers when relevant.  
- Run `pre-commit` before pushing; note any version/transformer constraints in the PR description.  
- Provide screenshots or result snippets when changing outputs or docs; keep configuration changes documented.

## Security & Configuration Tips
- Store API keys in `.env` (not committed). Sensitive env vars include `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DASHSCOPE_API_KEY`, etc.  
- Use `SPLIT_THINK=True` for thinking-mode models and `PRED_FORMAT=tsv` when expecting very long outputs.  
- Set `LMUData` to control dataset download location; avoid hardcoding absolute paths in configs.
