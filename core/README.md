# `core/` — Rewrite Experiment Pipeline

End-to-end pipeline for studying how text rewriting affects vision-language model claim verification. The three stages are:

```
rewrite  →  infer  →  analyze
```

All three are accessible through a single CLI entry point:

```bash
python -m core.run <subcommand> [options]
```

---

## Quick start

```bash
# 1. Generate rewritten claims
python -m core.run rewrite natural --dataset sciclaimeval --split dev --gen-model openai/gpt-4o
python -m core.run rewrite inject  --dataset sciclaimeval --split dev --gen-model openai/gpt-4o
python -m core.run rewrite mix     --dataset sciclaimeval --split dev --seed 42

# 2. Run inference
python -m core.run infer --dataset sciclaimeval --split dev --model Qwen/Qwen3-VL-32B-Instruct --mix-seed 42

# 3. Analyse and produce tables / plots
python -m core.run analyze --dataset sciclaimeval --split dev test --eval-mode consistency --mix-seed 42
```

---

## Module overview

### `config.py`
Central constants and YAML loaders. Defines the dataset registry (`DATASETS`), rewrite condition groups (`NATURAL_GROUPS`, `CONTROLLED_CONDITIONS`), eval mode mappings, and display metadata for plots. All experiment-wide parameters originate here or in `config/*.yaml`.

### `io.py`
Path conventions and JSON helpers. Encodes the directory layout:

| Data type | Location |
|-----------|----------|
| Rewrites  | `rewrites/<dataset>/<split>/natural\|inject/<condition>.json` |
| Runs      | `runs/<dataset>/<split>/<model-slug>/<condition>.json` |
| Results   | `results/<dataset>/<split>/<eval-mode>/` |

### `run.py`
CLI orchestration. Implements the `rewrite`, `infer`, and `analyze` subcommands. `analyze` supports multi-split pooling (`--split dev test`), single-variant mode (`--mix-seed`), and fast iteration (`--tables-only`).

---

## Subpackages

### `data/`
Dataset adapters (`formats.py`). Normalises raw JSON items from different benchmark formats into a common schema (`claim_id`, `claim`, `label`, `evi_path`, optional `claim_id_pair`). Raw data lives under the repo-level `data/` directory.

### `rewrite/`
Generates rewritten claim text before inference.

| Module | Purpose |
|--------|---------|
| `natural.py` | LLM-based stylistic rewrites (Language Polishing, Formal Instruction, etc.) via `ThreadPoolExecutor`. |
| `inject.py` | Controlled word injection: decorative (`D`) and conservative (`C`) word pools; LLM inserts exactly one word per claim; validates insertion distance for multi-word levels. |
| `mix.py` | Randomly assigns each claim to one of five natural-variant rewrites using a fixed seed, producing a single `<GROUP>-mix-<seed>.json` file per group. |
| `prompt.py` | Instruction strings for each natural rewrite condition. |

### `inference/`
Vision-language claim verification under self-consistency decoding.

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Runs N parallel CoT calls per claim, takes majority vote → `pred_label_consistency`, `p_supported_consistency`. |
| `prompt.py` | Builds multimodal CoT messages, resolves image paths, parses Supported/Refuted from model output. |
| `backend.py` | Async OpenAI-compatible client wrapping a vLLM or remote endpoint. |

Configuration (concurrency, `consistency_n`, temperature) lives in `config/inference.yaml`.

### `analysis/`
Post-hoc evaluation and figure generation. Functions are mostly pure; `run.py` handles all file I/O.

| Module | Purpose |
|--------|---------|
| `metrics.py` | Core math: pair mapping, performance (F1, pair accuracy), probability shifts (Δp), rewrite claim validation. |
| `stats.py` | Paired bootstrap p-values for all metrics; Benjamini–Hochberg FDR correction. |
| `tables.py` | LaTeX table generators: performance, shifts, flip rates — with signed-cell colour gradients and significance stars. |
| `compass.py` | Compass scatter plots (Δp\_sup vs Δp\_ref per condition). |
| `drift.py` | Drift of model confidence vs original confidence, binned and plotted. |
| `boundary.py` | p\_base vs p\_rewrite scatter with decision-boundary overlay; flip-rate computation. |
| `strip.py` | Cross-model strip chart of per-claim Δp, grouped by condition. |
| `rewrite_quality.py` | Standalone rewrite diagnostics: exact-match rate, normalised edit distance, optional BERTScore. Run directly: `python -m core.analysis.rewrite_quality`. |

---

## Extending the pipeline

| Goal | Where to edit |
|------|---------------|
| New dataset | `config.py` (`DATASETS`) + `data/formats.py` (normaliser) |
| New natural condition | `rewrite/prompt.py` + `config.py` (`NATURAL_GROUPS`) |
| New controlled condition | `rewrite/inject.py` (`POOLS`, `CONDITION_SPECS`) + `config.py` |
| New model | `config/models.yaml` |
| New inference backend | `inference/backend.py` + `config/inference.yaml` |
| New metric or plot | `analysis/metrics.py` + a new plot module + wire up in `run.py` `_analyze` |
