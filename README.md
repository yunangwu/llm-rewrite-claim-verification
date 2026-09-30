# LLM rewrite and claim verification

Code and results for measuring how rewriting a scientific claim changes a
vision-language model's confidence when verifying it against a table or figure.

Claims from SciClaimEval are rewritten by an LLM (hedging, boosting, and other
stylistic conditions), each model verifies the original and rewritten claims,
and we compare the resulting probabilities.

## Contents

- `core/`: pipeline (rewrite, inference, analysis)
- `rewrites/`: rewritten claims
- `runs/`: model predictions
- `results_0526_for_submission/`: results reported in the paper
- `results/`: later results, including the second rewriter (gemma-4-31B-it)
- `annotation/`: human annotation study

## Setup

```bash
uv sync
```

Place the SciClaimEval data in `data/SciClaimEval/` (not included).

## Reproduce the paper results

The model predictions are included, so the analysis can be rerun without a GPU:

```bash
uv run python -m core.run analyze --dataset sciclaimeval --split dev test --mix-seed 42 \
    --models gemma-4-26B-A4B-it gemma-4-31B-it GLM-4.6V-Flash InternVL3-2B InternVL3-8B \
             InternVL3-38B Kimi-VL-A3B-Thinking-2506 Qwen3-VL-2B-Instruct \
             Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct Qwen3-VL-32B-Instruct
uv run python scripts/make_word_pool_table.py
uv run python scripts/make_rewrite_table.py --claim-id val_tab_0484 --out results/rewrite_example.tex
```

Figures and tables are written to `results/sciclaimeval/dev+test/consistency/`,
and the two extra tables to `results/`. They match `results_0526_for_submission/`. The significance marks for pair accuracy can
vary slightly between runs; set `PYTHONHASHSEED=0` for identical output.

Human annotation scores:

```bash
uv run --with openpyxl python annotation/score_annotations.py
```
