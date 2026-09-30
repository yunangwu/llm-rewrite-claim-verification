"""
Self-consistency CoT inference pipeline.

For each claim, fires N CoT requests in parallel and aggregates by majority vote:
  p_supported_consistency = fraction of valid responses that voted "Supported"
  pred_label_consistency  = "Supported" if p >= 0.5 else "Refuted"

Output dict per item:
  claim_id, claim, label, evi_type, use_context,
  pred_label_consistency, p_supported_consistency, consistency_n_valid,
  [consistency_labels, consistency_responses]  ← only when store_responses=True
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

from core.inference.backend import Backend
from core.inference.prompt import (
    MAX_TOKENS_COT,
    build_prompt,
    parse_answer,
    resolve_image_path,
)

_DEFAULT_MAX_TOKENS = MAX_TOKENS_COT  # 2048 for standard models


async def _process_item(
    backend: Backend,
    item: dict,
    data_dir: str,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    pbar: atqdm,
    *,
    consistency_n: int,
    consistency_temp: float,
    parse_retries: int,
    store_responses: bool,
    fmt: str = "sciclaimeval",
    max_tokens_cot: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    claim_id = item.get("claim_id", str(index))

    async with semaphore:
        try:
            cot_content = build_prompt(item, data_dir, fmt)

            # Fire all N requests in parallel
            responses = await asyncio.gather(*[
                backend.chat_complete(
                    cot_content,
                    max_tokens=max_tokens_cot,
                    temperature=consistency_temp,
                )
                for _ in range(consistency_n)
            ])

            texts  = [r.choices[0].message.content or "" for r in responses]
            labels = [parse_answer(t, fmt) for t in texts]

            # Retry unparsed responses
            for i in range(consistency_n):
                retries = 0
                while labels[i] not in ("Supported", "Refuted") and retries < parse_retries:
                    retries += 1
                    resp = await backend.chat_complete(
                        cot_content,
                        max_tokens=max_tokens_cot,
                        temperature=consistency_temp,
                    )
                    texts[i]  = resp.choices[0].message.content or ""
                    labels[i] = parse_answer(texts[i], fmt)

            n_sup   = sum(1 for lb in labels if lb == "Supported")
            n_valid = sum(1 for lb in labels if lb in ("Supported", "Refuted"))
            p       = round(n_sup / n_valid, 6) if n_valid > 0 else None
            pred    = (
                "Supported" if p is not None and p >= 0.5
                else "Refuted" if p is not None
                else ""
            )

            out: dict = {
                "claim_id":                  claim_id,
                "claim":                     item.get("claim", ""),
                "label":                     item.get("label", ""),
                "evi_type":                  item.get("evi_type", ""),
                "use_context":               item.get("use_context", ""),
                "pred_label_consistency":    pred,
                "p_supported_consistency":   p,
                "consistency_n_valid":       n_valid,
            }
            if item.get("claim_id_pair") is not None:
                out["claim_id_pair"] = item["claim_id_pair"]
            if store_responses:
                out["consistency_labels"]    = labels
                out["consistency_responses"] = texts

            pbar.set_postfix(pred=pred, p=f"{p:.3f}" if p is not None else "N/A", refresh=False)
            pbar.update(1)
            return out

        except Exception as exc:
            atqdm.write(f"  ERROR [{claim_id}]: {exc}")
            pbar.update(1)
            return {
                "claim_id":                claim_id,
                "label":                   item.get("label", ""),
                "pred_label_consistency":  "",
                "p_supported_consistency": None,
                "consistency_n_valid":     0,
                "error":                   str(exc),
            }


async def run_dataset(
    backend: Backend,
    dataset: list[dict],
    data_dir: str,
    *,
    concurrency: int,
    consistency_n: int,
    consistency_temp: float,
    parse_retries: int,
    store_responses: bool,
    fmt: str = "sciclaimeval",
    limit: int | None = None,
    skip_missing_images: bool = False,
    desc: str = "infer",
    max_tokens_cot: int = _DEFAULT_MAX_TOKENS,
) -> list[dict]:
    """Run consistency inference on all items. Returns result list; caller saves to disk."""
    if skip_missing_images:
        before = len(dataset)
        dataset = [
            item for item in dataset
            if Path(resolve_image_path(data_dir, item["evi_path"])).exists()
        ]
        if (dropped := before - len(dataset)):
            atqdm.write(f"  Skipped {dropped} rows with missing images")

    if limit is not None:
        dataset = dataset[:limit]

    missing = sum(
        1 for item in dataset
        if not Path(resolve_image_path(data_dir, item["evi_path"])).exists()
    )
    if missing:
        raise FileNotFoundError(f"{missing}/{len(dataset)} images not found under {data_dir}")

    semaphore = asyncio.Semaphore(concurrency)
    with atqdm(total=len(dataset), desc=desc, unit="item", dynamic_ncols=True) as pbar:
        return await asyncio.gather(*[
            _process_item(
                backend, item, data_dir, i, len(dataset), semaphore, pbar,
                consistency_n=consistency_n,
                consistency_temp=consistency_temp,
                parse_retries=parse_retries,
                store_responses=store_responses,
                fmt=fmt,
                max_tokens_cot=max_tokens_cot,
            )
            for i, item in enumerate(dataset)
        ])
