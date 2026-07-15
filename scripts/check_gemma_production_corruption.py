"""
Check whether real production gemma_3 predictions.csv files show the same
empty/garbage generated_text signature found in the ablation study.

Root cause: models/vllm_vlm.py applied an UNCONDITIONAL monkeypatch to
vllm.transformers_utils.config.patch_rope_scaling_dict, originally written to
fix a Qwen2.5-VL-specific config quirk (legacy 'type'='mrope' vs modern
'rope_type'='default'). The patch fired for ANY model's rope_scaling dict
where 'type' != 'rope_type', not just Qwen's -- if gemma_3's config has
legitimately different values there (e.g. for its interleaved local/global
attention), the patch silently corrupted its positional embeddings, producing
near-garbage or empty generations. This was introduced in commit e058de5
("check mrope"), the same day the pipeline switched to vLLM (46c96e6) -- so
every vLLM-based gemma_3 run since then may be affected, including production.

Usage:
  python3 scripts/check_gemma_production_corruption.py \
    --root /group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/zero_shot_classification/
"""

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    found_any = False
    for p in sorted(root.rglob("predictions.csv")):
        if "gemma" not in str(p).lower():
            continue
        found_any = True
        try:
            with open(p) as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"SKIP (read error): {p} -- {e}")
            continue
        if not rows or "generated_text" not in rows[0]:
            continue
        n = len(rows)
        empty = sum(1 for r in rows if (r.get("generated_text") or "").strip() == "")
        if empty > 0:
            print(f"  {empty:5d}/{n:5d} empty  {p}")

    if not found_any:
        print(f"No gemma predictions.csv found under {root}")
    else:
        print("\nDone. Any nonzero empty count above indicates the same corruption.")


if __name__ == "__main__":
    main()
