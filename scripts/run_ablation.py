"""
Prompt-sensitivity ablations (reviewer response).

Runs three ablations for local vLLM models (default: qwen_vl + gemma_3) across all
datasets, with standardized prompts hardcoded here (NOT read from configs.yaml) so
the comparison is controlled:

  A  answer-order randomization  — reshuffle all MCQA options INCLUDING the
     "None of the above" position, across 5 seeds. (all task groups)
  B  NOTA-rate sensitivity       — vary the "None of the above" rate 10/30/50%.
     (all task groups)
  C  common vs scientific names  — substitute ONLY the {plant_type} token with the
     species' scientific name; class labels stay unchanged. Run for MCQA and OEQ.
     OEQ predictions are additionally scored with the LLM judge.
     (disease + pest/damage datasets that have a concrete plant_type)

Design:
  * The vLLM engine is loaded ONCE per model and reused across every condition.
  * Every run writes outputs/ablation/{exp}/{cond}/{model}/{dataset}/predictions.csv
    and a metrics.json. Existing complete runs are skipped, so the job is resumable.
  * Full datasets are used (no subsampling) — these are local, free-to-run models.

Usage:
  python scripts/run_ablation.py                       # all experiments, all models
  python scripts/run_ablation.py --experiments A B     # subset of experiments
  python scripts/run_ablation.py --models qwen_vl      # single model
  python scripts/run_ablation.py --datasets tomato_leaf_disease corn_maize_leaf_disease
  python scripts/run_ablation.py --no-judge            # skip OEQ judging (Exp C)
  python scripts/run_ablation.py --dry-run             # print the run matrix only
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root on path when invoked as `python scripts/run_ablation.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# NOTE: pandas / yaml / tasks.classification / utils.mcqa / utils.utils are
# deliberately NOT imported at module level. Generation (vLLM) and judging
# (gpt-oss via transformers) require mutually incompatible transformers
# versions (vllm needs <4.52, gpt-oss needs >=4.55) and run in separate venvs.
# --judge-only must be importable with only pandas/pyyaml/torch/transformers/
# accelerate installed -- it never needs vLLM, agml, sklearn, or PIL. Each
# function below imports exactly what it needs, so `--judge-only` doesn't
# drag in the generation-only dependency stack.

# ---------------------------------------------------------------------------
# Standardized prompts (hardcoded — mirror the paper's Table 1)
# ---------------------------------------------------------------------------

# MCQA-2 form: within-species classes + NOTA, no plant type. Used for A & B.
# get_mcqa_choices appends "None of the above" to the choice list, so {classes}
# already contains it — do not hardcode NOTA in the template.
PROMPT_MCQA2 = (
    "Classify this image into one of the following categories: {classes}. "
    "Respond with ONLY the category name, nothing else."
)

# MCQA-3 form: within-species classes + NOTA + plant type. Used for C (MCQA arm).
PROMPT_MCQA3 = (
    "Classify this image of a {plant_type} plant into one of the following "
    "categories: {classes}. Respond with ONLY the category name, nothing else."
)

# OEQ (disease/pest/damage) form with plant type. Used for C (OEQ arm).
PROMPT_OEQ_DISEASE = (
    "Respond in one sentence: What disease, pest, damage type, or other stress, "
    "if any, is exhibited in this image of a {plant_type} plant?"
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen_vl": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "min_pixels": 200704,   # 448 * 448
        "max_pixels": 200704,
    },
    "gemma_3": {
        "hf_id": "google/gemma-3-4b-it",
        "min_pixels": 200704,
        "max_pixels": 200704,
    },
}

MCQA_NUM_CHOICES = 5          # matches configs.yaml
FUZZY_THRESHOLD = 0.6
ANSWER_RATIO_BY_NOTA = {10: 0.9, 30: 0.7, 50: 0.5}  # NOTA% -> answer_included_ratio
RANDOMIZE_SEEDS = [0, 1, 2, 3, 4]                    # Exp A
NOTA_RATES = [10, 30, 50]                            # Exp B (percent)
DEFAULT_SEED = 42                                    # fixed seed for B and C


# ---------------------------------------------------------------------------
# Dataset metadata (parsed from datasets.txt) + scientific names
# ---------------------------------------------------------------------------

def load_dataset_meta(datasets_file: Path):
    """Return list of dicts: {dataset, plant_type, task, task_group}."""
    task_group_map = {
        "disease": "disease",
        "pest/damage": "pest_damage",
        "crops/weeds": "weeds",
    }
    rows = []
    for line in datasets_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        dataset, plant_type, task = parts[0], parts[1], parts[2]
        rows.append({
            "dataset": dataset,
            "plant_type": plant_type,
            "task": task,
            "task_group": task_group_map.get(task, "other"),
        })
    return rows


def load_scientific_names(path: Path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------------

def build_llm(model_key: str):
    """Load a vLLM engine once for the given model key."""
    # Importing models.vllm_vlm applies the Qwen rope-scaling patch as a side effect.
    from models.vllm_vlm import _to_image_url_block  # noqa: F401
    from vllm import LLM, SamplingParams

    spec = MODEL_REGISTRY[model_key]
    llm_kwargs = dict(
        model=spec["hf_id"],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        allowed_local_media_path="/",
    )
    mm = {}
    if spec.get("min_pixels") is not None:
        mm["min_pixels"] = spec["min_pixels"]
    if spec.get("max_pixels") is not None:
        mm["max_pixels"] = spec["max_pixels"]
    if mm:
        llm_kwargs["mm_processor_kwargs"] = mm

    print(f"[load] building vLLM engine for {model_key} ({spec['hf_id']}) ...", flush=True)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=50, temperature=0)
    return llm, sampling


def _conversation(image_path, prompt_text):
    from models.vllm_vlm import _to_image_url_block
    return [{
        "role": "user",
        "content": [
            _to_image_url_block(image_path),
            {"type": "text", "text": prompt_text},
        ],
    }]


# ---------------------------------------------------------------------------
# Core inference for one (condition, dataset) run
# ---------------------------------------------------------------------------

def load_dataset_df(dataset: str):
    from tasks.classification import load_agml_dataset, agml_to_df
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    return df


def run_mcqa(llm, sampling, df, template, *, seed, answer_included_ratio,
             randomize_nota_position, plant_type=None):
    """Run one MCQA condition over a dataframe. Returns a results DataFrame."""
    import pandas as pd
    from utils.mcqa import get_mcqa_choices
    from utils.utils import fuzzy_match_label

    class_names = sorted(df["label"].unique().tolist())
    paths = df["image_path"].tolist()

    base_template = template
    if plant_type is not None:
        base_template = base_template.replace("{plant_type}", plant_type)

    conversations, per_sample = [], []
    for i, image in enumerate(paths):
        true_label = df.iloc[i]["label"]
        choices, correct_answer, answer_included, _ = get_mcqa_choices(
            true_label=true_label,
            all_classes=class_names,
            options_within_dataset=True,
            mcqa_num_choices=MCQA_NUM_CHOICES,
            answer_included_ratio=answer_included_ratio,
            sample_index=i,
            seed=seed,
            randomize_nota_position=randomize_nota_position,
            print_sample=(i == 0),
        )
        prompt = base_template.format(classes=", ".join(choices))
        conversations.append(_conversation(image, prompt))
        per_sample.append((choices, correct_answer, answer_included, true_label))

    outputs = llm.chat(conversations, sampling, use_tqdm=True)
    generated = [o.outputs[0].text for o in outputs]

    rows = []
    for i, gen in enumerate(generated):
        choices, correct_answer, answer_included, true_label = per_sample[i]
        idx, score, matched_label = fuzzy_match_label(gen, choices, threshold=FUZZY_THRESHOLD)
        pred_choice = choices[idx] if idx is not None else None
        rows.append({
            "image_path": paths[i],
            "label": true_label,
            "generated_text": gen,
            "pred_label": pred_choice,
            "match_score": score if idx is not None else 0.0,
            "mcqa_correct_answer": correct_answer,
            "answer_included": answer_included,
            "is_correct": bool(pred_choice == correct_answer),
        })
    return pd.DataFrame(rows)


def run_oeq(llm, sampling, df, template, *, plant_type):
    """Run one OEQ condition over a dataframe. Returns a results DataFrame."""
    import pandas as pd
    from utils.utils import fuzzy_match_label

    class_names = sorted(df["label"].unique().tolist())
    paths = df["image_path"].tolist()
    prompt = template.replace("{plant_type}", plant_type)

    conversations = [_conversation(img, prompt) for img in paths]
    outputs = llm.chat(conversations, sampling, use_tqdm=True)
    generated = [o.outputs[0].text for o in outputs]

    rows = []
    for i, gen in enumerate(generated):
        true_label = df.iloc[i]["label"]
        idx, score, matched_label = fuzzy_match_label(gen, class_names, threshold=FUZZY_THRESHOLD)
        rows.append({
            "image_path": paths[i],
            "label": true_label,
            "generated_text": gen,
            # Deliberately NOT named "pred_label": LLMJudge.evaluate_predictions()
            # only judges raw generated_text when a "pred_label" column is absent
            # or mostly empty. Since our fuzzy matcher fills this in for most
            # rows, naming it "pred_label" made the judge re-confirm the fuzzy
            # match instead of judging the free-form text -- defeating the
            # point of judging OEQ output.
            "fuzzy_pred_label": matched_label if idx is not None else "",
            "match_score": score if idx is not None else 0.0,
            "is_correct": bool(matched_label == true_label),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run bookkeeping / resume
# ---------------------------------------------------------------------------

def run_dir(exp, cond, model, dataset):
    return REPO_ROOT / "outputs" / "ablation" / exp / cond / model / dataset


def is_complete(out_dir: Path, expect_judge: bool = False) -> bool:
    if not (out_dir / "predictions.csv").exists():
        return False
    if not (out_dir / "metrics.json").exists():
        return False
    if expect_judge and not (out_dir / "judge_metrics.json").exists():
        return False
    return True


def save_run(out_dir: Path, results: "pd.DataFrame", meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "predictions.csv", index=False)
    acc = float(results["is_correct"].mean()) if len(results) else 0.0
    metrics = {"n": int(len(results)), "fuzzy_accuracy": acc, **meta}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return acc


def purge_judging(ablation_dir: Path):
    """Delete all judge outputs under Experiment C so every OEQ run gets rejudged
    from scratch with a single, consistent judge model (predictions.csv is left
    untouched — only judge_metrics.json / predictions_with_judge.csv /
    judge_report.txt are removed).
    """
    c_dir = ablation_dir / "C"
    if not c_dir.exists():
        print("No Experiment C directory found; nothing to purge.", flush=True)
        return
    removed = 0
    for cond_dir in c_dir.glob("oeq_*"):
        for name in ("judge_metrics.json", "predictions_with_judge.csv", "judge_report.txt"):
            for f in cond_dir.rglob(name):
                f.unlink()
                removed += 1
    print(f"Purged {removed} judge output file(s) under {c_dir}.", flush=True)


def find_pending_oeq_dirs(ablation_dir: Path) -> list:
    """OEQ run dirs (Experiment C) that have predictions but no judge_metrics.json yet."""
    pending = []
    c_dir = ablation_dir / "C"
    if not c_dir.exists():
        return pending
    for cond_dir in c_dir.glob("oeq_*"):
        for preds in cond_dir.rglob("predictions.csv"):
            run_dir = preds.parent
            if not (run_dir / "judge_metrics.json").exists():
                pending.append(run_dir)
    return sorted(pending)


def run_judging_pass(ablation_dir: Path, judge_model: str, judge_provider: str):
    """Load the judge ONCE and score every pending OEQ run (Experiment C).

    Deliberately separate from generation: for --provider hf this loads a local
    model onto the GPU, which must not run concurrently with a live vLLM engine.
    Call this only after all vLLM engines have been freed.
    """
    pending = find_pending_oeq_dirs(ablation_dir)
    if not pending:
        print("No pending OEQ runs to judge.", flush=True)
        return

    print(f"\n===== Judging pass: {len(pending)} pending OEQ run(s) "
          f"(model={judge_model}, provider={judge_provider}) =====", flush=True)
    from utils.llm_judge import LLMJudge
    judge = LLMJudge(
        model_name=judge_model,
        api_provider=judge_provider,
        confidence_threshold=1,
        max_workers=10,
    )
    for i, run_dir in enumerate(pending, 1):
        print(f"[judge {i}/{len(pending)}] {run_dir.relative_to(ablation_dir)}", flush=True)
        try:
            judge.evaluate_predictions(
                predictions_csv=str(run_dir / "predictions.csv"),
                output_dir=str(run_dir),
                skip_completed=True,
            )
        except Exception as e:
            print(f"       ERROR judging {run_dir}: {e}", flush=True)
            continue


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def experiment_A(llm, sampling, model, datasets, df_cache):
    """Answer-order randomization: MCQA-2, 5 seeds, NOTA position randomized."""
    for ds in datasets:
        for seed in RANDOMIZE_SEEDS:
            cond = f"seed_{seed}"
            out_dir = run_dir("A", cond, model, ds)
            if is_complete(out_dir):
                print(f"[skip] A/{cond}/{model}/{ds}", flush=True)
                continue
            print(f"[run ] A/{cond}/{model}/{ds}", flush=True)
            df = df_cache.get(ds)
            res = run_mcqa(
                llm, sampling, df, PROMPT_MCQA2,
                seed=seed, answer_included_ratio=0.7,
                randomize_nota_position=True,
            )
            acc = save_run(out_dir, res, {
                "experiment": "A", "condition": cond, "model": model,
                "dataset": ds, "prompt": "MCQA2", "seed": seed,
                "nota_rate": 30, "randomize_nota_position": True,
            })
            print(f"       -> fuzzy_acc={acc:.4f} (n={len(res)})", flush=True)


def experiment_B(llm, sampling, model, datasets, df_cache):
    """NOTA-rate sensitivity: MCQA-2, NOTA rate in {10,30,50}%, fixed seed."""
    for ds in datasets:
        for nota in NOTA_RATES:
            cond = f"nota_{nota}"
            out_dir = run_dir("B", cond, model, ds)
            if is_complete(out_dir):
                print(f"[skip] B/{cond}/{model}/{ds}", flush=True)
                continue
            print(f"[run ] B/{cond}/{model}/{ds}", flush=True)
            df = df_cache.get(ds)
            res = run_mcqa(
                llm, sampling, df, PROMPT_MCQA2,
                seed=DEFAULT_SEED,
                answer_included_ratio=ANSWER_RATIO_BY_NOTA[nota],
                randomize_nota_position=False,
            )
            acc = save_run(out_dir, res, {
                "experiment": "B", "condition": cond, "model": model,
                "dataset": ds, "prompt": "MCQA2", "seed": DEFAULT_SEED,
                "nota_rate": nota, "randomize_nota_position": False,
            })
            print(f"       -> fuzzy_acc={acc:.4f} (n={len(res)})", flush=True)


def experiment_C(llm, sampling, model, meta_by_ds, sci_names, datasets, df_cache):
    """Common vs scientific plant_type, for MCQA-3 and OEQ. Class labels unchanged.

    Generation only — OEQ judging is a separate deferred pass (see
    run_judging_pass) so a local judge model never shares the GPU with a live
    vLLM engine, and the judge model is loaded once instead of per-run.
    """
    for ds in datasets:
        meta = meta_by_ds[ds]
        common = meta["plant_type"]
        sci = sci_names.get(common)
        if sci is None:
            print(f"[skip] C/{model}/{ds}: no scientific name for '{common}'", flush=True)
            continue
        if meta["task_group"] == "weeds":
            print(f"[skip] C/{model}/{ds}: weeds task has no plant_type in prompt", flush=True)
            continue

        for arm, template in (("mcqa", PROMPT_MCQA3), ("oeq", PROMPT_OEQ_DISEASE)):
            for name_kind, plant_value in (("common", common), ("scientific", sci)):
                cond = f"{arm}_{name_kind}"
                out_dir = run_dir("C", cond, model, ds)
                if is_complete(out_dir):
                    print(f"[skip] C/{cond}/{model}/{ds}", flush=True)
                    continue
                print(f"[run ] C/{cond}/{model}/{ds}  plant_type='{plant_value}'", flush=True)
                df = df_cache.get(ds)
                if arm == "mcqa":
                    res = run_mcqa(
                        llm, sampling, df, template,
                        seed=DEFAULT_SEED, answer_included_ratio=0.7,
                        randomize_nota_position=False, plant_type=plant_value,
                    )
                else:
                    res = run_oeq(llm, sampling, df, template, plant_type=plant_value)
                acc = save_run(out_dir, res, {
                    "experiment": "C", "condition": cond, "model": model,
                    "dataset": ds, "arm": arm, "name_kind": name_kind,
                    "plant_type": plant_value,
                })
                print(f"       -> fuzzy_acc={acc:.4f} (n={len(res)})", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--models", nargs="+", default=["qwen_vl", "gemma_3"],
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Subset of dataset names (default: all in datasets.txt).")
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "datasets.txt"))
    ap.add_argument("--sci-names-file", default=str(REPO_ROOT / "scientific_names.yaml"))
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip the OEQ judging pass after generation finishes.")
    ap.add_argument("--judge-only", action="store_true",
                    help="Skip generation entirely; just judge any pending OEQ "
                         "runs from a previous --no-judge pass (no vLLM/GPU model "
                         "load for generation, so it's safe to run standalone).")
    ap.add_argument("--purge-judge", action="store_true",
                    help="Delete all existing judge_metrics.json/predictions_with_judge.csv/"
                         "judge_report.txt under Experiment C before (re)judging, so every "
                         "OEQ run is scored with one consistent judge model. predictions.csv "
                         "(generation output) is untouched. Combine with --judge-only to "
                         "purge-then-rejudge in one step.")
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b",
                    help="Judge model. Local HF model id (with --judge-provider hf, "
                         "default) or an OpenAI model id (with --judge-provider openai, "
                         "subject to that account's daily rate limit).")
    ap.add_argument("--judge-provider", default="hf", choices=["openai", "hf"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the run matrix and exit (no model load).")
    args = ap.parse_args()

    ablation_dir = REPO_ROOT / "outputs" / "ablation"

    if args.purge_judge:
        purge_judging(ablation_dir)

    if args.judge_only:
        run_judging_pass(ablation_dir, args.judge_model, args.judge_provider)
        return

    meta_rows = load_dataset_meta(Path(args.datasets_file))
    meta_by_ds = {r["dataset"]: r for r in meta_rows}
    all_datasets = [r["dataset"] for r in meta_rows]
    datasets = args.datasets if args.datasets else all_datasets
    sci_names = load_scientific_names(Path(args.sci_names_file))
    do_judge = not args.no_judge

    if args.dry_run:
        print("=== ABLATION RUN MATRIX (dry run) ===")
        for model in args.models:
            for exp in args.experiments:
                if exp == "A":
                    print(f"{model} A: {len(datasets)} datasets x {len(RANDOMIZE_SEEDS)} seeds")
                elif exp == "B":
                    print(f"{model} B: {len(datasets)} datasets x {len(NOTA_RATES)} NOTA rates")
                else:
                    elig = [d for d in datasets
                            if meta_by_ds[d]["task_group"] != "weeds"
                            and sci_names.get(meta_by_ds[d]["plant_type"]) is not None]
                    print(f"{model} C: {len(elig)} eligible datasets x 2 arms x 2 name kinds "
                          f"(judge={'on' if do_judge else 'off'})")
        print(f"\nTask groups:")
        for g in ("disease", "pest_damage", "weeds"):
            names = [d for d in datasets if meta_by_ds[d]["task_group"] == g]
            print(f"  {g}: {len(names)} -> {names}")
        return

    for model in args.models:
        llm, sampling = build_llm(model)

        # Cache dataset dataframes for this model pass (loaded lazily).
        class _DFCache:
            def __init__(self):
                self._c = {}
            def get(self, ds):
                if ds not in self._c:
                    self._c[ds] = load_dataset_df(ds)
                return self._c[ds]
        df_cache = _DFCache()

        if "A" in args.experiments:
            print(f"\n===== [{model}] Experiment A: answer-order randomization =====", flush=True)
            experiment_A(llm, sampling, model, datasets, df_cache)
        if "B" in args.experiments:
            print(f"\n===== [{model}] Experiment B: NOTA-rate sensitivity =====", flush=True)
            experiment_B(llm, sampling, model, datasets, df_cache)
        if "C" in args.experiments:
            print(f"\n===== [{model}] Experiment C: common vs scientific names =====", flush=True)
            experiment_C(llm, sampling, model, meta_by_ds, sci_names, datasets, df_cache)

        # Free the engine before loading the next model (and before any judging).
        del llm
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\nAll requested generation runs complete.", flush=True)

    if do_judge and "C" in args.experiments:
        # Deferred: all vLLM engines are freed by now, so a local judge model
        # (--judge-provider hf) is safe to load without GPU contention, and it
        # loads exactly once here instead of once per OEQ run.
        run_judging_pass(ablation_dir, args.judge_model, args.judge_provider)


if __name__ == "__main__":
    main()
