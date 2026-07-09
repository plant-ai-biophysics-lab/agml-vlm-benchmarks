"""
Diagnostic for the Experiment C anomaly: gemma_3 produced byte-identical
predictions.csv for mcqa_common vs mcqa_scientific across all 21 datasets,
while qwen_vl showed small differences. That's suspicious -- 100% identical
greedy output across thousands of samples is not what "the model ignores the
species name" should look like; more likely something in the pipeline isn't
actually varying the prompt for gemma_3.

This script isolates ONE image, builds the two full prompts by hand (common vs
scientific plant_type), prints them so you can visually confirm they differ,
and runs each through gemma_3 individually (not batched) to see if the raw
generated text actually differs.

Usage (on the cluster, in the vLLM-capable venv):
  python scripts/debug_gemma_plant_type.py
  python scripts/debug_gemma_plant_type.py --dataset corn_maize_leaf_disease
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.classification import load_agml_dataset, agml_to_df
from utils.mcqa import get_mcqa_choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corn_maize_leaf_disease")
    ap.add_argument("--plant-common", default="corn")
    ap.add_argument("--plant-scientific", default="Zea mays")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--disable-prefix-caching", action="store_true",
                    help="Rule out vLLM automatic prefix caching as the cause.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from models.vllm_vlm import _to_image_url_block

    dataset_path = load_agml_dataset(args.dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    class_names = sorted(df["label"].unique().tolist())

    true_label = df.iloc[args.sample_index]["label"]
    image_path = df.iloc[args.sample_index]["image_path"]
    choices, correct_answer, _, _ = get_mcqa_choices(
        true_label=true_label,
        all_classes=class_names,
        options_within_dataset=True,
        mcqa_num_choices=5,
        answer_included_ratio=0.7,
        sample_index=args.sample_index,
        seed=42,
        randomize_nota_position=False,
    )

    template = (
        "Classify this image of a {plant_type} plant into one of the following "
        "categories: {classes}. Respond with ONLY the category name, nothing else."
    )
    prompt_common = template.replace("{plant_type}", args.plant_common).format(
        classes=", ".join(choices)
    )
    prompt_scientific = template.replace("{plant_type}", args.plant_scientific).format(
        classes=", ".join(choices)
    )

    print("=" * 80)
    print(f"Dataset: {args.dataset}  | image: {os.path.basename(image_path)}")
    print(f"True label: {true_label}  | choices: {choices}")
    print("-" * 80)
    print("PROMPT (common):")
    print(prompt_common)
    print("-" * 80)
    print("PROMPT (scientific):")
    print(prompt_scientific)
    print("=" * 80)

    if prompt_common == prompt_scientific:
        print("BUG CONFIRMED: the two prompts are textually identical -- the "
              "plant_type substitution did not happen. Stop here and fix the "
              "substitution logic before rerunning.")
        return

    print("Prompts differ as expected. Now checking whether gemma_3's output differs...\n")

    llm_kwargs = dict(
        model="google/gemma-3-4b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        allowed_local_media_path="/",
        mm_processor_kwargs={"min_pixels": 200704, "max_pixels": 200704},
    )
    if args.disable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = False

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=50, temperature=0)

    def conv(prompt_text):
        return [{
            "role": "user",
            "content": [
                _to_image_url_block(image_path),
                {"type": "text", "text": prompt_text},
            ],
        }]

    out_common = llm.chat([conv(prompt_common)], sampling, use_tqdm=False)
    out_scientific = llm.chat([conv(prompt_scientific)], sampling, use_tqdm=False)

    text_common = out_common[0].outputs[0].text
    text_scientific = out_scientific[0].outputs[0].text

    print(f"Generated (common):     {text_common!r}")
    print(f"Generated (scientific): {text_scientific!r}")
    print()
    if text_common == text_scientific:
        print("Outputs are identical for this single isolated sample too.")
        print("This suggests gemma_3 genuinely ignores the plant-type clause "
              "for this prompt/image, at least for this sample -- rerun with "
              "--disable-prefix-caching and a few different --sample-index "
              "values to build confidence before treating it as a real finding.")
    else:
        print("Outputs DIFFER here, despite the full-dataset run producing "
              "byte-identical files. That points to a bug in the batched "
              "ablation run (e.g. vLLM prefix caching, or the two condition "
              "runs not actually differing in practice) rather than genuine "
              "model behavior. Try rerunning Experiment C for gemma_3 with "
              "--disable-prefix-caching added to build_llm() in run_ablation.py.")


if __name__ == "__main__":
    main()
