"""
Diagnostic: gemma_3 produced 100% empty/garbage generated_text across the
ENTIRE ablation study (all 316,908 samples in Experiments A/B/C), while
qwen_vl was 0% empty using the identical script. Ruled out: an unconditional
rope-scaling monkeypatch in models/vllm_vlm.py (was suspected of corrupting
Gemma 3's positional embeddings; scoping it to Qwen's case specifically
produced byte-for-byte identical garbage, proving it wasn't the cause).

This tests several candidate mitigations, each as a separate LLM()
instantiation (engine kwargs are constructor-level, so each variant needs its
own load):

  1. baseline        -- current config (known garbage), all 3 prompt types
  2. dtype=bfloat16   -- vLLM never gets an explicit dtype anywhere in this
                         pipeline despite configs.yaml specifying bfloat16;
                         Gemma models are known to be numerically unstable
                         outside bf16 (activation overflow in fp16)
  3. enforce_eager    -- rules out CUDA graph capture corruption
  4. both combined    -- dtype=bfloat16 + enforce_eager
  5. no mm kwargs     -- drops min_pixels/max_pixels (Qwen-specific "smart
                         resize" param that may not be valid for Gemma 3's
                         SigLIP-based processor)
  6. text-only        -- no image at all, isolates the vision pathway

Usage (on the cluster, main .venv with vLLM):
  python scripts/debug_gemma_empty_output.py
  python scripts/debug_gemma_empty_output.py --dataset corn_maize_leaf_disease
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.classification import load_agml_dataset, agml_to_df


MCQA_PROMPT = (
    "Classify this image into one of the following categories: "
    "healthy, gray leaf spot, blight, common rust, None of the above. "
    "Respond with ONLY the category name, nothing else."
)
OEQ_PROMPT = (
    "Respond in one sentence: What disease, pest, damage type, or "
    "other stress, if any, is exhibited in this image of a corn plant?"
)
TRIVIAL_PROMPT = "What color is this image? Answer in one word."


def report(label, prompt_text, outputs):
    out = outputs[0].outputs[0]
    print(f"\n{'=' * 70}")
    print(f"Variant/prompt: {label}")
    print(f"  prompt text: {prompt_text!r}")
    print(f"  finish_reason: {out.finish_reason}")
    print(f"  stop_reason:   {out.stop_reason}")
    print(f"  num output tokens: {len(out.token_ids)}")
    print(f"  token_ids: {list(out.token_ids)[:20]}")
    print(f"  text: {out.text!r}")
    try:
        n_prompt_tokens = len(outputs[0].prompt_token_ids)
        print(f"  num prompt tokens: {n_prompt_tokens}")
    except Exception:
        pass


def run_variant(variant_label, llm_kwargs, image_path, prompts):
    """Build a fresh LLM with the given kwargs and run each (label, prompt) pair."""
    from vllm import LLM, SamplingParams
    from models.vllm_vlm import _to_image_url_block

    print(f"\n\n{'#' * 70}")
    print(f"# VARIANT: {variant_label}")
    print(f"# llm_kwargs: {llm_kwargs}")
    print(f"{'#' * 70}")

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=50, temperature=0)

    for label, prompt_text, use_image in prompts:
        if use_image:
            content = [
                _to_image_url_block(image_path),
                {"type": "text", "text": prompt_text},
            ]
        else:
            content = [{"type": "text", "text": prompt_text}]
        conversation = [{"role": "user", "content": content}]
        outputs = llm.chat([conversation], sampling, use_tqdm=False)
        report(f"{variant_label} / {label}", prompt_text, outputs)

    del llm
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corn_maize_leaf_disease")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    dataset_path = load_agml_dataset(args.dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    image_path = df.iloc[args.sample_index]["image_path"]
    true_label = df.iloc[args.sample_index]["label"]
    print(f"Dataset: {args.dataset} | image: {os.path.basename(image_path)} | label: {true_label}")

    base_kwargs = dict(
        model="google/gemma-3-4b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        allowed_local_media_path="/",
    )
    mm_kwargs = {"mm_processor_kwargs": {"min_pixels": 200704, "max_pixels": 200704}}

    all_prompts = [
        ("MCQA-2", MCQA_PROMPT, True),
        ("OEQ-disease", OEQ_PROMPT, True),
        ("trivial", TRIVIAL_PROMPT, True),
    ]
    mcqa_only = [("MCQA-2", MCQA_PROMPT, True)]

    # 1. Baseline (current known-garbage config), all 3 prompts.
    run_variant("1-baseline", {**base_kwargs, **mm_kwargs}, image_path, all_prompts)

    # 2. dtype=bfloat16 explicit.
    run_variant("2-dtype-bfloat16",
                {**base_kwargs, **mm_kwargs, "dtype": "bfloat16"}, image_path, mcqa_only)

    # 3. enforce_eager=True.
    run_variant("3-enforce-eager",
                {**base_kwargs, **mm_kwargs, "enforce_eager": True}, image_path, mcqa_only)

    # 4. Both combined.
    run_variant("4-dtype-and-eager",
                {**base_kwargs, **mm_kwargs, "dtype": "bfloat16", "enforce_eager": True},
                image_path, mcqa_only)

    # 5. No mm_processor_kwargs.
    run_variant("5-no-mm-kwargs", base_kwargs, image_path, mcqa_only)

    # 6. Text-only, no image.
    run_variant("6-text-only", {**base_kwargs, **mm_kwargs}, image_path,
                [("trivial-text-only", "Say the word 'hello' and nothing else.", False)])

    print("\n\nAll variants complete. Compare 'text' fields above -- whichever "
          "variant(s) produce coherent, non-repetitive text identify the fix.")


if __name__ == "__main__":
    main()
