"""
Diagnostic: gemma_3 produced 100% empty generated_text across the ENTIRE
ablation study (all 316,908 samples in Experiments A/B/C), while qwen_vl was
0% empty using the identical script. This isolates one sample and prints the
full vLLM output object (finish_reason, token count, raw text) to find out
whether the model is stopping immediately, hitting an exception that's being
silently swallowed, or something else.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corn_maize_leaf_disease")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from models.vllm_vlm import _to_image_url_block

    dataset_path = load_agml_dataset(args.dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    image_path = df.iloc[args.sample_index]["image_path"]
    true_label = df.iloc[args.sample_index]["label"]

    print(f"Dataset: {args.dataset} | image: {os.path.basename(image_path)} | label: {true_label}")

    llm_kwargs = dict(
        model="google/gemma-3-4b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        allowed_local_media_path="/",
        mm_processor_kwargs={"min_pixels": 200704, "max_pixels": 200704},
    )
    print(f"\nllm_kwargs: {llm_kwargs}")

    llm = LLM(**llm_kwargs)

    # Print the model's real max_model_len and the tokenizer's special tokens,
    # in case the prompt is silently being truncated to 0 remaining tokens.
    try:
        model_config = llm.llm_engine.model_config
        print(f"\nmodel max_model_len: {model_config.max_model_len}")
    except Exception as e:
        print(f"(could not read max_model_len: {e})")

    prompts_to_try = {
        "MCQA-2": (
            "Classify this image into one of the following categories: "
            "healthy, gray leaf spot, blight, common rust, None of the above. "
            "Respond with ONLY the category name, nothing else."
        ),
        "OEQ-disease": (
            "Respond in one sentence: What disease, pest, damage type, or "
            "other stress, if any, is exhibited in this image of a corn plant?"
        ),
        "trivial": "What color is this image? Answer in one word.",
    }

    for label, prompt_text in prompts_to_try.items():
        conversation = [{
            "role": "user",
            "content": [
                _to_image_url_block(image_path),
                {"type": "text", "text": prompt_text},
            ],
        }]

        sampling = SamplingParams(max_tokens=50, temperature=0)
        outputs = llm.chat([conversation], sampling, use_tqdm=False)
        out = outputs[0].outputs[0]

        print(f"\n{'=' * 70}")
        print(f"Prompt variant: {label}")
        print(f"  prompt text: {prompt_text!r}")
        print(f"  finish_reason: {out.finish_reason}")
        print(f"  stop_reason:   {out.stop_reason}")
        print(f"  num output tokens: {len(out.token_ids)}")
        print(f"  token_ids: {list(out.token_ids)[:20]}")
        print(f"  text: {out.text!r}")
        # If prompt_token_ids is available, show input length too.
        try:
            n_prompt_tokens = len(outputs[0].prompt_token_ids)
            print(f"  num prompt tokens: {n_prompt_tokens}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
