import os
from typing import Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams

# Qwen2.5-VL config.json has both legacy 'type=mrope' and modern 'rope_type=default',
# which some vLLM versions reject as a conflict. Patch the validator to reconcile them
# by promoting the legacy 'type' value into 'rope_type' before the check runs.
#
# Scoped specifically to Qwen's "type"=="mrope" case: this patch was previously
# unconditional (any rope_scaling dict with type != rope_type), which silently
# rewrote OTHER models' rope_scaling too -- including Gemma 3, whose config
# legitimately uses different type/rope_type values for its interleaved
# local/global attention layers. That corruption produced garbage output
# (near-empty / repeated-token generations) for every gemma_3 run.
try:
    import vllm.transformers_utils.config as _vllm_cfg

    _orig_patch = _vllm_cfg.patch_rope_scaling_dict

    def _patched_patch_rope_scaling_dict(rope_scaling: dict) -> None:
        if (
            isinstance(rope_scaling, dict)
            and rope_scaling.get("type") == "mrope"
            and "rope_type" in rope_scaling
            and rope_scaling["rope_type"] != "mrope"
        ):
            rope_scaling["rope_type"] = rope_scaling["type"]
        _orig_patch(rope_scaling)

    _vllm_cfg.patch_rope_scaling_dict = _patched_patch_rope_scaling_dict
except Exception:
    pass

from tasks.classification import load_agml_dataset, agml_to_df
from utils.prep_context import create_classification_message, build_prompt_descriptions
from utils.utils import save_classification_results, fuzzy_match_label
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes


def _to_image_url_block(path: str) -> dict:
    if not path.startswith(("file://", "http://", "https://", "data:")):
        path = f"file://{os.path.abspath(path)}"
    return {"type": "image_url", "image_url": {"url": path}}


def _normalize_conversation(conversation: list) -> list:
    """Convert internal Qwen-style image refs to vLLM-compatible image_url format."""
    normalized = []
    for msg in conversation:
        content = msg.get("content", [])
        if isinstance(content, list):
            new_content = []
            for block in content:
                if block.get("type") == "image":
                    new_content.append(_to_image_url_block(block["image"]))
                else:
                    new_content.append(block)
            content = new_content
        normalized.append({"role": msg["role"], "content": content})
    return normalized


def test(
    args: dict,
    model_type: str,
    dataset: str,
    output_dir: str,
    context: dict = None,
    max_num_class_context: Optional[int] = None,
    include_correct_class: bool = True,
    random_pool: bool = False,
):
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))

    sample_limit = args.get("sample_limit", None)
    if sample_limit and 0 < sample_limit < 1:
        seed = int(args.get("random_seed", 42))
        print(f"Sample limit: {sample_limit} | Seed: {seed} | Full dataset size: {len(df)}")
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_limit, random_state=seed)
        ).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Subsampled dataset size: {len(df)}")

    class_names = sorted(df["label"].unique().tolist())
    candidate_labels = class_names
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    classes_str = ", ".join(candidate_labels)

    mcqa_options = args.get("mcqa_options", None)
    all_dataset_classes = None
    answer_included_list = []
    mcqa_correct_answers = []
    mcqa_choices_list = []

    if mcqa_options:
        print("\nMCQA Mode Enabled:")
        print(f"  Options within dataset: {mcqa_options.get('options_within_dataset', True)}")
        print(f"  Number of choices: {mcqa_options.get('mcqa_num_choices', 4)}")
        if not mcqa_options.get("options_within_dataset", True):
            all_dataset_classes = load_all_dataset_classes()
            print(f"  Loaded {len(all_dataset_classes)} datasets for cross-dataset sampling")

    conversation_template = args["prompt_template"]
    if not mcqa_options:
        conversation_template = conversation_template.format(classes=classes_str)
    print("Conversation template:", conversation_template)

    use_desc = args.get("context_options", {}).get("use_desc", False)
    prepend_blocks, context_warnings = build_prompt_descriptions(
        dataset_name=dataset,
        use_desc=use_desc,
        datasets_file=args.get("datasets_file", "datasets.txt"),
        context_file=args.get("context_file", "context.yaml"),
    )
    for warn in context_warnings:
        print(f"WARNING [context]: {warn}")

    # Build vLLM engine
    llm_kwargs = dict(
        model=model_type,
        tensor_parallel_size=args.get("tensor_parallel_size", 1),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.90),
        trust_remote_code=args.get("trust_remote_code", True),
        # Allow multiple images per prompt for in-context learning
        limit_mm_per_prompt={"image": args.get("max_images_per_prompt", 20) if context is not None else 2},
    )
    if args.get("max_model_len"):
        llm_kwargs["max_model_len"] = args["max_model_len"]
    if args.get("enforce_eager"):
        llm_kwargs["enforce_eager"] = True
    llm_kwargs["allowed_local_media_path"] = args.get("allowed_local_media_path", "/")

    # Pass pixel constraints through to the vision processor (Qwen VL, etc.)
    mm_processor_kwargs = {}
    if args.get("min_pixels") is not None:
        mm_processor_kwargs["min_pixels"] = args["min_pixels"]
    if args.get("max_pixels") is not None:
        mm_processor_kwargs["max_pixels"] = args["max_pixels"]
    if mm_processor_kwargs:
        llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        print(f"Image size constraint: {mm_processor_kwargs}")

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(max_tokens=50, temperature=0)

    # Build all conversations up front; vLLM batches them internally
    paths = df["image_path"].tolist()
    all_conversations = []
    num_context_examples_list = []
    num_context_classes_list = []
    sample_prompt_printed = False

    for sample_index, image in enumerate(tqdm(paths, desc="Building prompts")):
        if mcqa_options:
            true_label = df.iloc[sample_index]["label"]
            choices, correct_answer, answer_included, _ = get_mcqa_choices(
                true_label=true_label,
                all_classes=candidate_labels,
                options_within_dataset=mcqa_options.get("options_within_dataset", True),
                mcqa_num_choices=mcqa_options.get("mcqa_num_choices", 4),
                all_dataset_classes=all_dataset_classes,
                current_dataset=dataset,
                answer_included_ratio=0.7,
                sample_index=sample_index,
                print_sample=(sample_index == 0),
            )
            answer_included_list.append(answer_included)
            mcqa_correct_answers.append(correct_answer)
            mcqa_choices_list.append(choices)
            prompt = conversation_template.format(classes=", ".join(choices))
        else:
            prompt = conversation_template

        if context is not None:
            message, context_meta = create_classification_message(
                task=None,
                template=prompt,
                query_image_path=image,
                context_examples=context,
                max_num_class_context=max_num_class_context,
                correct_class=df.iloc[sample_index]["label"],
                include_correct_class=include_correct_class,
                random_pool=random_pool,
                prepend_text=prepend_blocks,
            )
            conversation = _normalize_conversation([message])
            num_context_examples_list.append(context_meta["num_context_examples"])
            num_context_classes_list.append(context_meta["num_context_classes"])
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        _to_image_url_block(image),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            num_context_examples_list.append(0)
            num_context_classes_list.append(0)

        if not sample_prompt_printed:
            print("--- Sample prompt (first item this run) ---", flush=True)
            for block in conversation[0]["content"]:
                if block.get("type") == "text":
                    print(f"text: {block['text']}", flush=True)
                elif block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    print(f"image: {os.path.basename(url.split('?')[0])}", flush=True)
            print("------------------------------------------", flush=True)
            sample_prompt_printed = True

        all_conversations.append(conversation)

    # Run batch inference
    print(f"Running vLLM inference on {len(all_conversations)} samples...")
    outputs = llm.chat(all_conversations, sampling_params, use_tqdm=True)
    generated_texts = [out.outputs[0].text for out in outputs]
    total_input_tokens_list = [
        len(out.prompt_token_ids) if out.prompt_token_ids else 0 for out in outputs
    ]

    # Post-process predictions
    preds_ids = []
    probs_rows = []
    match_scores = []
    chosen_options = []

    for i, generated_text in enumerate(generated_texts):
        if mcqa_options:
            sample_choices = mcqa_choices_list[i]
            correct_answer = mcqa_correct_answers[i]
            predicted_class, match_score, _ = fuzzy_match_label(
                generated_text, sample_choices, threshold=0.6
            )
            if predicted_class is not None:
                chosen_options.append(predicted_class + 1)
                matched_choice = sample_choices[predicted_class]
                predicted_class = (
                    candidate_labels.index(matched_choice)
                    if matched_choice in candidate_labels
                    else None
                )
            else:
                chosen_options.append(None)
                match_score = 0.0
                print(f"WARNING [Sample {i}]: No match found")
                print(f"  Generated: '{generated_text}'")
                print(f"  Choices: {sample_choices}")
                print(f"  Correct: {correct_answer}")
        else:
            predicted_class, match_score, _ = fuzzy_match_label(
                generated_text, candidate_labels, threshold=0.6
            )
            chosen_options.append(None)
            if predicted_class is None:
                match_score = 0.0
                print(f"WARNING: No match found for: '{generated_text}'")

        preds_ids.append(predicted_class)
        match_scores.append(match_score)
        prob_row = [0.0] * len(candidate_labels)
        if predicted_class is not None:
            prob_row[predicted_class] = 1.0
        probs_rows.append(prob_row)

    extra_cols = {}
    if mcqa_options:
        if answer_included_list:
            extra_cols["answer_included"] = answer_included_list
        if mcqa_correct_answers:
            extra_cols["mcqa_correct_answer"] = mcqa_correct_answers
        if chosen_options:
            extra_cols["chosen_option"] = chosen_options

    if num_context_examples_list:
        extra_cols["num_context_examples"] = num_context_examples_list
    if num_context_classes_list:
        extra_cols["num_context_classes"] = num_context_classes_list
    if total_input_tokens_list:
        extra_cols["total_input_tokens"] = total_input_tokens_list

    y_true_adjusted = y_true.copy() if not mcqa_options else None
    if mcqa_options:
        y_true_adjusted = [
            candidate_labels.index(ans) if ans in candidate_labels else -1
            for ans in mcqa_correct_answers
        ]

    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true_adjusted,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores,
        **extra_cols,
    )
