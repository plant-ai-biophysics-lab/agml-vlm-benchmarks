import os

import torch
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Backward compatibility with older Transformers versions
    from transformers import AutoModelForCausalLM as AutoModelForImageTextToText

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from typing import Optional

from tasks.classification import agml_to_df, load_agml_dataset
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes
from utils.utils import batched, fuzzy_match_label, save_classification_results
from utils.prep_context import create_classification_message, build_prompt_descriptions
from utils.gpu import print_gpu_utilization, print_summary


def format_data(image, prompt, label):
    return {
      "images": [image],
      "messages": [
          {
              "role": "user",
              "content": [
                  {"type": "image", "image": image},
                  {"type": "text", "text": prompt}
              ],
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": label}
              ],
          },
      ]
    }


def train(args: dict, model_type: str, dataset: str | dict, output_dir: str):
    if isinstance(dataset, dict):
        train_dataset = dataset.get("train")
        val_dataset = dataset.get("val")
        print(f"Training on: {len(train_dataset) if isinstance(train_dataset, list) else 1} dataset(s)")
        print(f"Testing on: {len(val_dataset) if isinstance(val_dataset, list) else 1} dataset(s)")
        dataset_path = load_agml_dataset(train_dataset, split_name="train")
        val_dataset_path = load_agml_dataset(val_dataset, split_name="val")
        train_ds_full = load_dataset("imagefolder", data_dir=os.path.join(dataset_path, "train"))
        train_ds = train_ds_full["train"]
    else:
        dataset_path = load_agml_dataset(dataset)
        ds = load_dataset("imagefolder", data_dir=dataset_path)
        train_ds = ds["train"]
        val_dataset_path = dataset_path
        
    class_names = sorted(train_ds.features["label"].names)
    candidate_labels = class_names
    classes_str = ", ".join(candidate_labels)
    conversation_template = args["prompt_template"].format(classes=classes_str)
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(class_names):
        label2id[label] = i
        id2label[i] = label

    train_ds = [
        format_data(sample["image"], conversation_template, id2label[sample["label"]]) for sample in train_ds
    ]

    torch_dtype = args.get("dtype", torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        model_type,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.get("attn_implementation", "sdpa")
    )
    processor = AutoProcessor.from_pretrained(model_type)

    training_args = SFTConfig(
        output_dir=output_dir,
        report_to="none",
        push_to_hub=False,
        **args["trainer_config"]
    )
    lora_config = LoraConfig(**args["lora_config"])
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=processor,
        peft_config=lora_config
    )

    trainer.train()
    trainer.save_model(output_dir)

    test(
        args,
        model_type=model_type,
        dataset=val_dataset_path,
        output_dir=output_dir,
        lora_model=True,
        trained_weights=output_dir
    )


def test(
    args: dict,
    model_type: str,
    dataset: str,
    output_dir: str,
    lora_model: bool = False,
    trained_weights: str = None,
    context: dict = None,
    max_num_class_context: Optional[int] = None,
    include_correct_class: bool = True,
    random_pool: bool = False
):
    dataset_path = dataset if lora_model else load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))

    sample_limit = args.get("sample_limit", None)
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)

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

    print(f"Using model {model_type} for testing")
    print_gpu_utilization()

    torch_dtype = args.get("dtype", torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        model_type,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.get("attn_implementation", "sdpa"),
    )
    processor = AutoProcessor.from_pretrained(model_type)

    if lora_model and trained_weights:
        model.load_adapter(trained_weights)

    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    chosen_options = []
    num_context_examples_list = []
    num_context_classes_list = []
    total_input_tokens_list = []

    def _print_sample_prompt_once(conversation):
        nonlocal sample_prompt_printed
        if sample_prompt_printed:
            return
        content = conversation[0].get("content", [])
        preview_lines = []
        for idx, block in enumerate(content, start=1):
            if block.get("type") == "text":
                preview_lines.append(f"{idx}. text: {block.get('text')}")
            elif block.get("type") == "image":
                preview_lines.append(f"{idx}. image: {os.path.basename(block.get('image', ''))}")
            else:
                preview_lines.append(f"{idx}. {block}")
        print("--- Sample prompt (first item this run) ---", flush=True)
        print("\n".join(preview_lines), flush=True)
        print("------------------------------------------", flush=True)
        sample_prompt_printed = True

    sample_prompt_printed = False
    sample_index = 0
    batch_start_index = 0

    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        conversations = []

        for image in batch:
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
                prompt_text = conversation_template.format(classes=", ".join(choices))
            else:
                prompt_text = conversation_template

            if context is not None:
                message, context_meta = create_classification_message(
                    task=None,
                    template=prompt_text,
                    query_image_path=image,
                    context_examples=context,
                    max_num_class_context=max_num_class_context,
                    correct_class=df.iloc[sample_index]["label"],
                    include_correct_class=include_correct_class,
                    random_pool=random_pool,
                    prepend_text=prepend_blocks,
                )
                conversation = [message]
                num_context_examples_list.append(context_meta["num_context_examples"])
                num_context_classes_list.append(context_meta["num_context_classes"])
                _print_sample_prompt_once(conversation)
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                num_context_examples_list.append(0)
                num_context_classes_list.append(0)
                _print_sample_prompt_once(conversation)

            conversations.append(conversation)
            sample_index += 1

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch_dtype)

        if "attention_mask" in inputs:
            per_sample_input_tokens = inputs.attention_mask.sum(dim=1).tolist()
            total_input_tokens_list.extend([int(t) for t in per_sample_input_tokens])

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            outputs_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            batch_generated_texts = [
                processor.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for out in outputs_trimmed
            ]

        for batch_idx, generated_text in enumerate(batch_generated_texts):
            generated_texts.append(generated_text)

            if mcqa_options:
                current_sample_idx = batch_start_index + batch_idx
                sample_choices = mcqa_choices_list[current_sample_idx]
                correct_answer = mcqa_correct_answers[current_sample_idx]
                
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
                    print(f"WARNING [Sample {current_sample_idx}]: No match found")
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

        batch_start_index = sample_index

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
        y_true_adjusted = []
        for correct_ans in mcqa_correct_answers:
            if correct_ans in candidate_labels:
                y_true_adjusted.append(candidate_labels.index(correct_ans))
            else:
                y_true_adjusted.append(-1)

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
