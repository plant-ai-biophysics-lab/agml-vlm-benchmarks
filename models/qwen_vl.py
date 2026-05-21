import time

import torch
import os

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from typing import Optional

from tasks.classification import load_agml_dataset, agml_to_df
from utils.prep_context import create_classification_message, build_prompt_descriptions
from utils.utils import batched, save_classification_results, fuzzy_match_label
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes
from utils.gpu import print_gpu_utilization, print_summary

def format_data(image, prompt, label):
    
    return {
      "images": [image],
      "messages": [

          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": image,
                  },
                  {
                      "type": "text",
                      "text": prompt,
                  }
              ],
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": label
                  }
              ],
          },
      ]
    }

def train(args: dict, model_type: str, dataset: str | dict, output_dir: str):
    
    # handle different dataset input formats
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
        
    # format datasets
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

    # load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_type, 
        torch_dtype=args["dtype"], 
        device_map="auto", 
        attn_implementation=args["attn_implementation"]
    )
    processor_kwargs = {}
    if args.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = args["min_pixels"]
    if args.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = args["max_pixels"]
    processor = AutoProcessor.from_pretrained(model_type, **processor_kwargs)
    if processor_kwargs:
        print(f"Image size constraint: min_pixels={processor_kwargs.get('min_pixels')}, max_pixels={processor_kwargs.get('max_pixels')}")
    
    # Fix for Qwen2.5-VL batch processing bug - set padding side to left
    # See: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/54
    processor.tokenizer.padding_side = "left"
    
    # trainer
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
    
    # test on validation set after training
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
    
    # get dataset path
    if not lora_model:
        dataset_path = load_agml_dataset(dataset)
    else:
        dataset_path = dataset
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a stratified subset
    sample_limit = args.get("sample_limit", None)  # Default to full dataset
    if sample_limit and 0 < sample_limit < 1:
        # Group by label to maintain class distribution, then sample. The second sample shuffles the result.
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=sample_limit, random_state=int(os.environ.get("RANDOM_SEED", 42)))
        ).sample(frac=1, random_state=int(os.environ.get("RANDOM_SEED", 42))).reset_index(drop=True)
    
    # prepare data
    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()

    # build prompt for generative classification
    candidate_labels = class_names
    classes_str = ", ".join(candidate_labels)
    
    # Check if MCQA mode is enabled
    mcqa_options = args.get("mcqa_options", None)
    all_dataset_classes = None
    answer_included_list = []
    mcqa_correct_answers = []  # Track the correct answer for each sample
    mcqa_choices_list = []  # Track the choices for each sample
    
    if mcqa_options:
        print("\nMCQA Mode Enabled:")
        print(f"  Options within dataset: {mcqa_options.get('options_within_dataset', True)}")
        print(f"  Number of choices: {mcqa_options.get('mcqa_num_choices', 4)}")
        
        # Load all dataset classes if needed for cross-dataset sampling
        if not mcqa_options.get('options_within_dataset', True):
            all_dataset_classes = load_all_dataset_classes()
            print(f"  Loaded {len(all_dataset_classes)} datasets for cross-dataset sampling")
    
    # conversational format (template will be formatted per-sample for MCQA)
    conversation_template = args["prompt_template"]
    if not mcqa_options:
        conversation_template = conversation_template.format(classes=classes_str)
    print("Conversation template:", conversation_template)

    # Build optional task/dataset descriptions once per dataset
    use_desc = args.get("context_options", {}).get("use_desc", False)
    prepend_blocks, context_warnings = build_prompt_descriptions(
        dataset_name=dataset,
        use_desc=use_desc,
        datasets_file=args.get("datasets_file", "datasets.txt"),
        context_file=args.get("context_file", "context.yaml"),
    )
    for warn in context_warnings:
        print(f"WARNING [context]: {warn}")
    
    if "3" in model_type:
        print("Using Qwen VL 3.0 model for testing")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_type, 
            dtype=args["dtype"], 
            device_map="auto",
            attn_implementation=args.get("attn_implementation", "sdpa")
        )
    else:
        print("Using Qwen VL 2.5 model for testing")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_type, 
            dtype=args["dtype"], 
            device_map="auto",
            attn_implementation=args.get("attn_implementation", "sdpa")
        )
    print_gpu_utilization()  # Check GPU memory before loading processor
        
    processor_kwargs = {}
    if args.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = args["min_pixels"]
    if args.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = args["max_pixels"]
    processor = AutoProcessor.from_pretrained(model_type, **processor_kwargs)
    if processor_kwargs:
        print(f"Image size constraint: min_pixels={processor_kwargs.get('min_pixels')}, max_pixels={processor_kwargs.get('max_pixels')}")
    
    # Fix for Qwen2.5-VL batch processing bug - set padding side to left
    # See: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/54
    processor.tokenizer.padding_side = "left"
    
    if lora_model:
        model.load_adapter(trained_weights)

    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    chosen_options = []  # Track which option number (1, 2, 3, ...) was chosen
    num_context_examples_list = []  # Track context image count per sample
    num_context_classes_list = []   # Track context class count per sample
    total_input_tokens_list = []    # Track actual (non-padded) input token count per sample

    def _print_sample_prompt_once(conversation):
        # Show a human-readable view of the first prompt in a run
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
        # Use flush so the sample prompt shows up immediately in batch logs
        print("--- Sample prompt (first item this run) ---", flush=True)
        print("\n".join(preview_lines), flush=True)
        print("------------------------------------------", flush=True)
        sample_prompt_printed = True

    sample_prompt_printed = False
    
    sample_index = 0
    batch_start_index = 0
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        
        # prepare batch of conversations
        conversations = []
        image_inputs_list = []
        
        for image in batch:
            # Generate MCQA choices if enabled
            if mcqa_options:
                true_label = df.iloc[sample_index]["label"]
                choices, correct_answer, answer_included, correct_answer_index = get_mcqa_choices(
                    true_label=true_label,
                    all_classes=candidate_labels,
                    options_within_dataset=mcqa_options.get('options_within_dataset', True),
                    mcqa_num_choices=mcqa_options.get('mcqa_num_choices', 4),
                    all_dataset_classes=all_dataset_classes,
                    current_dataset=dataset,
                    answer_included_ratio=0.7,
                    sample_index=sample_index,
                    print_sample=(sample_index == 0)  # Print first sample only
                )
                answer_included_list.append(answer_included)
                mcqa_correct_answers.append(correct_answer)
                mcqa_choices_list.append(choices)
                sample_choices_str = ", ".join(choices)
                prompt = conversation_template.format(classes=sample_choices_str)
            else:
                prompt = conversation_template
                
            if context is not None:
                
                # in-context learning prompt with examples
                message, context_meta = create_classification_message(
                    task=None, # NOTE: for now will define as None
                    template=prompt,
                    query_image_path=image,
                    context_examples=context,
                    max_num_class_context=max_num_class_context,
                    correct_class=df.iloc[sample_index]["label"],
                    include_correct_class=include_correct_class,
                    random_pool=random_pool,
                    prepend_text=prepend_blocks,
                    # output_path="temp_message.json"  # Optional: save the message for inspection
                )
                conversation = [message]
                num_context_examples_list.append(context_meta["num_context_examples"])
                num_context_classes_list.append(context_meta["num_context_classes"])
                _print_sample_prompt_once(conversation)
                
            else:
            
                # normal zero_shot prompt without context
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                num_context_examples_list.append(0)
                num_context_classes_list.append(0)
                _print_sample_prompt_once(conversation)
                
            conversations.append(conversation)
            sample_index += 1

        # Process vision info for all conversations in the batch at once
        image_inputs_list, _ = process_vision_info(conversations)

        # apply chat template to all conversations in batch
        prompts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        
        # process all images and texts together
        inputs = processor(
            images=image_inputs_list,
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Record actual (non-padded) input token counts per sample in this batch
        per_sample_input_tokens = inputs.attention_mask.sum(dim=1).tolist()
        total_input_tokens_list.extend([int(t) for t in per_sample_input_tokens])

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # trim outputs for each item in batch
            outputs_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            
            # decode all outputs
            batch_generated_texts = [
                processor.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for out in outputs_trimmed
            ]
        
        # process each generated text in the batch
        for batch_idx, generated_text in enumerate(batch_generated_texts):
            generated_texts.append(generated_text)
            
            # For MCQA, match against the choices for this sample
            if mcqa_options:
                current_sample_idx = batch_start_index + batch_idx
                sample_choices = mcqa_choices_list[current_sample_idx]
                correct_answer = mcqa_correct_answers[current_sample_idx]
                
                # fuzzy matching to find the predicted class from the MCQA choices
                predicted_class, match_score, matched_label = fuzzy_match_label(
                    generated_text, sample_choices, threshold=0.6
                )
                
                # Track which option was chosen (1-indexed)
                if predicted_class is not None:
                    chosen_options.append(predicted_class + 1)  # Convert to 1-indexed
                    # Map the choice back to the original candidate_labels for storing pred_id
                    matched_choice = sample_choices[predicted_class]
                    if matched_choice in candidate_labels:
                        predicted_class = candidate_labels.index(matched_choice)
                    else:
                        # It's "None of the above" or not in original labels
                        predicted_class = None
                else:
                    chosen_options.append(None)
                    match_score = 0.0
                    print(f"WARNING [Sample {current_sample_idx}]: No match found")
                    print(f"  Generated: '{generated_text}'")
                    print(f"  Choices: {sample_choices}")
                    print(f"  Correct: {correct_answer}")
            else:
                # Standard fuzzy matching to find the predicted class
                predicted_class, match_score, matched_label = fuzzy_match_label(
                    generated_text, candidate_labels, threshold=0.6
                )
                chosen_options.append(None)  # Not applicable for non-MCQA
                
                # if no match found, keep as None (for open-ended evaluation)
                if predicted_class is None:
                    match_score = 0.0
                    print(f"WARNING: No match found for: '{generated_text}'")

            preds_ids.append(predicted_class)
            match_scores.append(match_score)

            # create one-hot encoded probabilities
            prob_row = [0.0] * len(candidate_labels)
            if predicted_class is not None:
                prob_row[predicted_class] = 1.0
            probs_rows.append(prob_row)
        
        batch_start_index = sample_index
        
    # save metrics
    extra_cols = {}
    if mcqa_options:
        if answer_included_list:
            extra_cols['answer_included'] = answer_included_list
        if mcqa_correct_answers:
            extra_cols['mcqa_correct_answer'] = mcqa_correct_answers
        if chosen_options:
            extra_cols['chosen_option'] = chosen_options

    # Always track context and token scaling columns
    if num_context_examples_list:
        extra_cols['num_context_examples'] = num_context_examples_list
    if num_context_classes_list:
        extra_cols['num_context_classes'] = num_context_classes_list
    if total_input_tokens_list:
        extra_cols['total_input_tokens'] = total_input_tokens_list
    
    # For MCQA, we need to adjust y_true to reflect the correct answer
    # (which may be "None of the above" when answer is not included)
    y_true_adjusted = y_true.copy() if not mcqa_options else None
    if mcqa_options:
        y_true_adjusted = []
        for i, correct_ans in enumerate(mcqa_correct_answers):
            if correct_ans in candidate_labels:
                y_true_adjusted.append(candidate_labels.index(correct_ans))
            else:
                # "None of the above" - use a special marker (-1 will be handled in save_classification_results)
                y_true_adjusted.append(None)
        y_true_adjusted = [y if y is not None else -1 for y in y_true_adjusted]
    
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true_adjusted,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores,
        **extra_cols
    )
