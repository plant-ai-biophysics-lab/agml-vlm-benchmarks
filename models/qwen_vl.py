import torch
import os

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, save_classification_results, fuzzy_match_label
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes

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
    processor = AutoProcessor.from_pretrained(model_type)
    
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

def test(args: dict, model_type: str, dataset: str, output_dir: str, lora_model: bool = False, trained_weights: str = None):
    
    # get dataset path
    if not lora_model:
        dataset_path = load_agml_dataset(dataset)
    else:
        dataset_path = dataset
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a subset
    sample_limit = args.get("sample_limit", None)  # Default to full dataset
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)
    
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
        print(f"\nMCQA Mode Enabled:")
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
        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_type, torch_dtype=args["dtype"], device_map="auto", attn_implementation=args["attn_implementation"])
    processor = AutoProcessor.from_pretrained(model_type)
    
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
    
    sample_index = 0
    batch_start_index = 0
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        
        # prepare batch of conversations
        conversations = []
        image_inputs_list = []
        batch_prompts = []
        
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
            conversations.append(conversation)
            sample_index += 1
            
            # process vision info for this conversation
            img_inputs, _ = process_vision_info(conversation)
            image_inputs_list.extend(img_inputs)  # Extend, not append
        
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

        with torch.no_grad():
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
