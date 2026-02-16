import torch
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, batch_images, save_classification_results, fuzzy_match_label
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes

def test(args: dict, model_type: str, dataset: str, output_dir: str, lora_model: bool = False, trained_weights: str = None):

    # get dataset path
    if not lora_model:
        dataset_path = load_agml_dataset(dataset)
    else:
        dataset_path = dataset
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
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
    
    # DeepSeek uses conversational format (template will be formatted per-sample for MCQA)
    conversation_template = args["prompt_template"]
    if not mcqa_options:
        conversation_template = conversation_template.format(classes=classes_str)
    print("Conversation template:", conversation_template)

    model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=args["dtype"], device_map="auto")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_type)
    tokenizer = vl_chat_processor.tokenizer
    
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
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):

        for image in batch: # sequential processing due to conversational nature
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
                prompt_text = conversation_template.format(classes=sample_choices_str)
            else:
                prompt_text = conversation_template

            conversation = [
                {
                    "role": "User",
                    "content": prompt_text,
                    "images": [image]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(model.device)
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            with torch.no_grad():
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=50,
                    do_sample=False,
                    use_cache=True
                )
                generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
            generated_texts.append(generated_text)
            
            # For MCQA, match against the choices for this sample
            if mcqa_options:
                sample_choices = mcqa_choices_list[sample_index]
                
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
                    print(f"WARNING: No match found for: '{generated_text}'")
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
            
            # create one-hot encoded probabilities (generative models don't provide confidence scores)
            prob_row = [0.0] * len(candidate_labels)
            if predicted_class is not None:
                prob_row[predicted_class] = 1.0
            probs_rows.append(prob_row)
            
            sample_index += 1
        
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