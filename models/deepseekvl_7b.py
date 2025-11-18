import torch
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, batch_images, save_classification_results, fuzzy_match_label

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
    
    # LLaVa-Next uses conversational format
    conversation_template = args["prompt_template"].format(classes=classes_str)
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
    
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):

        for image in batch: # sequential processing due to conversational nature

            conversation = [
                {
                    "role": "User",
                    "content": conversation_template,
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
            
            # fuzzy matching to find the predicted class
            predicted_class, match_score, matched_label = fuzzy_match_label(
                generated_text, candidate_labels, threshold=0.6
            )
            
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
        
    # save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores
    )