import torch
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, save_classification_results


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

    model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=args["dtype"], device_map="auto")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_type)
    tokenizer = vl_chat_processor.tokenizer
    
    if lora_model:
        model.load_adapter(trained_weights)
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        
        # images = batch_images(batch)

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

            # parse the generated text to find the predicted class
            predicted_class = None
            generated_lower = generated_text.lower()
            
            for idx, label in enumerate(candidate_labels):
                if label.lower() in generated_lower:
                    predicted_class = idx
                    break
            
            # if no match found, default to first class
            if predicted_class is None:
                predicted_class = 0
            
            preds_ids.append(predicted_class)
            
            # create one-hot encoded probabilities (generative models don't provide confidence scores)
            prob_row = [0.0] * len(candidate_labels)
            prob_row[predicted_class] = 1.0
            probs_rows.append(prob_row)
        
    # save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir
    )