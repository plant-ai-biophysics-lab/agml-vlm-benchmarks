import torch
import os

from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, batch_images, save_classification_results, fuzzy_match_label

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
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_type,
        torch_dtype=args["dtype"],
        device_map="auto",
        attn_implementation=args["attn_implementation"]
    )
    processor = LlavaNextProcessor.from_pretrained(model_type)

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

    model = LlavaNextForConditionalGeneration.from_pretrained(model_type, torch_dtype=args["dtype"], device_map="auto", attn_implementation=args["attn_implementation"])
    processor = LlavaNextProcessor.from_pretrained(model_type)
    
    if lora_model:
        model.load_adapter(trained_weights)
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        
        images = batch_images(batch)
        
        # Prepare batch of conversations and prompts
        conversations = []
        prompts = []
        
        for image in images:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": conversation_template},
                    ],
                },
            ]
            conversations.append(conversation)
            
            # Apply chat template for each conversation
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            prompts.append(prompt)
        
        # Process all images and prompts together
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # Decode all outputs in the batch
            batch_generated_texts = [
                processor.decode(output, skip_special_tokens=True) 
                for output in outputs
            ]
        
        # Process each generated text in the batch
        for generated_text in batch_generated_texts:
            generated_texts.append(generated_text)
            
            # Fuzzy matching to find the predicted class
            predicted_class, match_score, matched_label = fuzzy_match_label(
                generated_text, candidate_labels, threshold=0.6
            )

            # If no match found, keep as None (for open-ended evaluation)
            if predicted_class is None:
                match_score = 0.0
                print(f"WARNING: No match found for: '{generated_text}'")
            
            preds_ids.append(predicted_class)
            match_scores.append(match_score)
            
            # Create one-hot encoded probabilities
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