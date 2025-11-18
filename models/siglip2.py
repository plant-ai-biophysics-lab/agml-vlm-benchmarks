import torch
import os
import numpy as np
import evaluate
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer, SiglipModel
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from functools import partial

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, batch_images, save_classification_results

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
    CenterCrop,
)

class SiglipContrastiveTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(**inputs)
        
        # get the similarity logits
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        # create labels: diagonal elements should be 1 (matching pairs), others 0
        batch_size = logits_per_image.shape[0]
        labels = torch.eye(batch_size, device=logits_per_image.device)
        
        # SigLIP uses sigmoid loss (binary cross-entropy)
        # each pair is treated independently
        loss_img = F.binary_cross_entropy_with_logits(
            logits_per_image, labels, reduction='mean'
        )
        loss_txt = F.binary_cross_entropy_with_logits(
            logits_per_text, labels, reduction='mean'
        )
        
        # average the two losses
        loss = (loss_img + loss_txt) / 2
        
        return (loss, outputs) if return_outputs else loss

def collate_fn_contrastive(batch, processor, prompt_template, label_names):
    
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    
    labels = [label_names[example["label"]] for example in batch]
    texts = [prompt_template.format(label) for label in labels]
    
    text_inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    
    return_dict = {
        "pixel_values": pixel_values,
        "input_ids": text_inputs["input_ids"],
    }

    return return_dict

# def collate_fn(batch):

#     pixel_values = torch.stack([example["pixel_values"] for example in batch])
#     labels = torch.tensor([example["label"] for example in batch])
    
#     return {"pixel_values": pixel_values, "labels": labels}

def get_transforms(image_processor):
    
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    
    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )
    
    return train_transforms, val_transforms

def compute_metrics(eval_pred):
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        
        return accuracy.compute(predictions=predictions, references=labels)

def train(args: dict, model_type: str, dataset: str | dict, output_dir: str):

    def _preprocess_train(batch):
        batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in batch["image"]]
        return batch
    
    def _preprocess_val(batch):
        batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in batch["image"]]
        return batch
    
    # handle different dataset input formats
    if isinstance(dataset, dict):

        train_dataset = dataset.get("train")
        val_dataset = dataset.get("val")
        
        print(f"Training on: {len(train_dataset) if isinstance(train_dataset, list) else 1} dataset(s)")
        print(f"Testing on: {len(val_dataset) if isinstance(val_dataset, list) else 1} dataset(s)")

        dataset_path = load_agml_dataset(train_dataset, split_name="train")
        val_dataset_path = load_agml_dataset(val_dataset, split_name="val")
        train_ds_full = load_dataset("imagefolder", data_dir=os.path.join(dataset_path, "train"))
        
        # split the training dataset 80/20 for train/validation
        split_ds = train_ds_full["train"].train_test_split(test_size=0.2, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]
        
    else:

        dataset_path = load_agml_dataset(dataset)
        ds_full = load_dataset("imagefolder", data_dir=dataset_path)
        
        # split the dataset 80/20 for train/validation
        split_ds = ds_full["train"].train_test_split(test_size=0.2, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]

    # prep labels from training data
    labels_list = sorted(train_ds.features["label"].names)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label

    # Use base SigLIP model for contrastive learning (keeps zero-shot capability)
    model = SiglipModel.from_pretrained(model_type)
    processor = AutoProcessor.from_pretrained(model_type)
    lora_config = LoraConfig(**args["lora_config"])
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    collate_fn_with_text = partial(
        collate_fn_contrastive,
        processor=processor,
        prompt_template=args["prompt_template"],
        label_names=labels_list
    )
    
    # train
    train_transforms, val_transforms = get_transforms(processor.image_processor)
    train_ds.set_transform(_preprocess_train)
    val_ds.set_transform(_preprocess_val)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="epoch",
        logging_steps=10,
        **args["trainer_config"]
    )
    trainer = SiglipContrastiveTrainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.image_processor,
        data_collator=collate_fn_with_text,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model() 
    
    # test on validation set after training using zero-shot with LoRA-adapted encoder
    test(
        args,
        model_type=model_type,
        dataset=val_dataset_path,
        output_dir=output_dir,
        lora_model=True,
        trained_weights=output_dir
    )

def test(args: dict, model_type: str, dataset: str, output_dir: str, lora_model: bool = False, trained_weights: str = None, **kwargs):

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

    # build texts for zero-shot classification
    candidate_labels = class_names
    
    if lora_model:

        peft_config = PeftConfig.from_pretrained(trained_weights)
        model = SiglipModel.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, trained_weights)
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        use_text = True  # Use text prompts for zero-shot classification
        
    else:

        model = AutoModel.from_pretrained(model_type, dtype = args["dtype"], device_map = "auto", attn_implementation = args["attn_implementation"])
        use_text = True
        
    processor = AutoProcessor.from_pretrained(model_type)
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    
    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):
        
        images = batch_images(batch)

        if use_text:

            texts = [args["prompt_template"].format(label) for label in candidate_labels]
            inputs = processor(text=texts, images=images, padding="max_length", max_num_patches=256, return_tensors="pt").to(model.device)
        else:

            inputs = processor(images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, "logits_per_image"):
                logits = outputs.logits_per_image
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                raise ValueError("Model outputs do not contain logits.")
            
            # use softmax for classification (not sigmoid)
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

        top_ids = probs.argmax(axis=1).tolist()
        preds_ids.extend(top_ids)
        probs_rows.extend(probs.tolist())
        
    # save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir
    )