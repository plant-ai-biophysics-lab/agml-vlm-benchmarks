import torch
import os

from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, SiglipForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

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

def collate_fn(batch):

    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    
    return {"pixel_values": pixel_values, "labels": labels}

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

def train(args: dict, model_type: str, dataset: str | dict, output_dir: str):

    def _preprocess_train(batch):
        batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in batch["image"]]
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
        train_ds = train_ds_full["train"]
        
    else:

        dataset_path = load_agml_dataset(dataset)
        ds = load_dataset("imagefolder", data_dir=dataset_path)
        train_ds = ds["train"]

    # prep labels from training data
    labels_list = sorted(train_ds.features["label"].names)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label

    # train configs
    model = SiglipForImageClassification.from_pretrained(
        model_type, 
        num_labels=len(labels_list),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    processor = AutoProcessor.from_pretrained(model_type)
    lora_config = LoraConfig(**args["lora_config"])
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # train
    train_transforms, val_transforms = get_transforms(processor.image_processor)
    train_ds.set_transform(_preprocess_train)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        **args["trainer_config"]
    )
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        tokenizer=processor.image_processor,
        data_collator=collate_fn,
        compute_metrics=None,
    )
    trainer.train()
    trainer.save_model() 
    
    # test on validation set after training
    test(
        args,
        model_type=model_type,
        dataset=val_dataset_path,
        output_dir=output_dir,
        lora_model=True,
        trained_weights=output_dir,
        label2id=label2id,
        id2label=id2label
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

    # build texts (only for zero-shot models)
    candidate_labels = class_names
    
    if lora_model:

        peft_config = PeftConfig.from_pretrained(trained_weights)
        model = SiglipForImageClassification.from_pretrained(
            peft_config.base_model_name_or_path, 
            label2id=kwargs.get("label2id"), 
            id2label=kwargs.get("id2label"),
            ignore_mismatched_sizes=True
        )
        model = PeftModel.from_pretrained(
            model, trained_weights
        )
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        use_text = False
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
            
            # Use softmax for classification (not sigmoid)
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