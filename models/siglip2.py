import torch
import os

from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import batched, batch_images, save_classification_results


def test(args: dict, model_type: str, dataset: str, output_dir: str):

    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))

    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    candidate_labels = class_names

    model = AutoModel.from_pretrained(
        model_type,
        dtype=args["dtype"],
        device_map="auto",
        attn_implementation=args["attn_implementation"],
    )
    processor = AutoProcessor.from_pretrained(model_type)

    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []

    for batch in tqdm(list(batched(paths, args["batch_size"])), desc="Testing"):

        images = batch_images(batch)
        texts = [args["prompt_template"].format(label) for label in candidate_labels]
        inputs = processor(
            text=texts, images=images, padding="max_length", max_num_patches=256, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "logits_per_image"):
                logits = outputs.logits_per_image
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                raise ValueError("Model outputs do not contain logits.")
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

        top_ids = probs.argmax(axis=1).tolist()
        preds_ids.extend(top_ids)
        probs_rows.extend(probs.tolist())

    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir,
    )
