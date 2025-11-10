import json
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix,
    f1_score
)

from tasks.classification import  parse_class_to_dataset

def compute_per_dataset_metrics(dataset_names, y_true_arr, y_pred_arr, labels_list, split_name="val"):

    # map each sample to its dataset by parsing the class name prefix
    dataset_indices = []
    for label_id in y_true_arr:
        class_name = labels_list[label_id]
        dataset_name, _ = parse_class_to_dataset(class_name)
        dataset_indices.append(dataset_name)
    
    # compute metrics per dataset
    per_dataset_metrics = []
    for ds_name in dataset_names:
        mask = [i for i, d in enumerate(dataset_indices) if d == ds_name]
        if len(mask) > 0:
            ds_y_true = y_true_arr[mask]
            ds_y_pred = y_pred_arr[mask]
            
            ds_acc = accuracy_score(ds_y_true, ds_y_pred)
            ds_f1 = f1_score(ds_y_true, ds_y_pred, average='macro')
            
            # get classes for this dataset
            ds_classes = [labels_list[i] for i in sorted(set(ds_y_true))]
            
            per_dataset_metrics.append({
                "dataset": ds_name,
                "n_samples": len(mask),
                "n_classes": len(ds_classes),
                "accuracy": ds_acc,
                "f1_macro": ds_f1
            })
            
            print(f"\n{ds_name} ({split_name}):")
            print(f"  Samples: {len(mask)}")
            print(f"  Classes: {len(ds_classes)}")
            print(f"  Accuracy: {ds_acc:.4f}")
            print(f"  F1 Score: {ds_f1:.4f}")
        else:
            print(f"\nWarning: No samples found for {ds_name} in {split_name} set")
    
    # check for unknown assignments
    unknown_mask = [i for i, d in enumerate(dataset_indices) if d == "unknown"]
    if len(unknown_mask) > 0:
        print(f"\nWarning: {len(unknown_mask)} {split_name} samples could not be mapped to a dataset")
        unknown_classes = list(set([labels_list[y_true_arr[i]] for i in unknown_mask[:20]]))
        print(f"Sample unknown classes: {unknown_classes[:10]}")
    
    return per_dataset_metrics

def batched(iterable, n):
    
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]
        
def batch_images(batch: list):
    
    imgs = []
    for p in batch:

        try:
            imgs.append(Image.open(p). convert("RGB"))
            
        except Exception as e:
            print(f"Error loading image {p}: {e}")
            continue
        
    return imgs
        
def save_classification_results( # TODO: ADD DATA TYPES
    candidate_labels,
    preds_ids,
    probs_rows,
    df,
    y_true,
    output_dir='outputs'
):
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # prepare output dataframes
    predictions_csv = f"{output_dir}/predictions.csv"
    metrics_csv = f"{output_dir}/metrics.csv"
    per_class_csv = f"{output_dir}/per_class.csv"
    confusion_matrix_csv = f"{output_dir}/confusion_matrix.csv"
    
    # Debug: Check dimensions
    print(f"Number of candidate labels: {len(candidate_labels)}")
    print(f"Number of predictions: {len(preds_ids)}")
    print(f"Number of probability rows: {len(probs_rows)}")
    if probs_rows:
        print(f"Shape of first prob row: {len(probs_rows[0])}")
    
    # Ensure probabilities match candidate labels length
    if probs_rows and len(probs_rows[0]) != len(candidate_labels):
        print(f"WARNING: Probability dimension ({len(probs_rows[0])}) != number of labels ({len(candidate_labels)})")
        print("Truncating or padding probabilities to match labels...")
        
        # Pad or truncate each probability row
        fixed_probs_rows = []
        for p in probs_rows:
            if len(p) < len(candidate_labels):
                # Pad with zeros
                fixed_p = list(p) + [0.0] * (len(candidate_labels) - len(p))
            else:
                # Truncate
                fixed_p = list(p[:len(candidate_labels)])
            fixed_probs_rows.append(fixed_p)
        probs_rows = fixed_probs_rows
    
    pred_labels = [candidate_labels[i] for i in preds_ids]
    top_scores = [row[idx] for row, idx in zip(probs_rows, preds_ids)]

    out_df = pd.DataFrame({
        "id": df["id"],
        "image_path": df["image_path"],
        "label": df["label"],
        "pred_label": pred_labels,
        "pred_score": top_scores,
    })
    
    out_df["probs_json"] = [
        json.dumps({candidate_labels[j]: float(p[j]) for j in range(len(candidate_labels))})
        for p in probs_rows
    ]
    
    out_df.to_csv(predictions_csv, index=False)

    # metrics (overall + per-class + confusion matrix)
    acc = accuracy_score(y_true, preds_ids)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, preds_ids, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, preds_ids, average="macro", zero_division=0)

    pd.DataFrame([{
        "accuracy": acc,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m,
        "num_classes": len(candidate_labels),
        "num_images": len(df),
    }]).to_csv(metrics_csv, index=False)

    # per-class
    report = classification_report(
        y_true, preds_ids,
        target_names=candidate_labels,
        output_dict=True, zero_division=0
    )
    
    per_class_rows = []
    for cls in candidate_labels:
        r = report.get(cls, {})
        per_class_rows.append({
            "class": cls,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": r.get("support", 0),
        })
        
    pd.DataFrame(per_class_rows).to_csv(per_class_csv, index=False)

    # confusion matrix
    cm = confusion_matrix(y_true, preds_ids, labels=list(range(len(candidate_labels))))
    
    cm_df = pd.DataFrame(
        cm,
        index=[f"true::{c}" for c in candidate_labels],
        columns=[f"pred::{c}" for c in candidate_labels],
    )
    
    cm_df.to_csv(confusion_matrix_csv, index=True)
    print(f"Saved classification results to {output_dir}")
    
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Reds, figsize=(10, 8)):

    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
