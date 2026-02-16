import json
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from difflib import SequenceMatcher
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix,
    f1_score
)

from tasks.classification import  parse_class_to_dataset

def fuzzy_match_label(generated_text: str, candidate_labels: list, threshold: float = 0.6) -> tuple:
    """
    Fuzzy match generated text to candidate labels.
    
    Args:
        generated_text: The generated text from the model
        candidate_labels: List of possible class labels
        threshold: Minimum similarity score (0-1) to consider a match
    
    Returns:
        tuple: (matched_index, similarity_score, matched_label) or (None, 0, None) if no match
    """
    # Clean and normalize the generated text
    generated_lower = generated_text.lower().strip()
    
    # Remove common prefixes/suffixes that models sometimes add
    prefixes_to_remove = ['the answer is', 'category:', 'class:', 'label:', 'answer:', 'it is', 'this is']
    for prefix in prefixes_to_remove:
        if generated_lower.startswith(prefix):
            generated_lower = generated_lower[len(prefix):].strip()
    
    # Remove trailing punctuation
    generated_lower = generated_lower.rstrip('.,;:!?')
    
    best_match_idx = None
    best_score = 0.0
    best_label = None
    
    for idx, label in enumerate(candidate_labels):
        label_lower = label.lower().strip()
        
        # Exact match (highest priority)
        if label_lower == generated_lower:
            return idx, 1.0, label
        
        # Check if label is contained in generated text or vice versa
        if label_lower in generated_lower or generated_lower in label_lower:
            return idx, 1.0, label
        
        # Fuzzy match using sequence matcher
        similarity = SequenceMatcher(None, generated_lower, label_lower).ratio()
        
        # Also check if label words are in generated text
        label_words = set(label_lower.replace('_', ' ').replace('-', ' ').split())
        generated_words = set(generated_lower.replace('_', ' ').replace('-', ' ').split())
        word_overlap = len(label_words & generated_words) / len(label_words) if label_words else 0
        
        # Take the maximum of both similarity measures
        combined_score = max(similarity, word_overlap)
        
        if combined_score > best_score:
            best_score = combined_score
            best_match_idx = idx
            best_label = label
    
    # Only return match if above threshold
    if best_score >= threshold:
        return best_match_idx, best_score, best_label
    else:
        return None, best_score, None

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
        
def save_classification_results(
    candidate_labels,
    preds_ids,
    probs_rows,
    df,
    y_true,
    output_dir='outputs',
    generated_texts=None,
    match_scores=None,
    **extra_columns
):
    """
    Save classification results with optional generated texts and match scores for debugging.
    Handles MCQA mode where "None of the above" may be the correct answer.
    
    Args:
        candidate_labels: List of possible class labels
        preds_ids: List of predicted class indices
        probs_rows: List of probability distributions
        df: DataFrame with image paths and true labels
        y_true: Array of true class indices (may include -1 for "None of the above" in MCQA)
        output_dir: Directory to save results
        generated_texts: Optional list of raw generated texts from models
        match_scores: Optional list of fuzzy match scores
        **extra_columns: Additional columns to add to the predictions CSV (e.g., answer_included, mcqa_correct_answer, chosen_option)
    """
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if this is MCQA mode (if y_true contains -1 or mcqa_correct_answer is in extra_columns)
    is_mcqa = 'mcqa_correct_answer' in extra_columns or (hasattr(y_true, '__iter__') and -1 in y_true)
    
    # For MCQA, add "None of the above" to candidate labels for metrics
    if is_mcqa:
        extended_labels = candidate_labels + ["None of the above"]
        # Map -1 to the index of "None of the above"
        none_of_above_idx = len(candidate_labels)
        y_true = [none_of_above_idx if y == -1 else y for y in y_true]
        preds_ids = [none_of_above_idx if p == -1 else p for p in preds_ids]
        
        # Extend probability rows to include "None of the above"
        extended_probs_rows = []
        for p in probs_rows:
            extended_p = list(p) + [0.0]  # Add 0 probability for "None of the above"
            extended_probs_rows.append(extended_p)
        probs_rows = extended_probs_rows
        
        labels_for_metrics = extended_labels
    else:
        labels_for_metrics = candidate_labels
    
    # prepare output dataframes
    predictions_csv = f"{output_dir}/predictions.csv"
    metrics_csv = f"{output_dir}/metrics.csv"
    per_class_csv = f"{output_dir}/per_class.csv"
    confusion_matrix_csv = f"{output_dir}/confusion_matrix.csv"
    debug_log = f"{output_dir}/debug_log.jsonl"
    
    # Debug: Check dimensions
    print(f"Number of candidate labels: {len(candidate_labels)}")
    print(f"Number of labels for metrics: {len(labels_for_metrics)}")
    print(f"Number of predictions: {len(preds_ids)}")
    print(f"Number of probability rows: {len(probs_rows)}")
    if probs_rows:
        print(f"Shape of first prob row: {len(probs_rows[0])}")
    
    # Ensure probabilities match labels_for_metrics length
    if probs_rows and len(probs_rows[0]) != len(labels_for_metrics):
        print(f"WARNING: Probability dimension ({len(probs_rows[0])}) != number of labels ({len(labels_for_metrics)})")
        print("Truncating or padding probabilities to match labels...")
        
        # Pad or truncate each probability row
        fixed_probs_rows = []
        for p in probs_rows:
            if len(p) < len(labels_for_metrics):
                # Pad with zeros
                fixed_p = list(p) + [0.0] * (len(labels_for_metrics) - len(p))
            else:
                # Truncate
                fixed_p = list(p[:len(labels_for_metrics)])
            fixed_probs_rows.append(fixed_p)
        probs_rows = fixed_probs_rows
    
    # Handle None values in preds_ids (for non-matches in open-ended evaluation)
    pred_labels = [labels_for_metrics[i] if i is not None else None for i in preds_ids]
    top_scores = [row[idx] if idx is not None else 0.0 for row, idx in zip(probs_rows, preds_ids)]

    # If generated_texts are available, map fuzzy matches for "None of the above"
    # to the MCQA none_of_above index so excluded samples are counted correctly.
    if is_mcqa and generated_texts is not None:
        for i, gen in enumerate(generated_texts):
            if i >= len(preds_ids):
                break
            if preds_ids[i] is None and isinstance(gen, str) and gen.strip():
                try:
                    m_idx, m_score, m_label = fuzzy_match_label(gen, labels_for_metrics)
                except Exception:
                    m_idx = None
                    m_score = 0.0
                    m_label = None
                if m_idx is not None and m_label and 'none' in m_label.lower():
                    # assign the matched "None of the above" index
                    preds_ids[i] = m_idx
                    pred_labels[i] = m_label
                    if probs_rows and i < len(probs_rows) and m_idx < len(probs_rows[i]):
                        top_scores[i] = probs_rows[i][m_idx]
                    if match_scores is not None:
                        match_scores[i] = m_score

    out_df = pd.DataFrame({
        "id": df["id"],
        "image_path": df["image_path"],
        "label": df["label"],
        "pred_label": pred_labels,
        "pred_score": top_scores,
    })
    
    # Add generated text and match scores if available
    if generated_texts is not None:
        out_df["generated_text"] = generated_texts
    if match_scores is not None:
        out_df["match_score"] = match_scores
    
    # Add any extra columns (e.g., answer_included for MCQA)
    for col_name, col_data in extra_columns.items():
        if col_data is not None:
            out_df[col_name] = col_data
    
    out_df["probs_json"] = [
        json.dumps({labels_for_metrics[j]: float(p[j]) for j in range(len(labels_for_metrics))})
        for p in probs_rows
    ]
    
    out_df.to_csv(predictions_csv, index=False)
    
    # Save detailed debug log if generated texts are available
    if generated_texts is not None:
        with open(debug_log, 'w') as f:
            for i in range(len(df)):
                log_entry = {
                    "id": df["id"].iloc[i],
                    "image_path": df["image_path"].iloc[i],
                    "true_label": df["label"].iloc[i],
                    "pred_label": pred_labels[i],
                    "generated_text": generated_texts[i] if i < len(generated_texts) else None,
                    "match_score": match_scores[i] if match_scores and i < len(match_scores) else None,
                    "correct": df["label"].iloc[i] == pred_labels[i]
                }
                f.write(json.dumps(log_entry) + '\n')
        print(f"Debug log saved to: {debug_log}")

    # Prepare predictions for metrics (convert None to a new index beyond valid range)
    # This ensures None predictions are counted as incorrect
    max_label_idx = len(labels_for_metrics)
    preds_ids_for_metrics = [p if p is not None else max_label_idx for p in preds_ids]
    
    # Calculate metrics
    acc = accuracy_score(y_true, preds_ids_for_metrics)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, preds_ids_for_metrics, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, preds_ids_for_metrics, average="macro", zero_division=0)
    
    # Count non-matches
    num_non_matches = sum(1 for p in preds_ids if p is None)

    metrics_dict = {
        "accuracy": acc,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m,
        "num_classes": len(labels_for_metrics),
        "num_images": len(df),
        "num_non_matches": num_non_matches,
    }
    
    # Add MCQA-specific metrics if applicable
    if is_mcqa and 'answer_included' in extra_columns:
        answer_included = extra_columns['answer_included']
        # Calculate accuracy for samples where answer was included
        included_mask = [i for i, inc in enumerate(answer_included) if inc]
        if included_mask:
            y_true_included = [y_true[i] for i in included_mask]
            preds_included = [preds_ids_for_metrics[i] for i in included_mask]
            acc_included = accuracy_score(y_true_included, preds_included)
            metrics_dict['accuracy_answer_included'] = acc_included
        
        # Calculate accuracy for samples where answer was NOT included
        excluded_mask = [i for i, inc in enumerate(answer_included) if not inc]
        if excluded_mask:
            y_true_excluded = [y_true[i] for i in excluded_mask]
            preds_excluded = [preds_ids_for_metrics[i] for i in excluded_mask]
            acc_excluded = accuracy_score(y_true_excluded, preds_excluded)
            metrics_dict['accuracy_answer_not_included'] = acc_excluded
        
        metrics_dict['num_answer_included'] = len(included_mask)
        metrics_dict['num_answer_not_included'] = len(excluded_mask)
    
    pd.DataFrame([metrics_dict]).to_csv(metrics_csv, index=False)

    # per-class
    report = classification_report(
        y_true, preds_ids_for_metrics,
        labels=list(range(len(labels_for_metrics))),  # Use all labels including "None of the above" if MCQA
        target_names=labels_for_metrics,
        output_dict=True, zero_division=0
    )
    
    per_class_rows = []
    for cls in labels_for_metrics:
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
    cm = confusion_matrix(y_true, preds_ids_for_metrics, labels=list(range(len(labels_for_metrics))))
    
    cm_df = pd.DataFrame(
        cm,
        index=[f"true::{c}" for c in labels_for_metrics],
        columns=[f"pred::{c}" for c in labels_for_metrics],
    )
    
    cm_df.to_csv(confusion_matrix_csv, index=True)
    
    print(f"Saved classification results to {output_dir}")
    if num_non_matches > 0:
        print(f"Warning: {num_non_matches}/{len(df)} predictions had no fuzzy match (returned None)")
    if is_mcqa:
        print(f"MCQA mode: Added 'None of the above' to labels for metrics and confusion matrix")
    
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
