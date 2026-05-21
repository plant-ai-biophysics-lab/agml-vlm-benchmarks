import pandas as pd
from pathlib import Path
import sys

# Assume utils is in Python path since we are in vlm_investigation
sys.path.append('.')
from utils.utils import fuzzy_match_label

model_dir = Path("/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/zero_shot_classification/oeq_1/gemini-3-pro-preview")
datasets = sorted([d for d in model_dir.iterdir() if d.is_dir()])

thresholds = [0.5, 0.6, 0.7]
results = {t: 0 for t in thresholds}
total_samples = 0

for dataset_dir in datasets:
    preds_csv = dataset_dir / "predictions.csv"
    if preds_csv.exists():
        df = pd.read_csv(preds_csv)
        classes = df['label'].unique().tolist()
        
        for _, row in df.iterrows():
            total_samples += 1
            gen_text = str(row['generated_text'])
            true_label = row['label']
            
            for t in thresholds:
                pred, score = fuzzy_match_label(gen_text, classes, threshold=t)
                if pred == true_label:
                    results[t] += 1

print("--- Fuzzy Matching Sensitivity Analysis ---")
print(f"Total Samples (across all datasets): {total_samples}")
for t in thresholds:
    acc = results[t] / total_samples if total_samples else 0
    print(f"Threshold {t:.1f}: {results[t]} correct -> {acc:.4f} Accuracy")
