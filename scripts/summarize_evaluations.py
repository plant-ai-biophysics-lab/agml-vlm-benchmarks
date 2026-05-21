import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score

def load_datasets(filepath):
    task_map = {}
    if not Path(filepath).exists():
        return task_map
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line: continue
            if clean_line.startswith('#'):
                clean_line = clean_line.lstrip('#').strip()
            if ',' not in clean_line or "dataset_name" in clean_line: continue
            parts = [p.strip() for p in clean_line.split(',')]
            if len(parts) >= 3:
                dataset_name, _, task = parts[0], parts[1], parts[2]
                task_map[dataset_name] = task
    return task_map

def summarize_evaluations(model_dir):
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Directory not found - {model_dir}")
        return

    # 1. LLM Judge Evaluation Summary
    print("=" * 115)
    print(f"{'- Dataset Name -':<45} | {'Agreements':<10} | {'Eval Samples':<12} | {'Mismatch Kappa':<15} | {'Overall Kappa':<15}")
    print("-" * 115)
    
    total_agreements = 0
    total_eval_samples = 0
    summary_data = []
    all_judge_mismatch = []
    all_human_mismatch = []
    all_judge_overall = []
    all_human_overall = []
    
    # Lists for Task Metrics (accuracy, f1-score)
    metrics_dict = {'disease': {'acc': [], 'f1': []}, 'pest/damage': {'acc': [], 'f1': []}, 'crops/weeds': {'acc': [], 'f1': []}, 'overall': {'acc': [], 'f1': []}}
    dataset_to_task = load_datasets('/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt')

    dataset_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()])
    for dataset_dir in dataset_dirs:
        if dataset_dir.name == 'plant_village_classification':
            continue
            
        # --- Gather Task Metrics from metrics.csv ---
        metrics_csv = dataset_dir / "metrics.csv"
        task = dataset_to_task.get(dataset_dir.name, "unknown")
        if metrics_csv.exists() and task in metrics_dict:
            try:
                df_m = pd.read_csv(metrics_csv)
                df_m.columns = [c.strip() for c in df_m.columns]
                acc = float(df_m['accuracy'].iloc[0])
                f1 = float(df_m['f1_macro'].iloc[0])
                
                metrics_dict[task]['acc'].append(acc)
                metrics_dict[task]['f1'].append(f1)
                metrics_dict['overall']['acc'].append(acc)
                metrics_dict['overall']['f1'].append(f1)
            except Exception as e:
                pass
        
        # --- Gather Judge Evaluations ---
        report_csv = dataset_dir / "judge_report_mismatches.csv"
        preds_csv = dataset_dir / "predictions_with_judge.csv"
        
        if report_csv.exists() and preds_csv.exists():
            df_report = pd.read_csv(report_csv)
            df_preds = pd.read_csv(preds_csv)
            if len(df_report) == 0:
                continue
            
            eval_samples = len(df_report)
            agreements = df_report['human_agreement'].sum()
            
            judge_decisions_mismatch = df_report['judge_decision'].astype(bool).tolist()
            human_agreed = df_report['human_agreement'].astype(bool).tolist()
            human_decisions_mismatch = [j if a else not j for j, a in zip(judge_decisions_mismatch, human_agreed)]
            
            try:
                mismatch_kappa = cohen_kappa_score(judge_decisions_mismatch, human_decisions_mismatch)
            except Exception:
                mismatch_kappa = float('nan')
            
            disagreement_ids = df_report[df_report['human_agreement'] == False]['id'].tolist()
            judge_decisions_overall = df_preds['judge_is_match'].astype(bool).tolist()
            
            human_decisions_overall = []
            for _, row in df_preds.iterrows():
                j_dec = bool(row['judge_is_match'])
                if row['id'] in disagreement_ids:
                    human_decisions_overall.append(not j_dec)
                else:
                    human_decisions_overall.append(j_dec)
                    
            try:
                overall_kappa = cohen_kappa_score(judge_decisions_overall, human_decisions_overall)
            except Exception:
                overall_kappa = float('nan')
            
            mk_str = f"{mismatch_kappa:.4f}" if pd.notna(mismatch_kappa) else "NaN"
            ok_str = f"{overall_kappa:.4f}" if pd.notna(overall_kappa) else "NaN"
            
            print(f"{dataset_dir.name[:43]:<45} | {agreements:<10} | {eval_samples:<12} | {mk_str:<15} | {ok_str:<15}")
            
            summary_data.append({
                'Dataset Name': dataset_dir.name,
                'Agreements': agreements,
                'Eval Samples': eval_samples,
                'Mismatch Kappa': mismatch_kappa,
                'Overall Kappa': overall_kappa
            })
            
            total_agreements += agreements
            total_eval_samples += eval_samples
            all_judge_mismatch.extend(judge_decisions_mismatch)
            all_human_mismatch.extend(human_decisions_mismatch)
            all_judge_overall.extend(judge_decisions_overall)
            all_human_overall.extend(human_decisions_overall)
            
    if total_eval_samples > 0:
        total_mismatch_kappa = cohen_kappa_score(all_judge_mismatch, all_human_mismatch)
        total_overall_kappa = cohen_kappa_score(all_judge_overall, all_human_overall)
        print("-" * 115)
        print(f"{'TOTAL / AGGREGATE':<45} | {total_agreements:<10} | {total_eval_samples:<12} | {total_mismatch_kappa:<15.4f} | {total_overall_kappa:.4f}")
        print("=" * 115)
        print("\n* \'Mismatch Kappa\': Cohen Kappa calculated ONLY on the borderline mismatches you evaluated.")
        print("* \'Overall Kappa\': Inferred Cohen Kappa across ALL dataset predictions, assuming you agree with the judge everywhere you didn\'t explicitly disagree.")
        
        summary_data.append({
            'Dataset Name': 'TOTAL / AGGREGATE',
            'Agreements': total_agreements,
            'Eval Samples': total_eval_samples,
            'Mismatch Kappa': total_mismatch_kappa,
            'Overall Kappa': total_overall_kappa
        })
        
        out_csv = model_path / "judge_evaluation_summary.csv"
        pd.DataFrame(summary_data).to_csv(out_csv, index=False)
        print(f"\n✅ Saved judge evaluation report to: {out_csv}")
    else:
        print("No evaluation reports found. Run eval_judge_manual.py to verify at least one dataset.")

    # 2. Output Task Metrics Summary
    task_summary_data = []
    print("\n\n" + "="*80)
    print(f"--- Task-Level Metrics Summary for {model_path.name} ---")
    print(f"{'Task':<15} | {'Acc (Mean ± SD)':<20} | {'F1-Macro (Mean ± SD)':<25}")
    print("-" * 80)
    
    for task in ['disease', 'pest/damage', 'crops/weeds', 'overall']:
        acc_list = metrics_dict[task]['acc']
        f1_list = metrics_dict[task]['f1']
        if len(acc_list) > 0:
            mean_acc = np.mean(acc_list) * 100.0
            # Sample std div (ddof=1) if > 1 sample, else 0.0
            std_acc = np.std(acc_list, ddof=1) * 100.0 if len(acc_list) > 1 else 0.0
            mean_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list, ddof=1) if len(f1_list) > 1 else 0.0
            
            print(f"{task.capitalize():<15} | {mean_acc:.1f}% ± {std_acc:.1f}%       | {mean_f1:.4f} ± {std_f1:.4f}")
            
            task_summary_data.append({
                'model': model_path.name,
                'task': task,
                'averaged_accuracy': mean_acc,
                'accuracy_standard_deviation': std_acc,
                'averaged_f1_score': mean_f1,
                'f1_standard_deviation': std_f1
            })
            
    if len(task_summary_data) > 0:
        out_metrics_csv = model_path / "task_metrics_summary.csv"
        pd.DataFrame(task_summary_data).to_csv(out_metrics_csv, index=False)
        print("="*80)
        print(f"✅ Saved task-level metrics report to: {out_metrics_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summarize LLM Judge Manual Evaluations and Task Metrics')
    parser.add_argument('model_dir', help='Path to the model directory containing dataset folders')
    args = parser.parse_args()
    summarize_evaluations(args.model_dir)
