import argparse
import yaml
import os

from tasks.classification import is_dataset_avail

def main(args):
    
    # Format prompt template with plant_type if provided
    if hasattr(args, 'plant_type') and args.plant_type:
        if 'prompt_template' in args.cfg and '{plant_type}' in args.cfg['prompt_template']:
            args.cfg['prompt_template'] = args.cfg['prompt_template'].format(plant_type=args.plant_type)
            print(f"Using prompt with plant type: {args.plant_type}")
    
    # handle different dataset specifications
    if args.splits_file:
        
        # load train/val split from YAML file
        with open(args.splits_file, 'r') as f:
            splits = yaml.safe_load(f)
        
        fold = args.fold or "fold_1"
        if fold not in splits:
            raise ValueError(f"Fold '{fold}' not found in {args.splits_file}")
        
        train_datasets = splits[fold]["train"]
        val_datasets = splits[fold]["val"]
        
        # validate all datasets
        for dataset in train_datasets + val_datasets:
            if not is_dataset_avail(dataset):
                raise ValueError(f"Dataset {dataset} is not available in AgML.")
        
        args.dataset = {"train": train_datasets, "val": val_datasets}
        output_name = f"{fold}_train{len(train_datasets)}_val{len(val_datasets)}"
        
        print(f"Using fold: {fold}")
        print(f"Training datasets: {len(train_datasets)}")
        print(f"Testing datasets: {len(val_datasets)}")
        
    else:
        # single or multiple datasets for both train/val
        datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]
        
        for dataset in datasets:
            if not is_dataset_avail(dataset):
                raise ValueError(f"Dataset {dataset} is not available in AgML.")
        
        # create output directory name
        if len(datasets) == 1:
            output_name = datasets[0]
            args.dataset = datasets[0]
        else:
            output_name = "all_datasets"
            args.dataset = datasets
    
    output_dir = os.path.join(args.output_dir, "fine_tune_classification", args.model_type, output_name)
    
    if args.model_type == "yolo":
        
        from models.yolo11 import train

        train(args.cfg, model_type="yolo11x-cls", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "siglip2":
        
        from models.siglip2 import train

        train(args.cfg, model_type="google/siglip2-base-patch16-224", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "llava_next":
        
        from models.llava_next import train

        train(args.cfg, model_type="llava-hf/llama3-llava-next-8b-hf", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "qwen_vl":
        
        from models.qwen_vl import train

        train(args.cfg, model_type="Qwen/Qwen2.5-VL-7B-Instruct", dataset=args.dataset, output_dir=output_dir)  
        
    elif args.model_type == "gemma_3":
        
        from models.gemma_3 import train

        train(args.cfg, model_type="google/gemma-3-4b-it", dataset=args.dataset, output_dir=output_dir)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs='+', help="Name of dataset(s) to pull from AgML. Can specify multiple datasets.")
    parser.add_argument("--plant-type", type=str, default=None, help="Plant type for open-ended prompts.")
    parser.add_argument("--splits-file", type=str, help="Path to YAML file with train/val splits (e.g., splits.yaml)")
    parser.add_argument("--fold", type=str, help="Fold name from splits file (e.g., fold_1)")
    parser.add_argument("--model-type", type=str, default="yolo", help="Type of model to use.")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to save outputs.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset and not args.splits_file:
        parser.error("Either --dataset or --splits-file must be specified")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args.cfg = yaml_cfg.get(args.model_type, {})
    
    main(args)