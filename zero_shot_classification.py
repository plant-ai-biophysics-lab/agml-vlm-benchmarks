import argparse
import yaml
import os

from tasks.classification import is_dataset_avail

def main(args):
    
    output_dir = os.path.join(args.output_dir, "zero_shot_classification", args.model_type, args.dataset)
    
    if not is_dataset_avail(args.dataset):
        
        raise ValueError(f"Dataset {args.dataset} is not available in AgML.")
    
    if args.model_type == "siglip2":
        
        from models.siglip2 import test
        
        test(args.cfg, model_type="google/siglip2-base-patch16-naflex", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "llava_next":
        
        from models.llava_next import test
        
        test(args.cfg, model_type="llava-hf/llama3-llava-next-8b-hf", dataset=args.dataset, output_dir=output_dir)
    
    elif args.model_type == "qwen_vl":
        
        from models.qwen_vl import test
        
        test(args.cfg, model_type="Qwen/Qwen2.5-VL-7B-Instruct", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "gemma_3":
        
        from models.gemma_3 import test
        
        test(args.cfg, model_type="google/gemma-3-4b-it", dataset=args.dataset, output_dir=output_dir)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to pull from AgML.")
    parser.add_argument("--model-type", type=str, default="yolo", help="Type of model to use.")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to save outputs.")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args.cfg = yaml_cfg.get(args.model_type, {})
    
    main(args)