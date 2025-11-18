import argparse
import yaml
import os

from tasks.classification import is_dataset_avail

def main(args):
    
    output_dir = os.path.join(args.output_dir, args.model_type, args.dataset)
    
    if not is_dataset_avail(args.dataset):
        
        raise ValueError(f"Dataset {args.dataset} is not available in AgML.")
    
    # Format prompt template with plant_type if provided
    if hasattr(args, 'plant_type') and args.plant_type:
        if 'prompt_template' in args.cfg and '{plant_type}' in args.cfg['prompt_template']:
            args.cfg['prompt_template'] = args.cfg['prompt_template'].format(plant_type=args.plant_type)
            print(f"Using prompt with plant type: {args.plant_type}")
    
    if args.model_type == "siglip2":
        
        from models.siglip2 import test
        
        test(args.cfg, model_type="google/siglip2-base-patch16-naflex", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "llava_next":
        
        from models.llava_next import test
        
        test(args.cfg, model_type="llava-hf/llama3-llava-next-8b-hf", dataset=args.dataset, output_dir=output_dir)
    
    elif args.model_type == "qwen_vl":
        
        from models.qwen_vl import test
        
        test(args.cfg, model_type="Qwen/Qwen2.5-VL-7B-Instruct", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "qwen_vl_72b":
        
        from models.qwen_vl import test
        
        test(args.cfg, model_type="Qwen/Qwen2.5-VL-72B-Instruct", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "gemma_3":
        
        from models.gemma_3 import test
        
        test(args.cfg, model_type="google/gemma-3-4b-it", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "deepseek_vl":
        
        from models.deepseekvl_7b import test
        
        test(args.cfg, model_type="deepseek-ai/deepseek-vl-7b-chat", dataset=args.dataset, output_dir=output_dir)

    elif args.model_type == "gpt-5-nano":
        
        from models.api_vlms import test_openai
        
        test_openai(args.cfg, model_type="gpt-5-nano", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "gpt-5":
        
        from models.api_vlms import test_openai

        test_openai(args.cfg, model_type="gpt-5", dataset=args.dataset, output_dir=output_dir)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to pull from AgML.")
    parser.add_argument("--plant-type", type=str, default=None, help="Plant type for open-ended prompts.")
    parser.add_argument("--model-type", type=str, default="yolo", help="Type of model to use.")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to save outputs.")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args.cfg = yaml_cfg.get(args.model_type, {})
    
    main(args)