import argparse
import yaml
import os

from tasks.classification import is_dataset_avail

def main(args):
    
    output_dir = os.path.join(args.output_dir, args.model_type, args.dataset)
    
    if not is_dataset_avail(args.dataset):
        
        raise ValueError(f"Dataset {args.dataset} is not available in AgML.")
    
    # Format prompt template with plant_type and task if provided
    if 'prompt_template' in args.cfg:
        template = args.cfg['prompt_template']
        format_dict = {}
        
        if hasattr(args, 'plant_type') and args.plant_type:
            format_dict['plant_type'] = args.plant_type
            
        if hasattr(args, 'task') and args.task:
            format_dict['task'] = args.task
        
        # Only format if we have placeholders to fill
        if format_dict:
            # Use SafeFormatDict to handle missing placeholders gracefully
            class SafeFormatDict(dict):
                def __missing__(self, key):
                    return '{' + key + '}'
            
            args.cfg['prompt_template'] = template.format_map(SafeFormatDict(**format_dict))
            
            if 'plant_type' in format_dict:
                print(f"Using prompt with plant type: {format_dict['plant_type']}")
            if 'task' in format_dict:
                print(f"Using prompt with task: {format_dict['task']}")
    
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
        
    elif args.model_type == "qwen_vl_3":
        
        from models.qwen_vl import test
        
        test(args.cfg, model_type="Qwen/Qwen3-VL-8B-Instruct", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "gemma_3":
        
        from models.gemma_3 import test
        
        # Check config for a fine-tuned model path, otherwise use default
        model_path = args.cfg.get('model_path', "google/gemma-3-4b-it")
        test(args.cfg, model_type=model_path, dataset=args.dataset, output_dir=output_dir)

    elif args.model_type == "gemm_4":
        
        from models.gemm_4 import test
        
        # Check config for a fine-tuned model path, otherwise use default
        model_path = args.cfg.get('model_path', "google/gemma-4-E2B-it") # Or whichever gemma 4 variant
        test(args.cfg, model_type=model_path, dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "deepseek_vl":
        
        from models.deepseekvl_7b import test
        
        test(args.cfg, model_type="deepseek-ai/deepseek-vl-7b-chat", dataset=args.dataset, output_dir=output_dir)

    elif args.model_type == "gpt-5-nano":
        
        from models.api_vlms import test_openai
        
        test_openai(args.cfg, model_type="gpt-5-nano", dataset=args.dataset, output_dir=output_dir)
        
    elif args.model_type == "gpt-5":
        
        from models.api_vlms import test_openai

        test_openai(args.cfg, model_type="gpt-5", dataset=args.dataset, output_dir=output_dir)

    elif args.model_type.startswith("gemini"):
        
        from models.api_vlms import test_gemini
        
        # Extract model name from model_type (e.g., "gemini_25_flash" -> "gemini-2.5-flash")
        # or use a mapping from config
        model_name_map = {
            "gemini-3-pro-preview": "gemini-3-pro-preview",
            "gemini_25_flash": "gemini-2.5-flash",
        }
        gemini_model = model_name_map.get(args.model_type, "gemini-2.5-flash")
        
        test_gemini(args.cfg, model_type=gemini_model, dataset=args.dataset, output_dir=output_dir)

    elif args.model_type.startswith("claude"):
        
        from models.api_vlms import test_claude
        
        # Map config names to Claude API model names
        model_name_map = {
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5": "claude-haiku-4-5-20251001",
            "claude-opus-4-5": "claude-opus-4-5-20251101",
        }
        claude_model = model_name_map.get(args.model_type, "claude-haiku-4-5-20251001")
        
        test_claude(args.cfg, model_type=claude_model, dataset=args.dataset, output_dir=output_dir)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to pull from AgML.")
    parser.add_argument("--plant-type", type=str, default=None, help="Plant type for open-ended prompts.")
    parser.add_argument("--task", type=str, default=None, help="Task type (e.g., 'disease', 'pest/damage', 'crops/weeds') for prompt templates.")
    parser.add_argument("--model-type", type=str, default="yolo", help="Type of model to use.")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to save outputs.")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args.cfg = yaml_cfg.get(args.model_type, {})
    
    main(args)