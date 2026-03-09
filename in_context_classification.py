import argparse
import yaml
import os

from tasks.classification import is_dataset_avail
from utils.prep_context import get_context


def _dispatch_model(args, dataset, output_dir, context, max_num_context, include_correct_class, random_pool):
    """Dispatch model inference for a single dataset."""

    if args.model_type == "siglip2":
        from models.siglip2 import test
        test(args.cfg, model_type="google/siglip2-base-patch16-naflex", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "llava_next":
        from models.llava_next import test
        test(args.cfg, model_type="llava-hf/llama3-llava-next-8b-hf", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "qwen_vl":
        from models.qwen_vl import test
        test(
            args.cfg, model_type="Qwen/Qwen2.5-VL-7B-Instruct", dataset=dataset, output_dir=output_dir,
            context=context, max_num_class_context=max_num_context,
            include_correct_class=include_correct_class, random_pool=random_pool,
        )

    elif args.model_type == "qwen_vl_72b":
        from models.qwen_vl import test
        test(args.cfg, model_type="Qwen/Qwen2.5-VL-72B-Instruct", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "qwen_vl_3":
        from models.qwen_vl import test
        test(
            args.cfg, model_type="Qwen/Qwen3-VL-8B-Instruct", dataset=dataset, output_dir=output_dir,
            context=context, max_num_class_context=max_num_context,
            include_correct_class=include_correct_class, random_pool=random_pool,
        )

    elif args.model_type == "gemma_3":
        from models.gemma_3 import test
        test(args.cfg, model_type="google/gemma-3-4b-it", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "deepseek_vl":
        from models.deepseekvl_7b import test
        test(args.cfg, model_type="deepseek-ai/deepseek-vl-7b-chat", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "gpt-5-nano":
        from models.api_vlms import test_openai
        test_openai(args.cfg, model_type="gpt-5-nano", dataset=dataset, output_dir=output_dir)

    elif args.model_type == "gpt-5":
        from models.api_vlms import test_openai
        test_openai(args.cfg, model_type="gpt-5", dataset=dataset, output_dir=output_dir)

    elif args.model_type.startswith("gemini"):
        from models.api_vlms import test_gemini
        model_name_map = {
            "gemini-3-pro-preview": "gemini-3-pro-preview",
            "gemini_25_flash": "gemini-2.5-flash",
        }
        gemini_model = model_name_map.get(args.model_type, "gemini-2.5-flash")
        test_gemini(args.cfg, model_type=gemini_model, dataset=dataset, output_dir=output_dir)

    elif args.model_type.startswith("claude"):
        from models.api_vlms import test_claude
        model_name_map = {
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5": "claude-haiku-4-5-20251001",
            "claude-opus-4-5": "claude-opus-4-5-20251101",
        }
        claude_model = model_name_map.get(args.model_type, "claude-haiku-4-5-20251001")
        test_claude(args.cfg, model_type=claude_model, dataset=dataset, output_dir=output_dir)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def main(args):
    # Validate required args
    if not args.fold and not args.dataset:
        raise ValueError("Either --dataset or --fold must be specified.")
    if not args.fold and not is_dataset_avail(args.dataset):
        raise ValueError(f"Dataset {args.dataset} is not available in AgML.")

    # Format prompt template with plant_type and task if provided
    if "prompt_template" in args.cfg:
        template = args.cfg["prompt_template"]
        format_dict = {}

        if hasattr(args, "plant_type") and args.plant_type:
            format_dict["plant_type"] = args.plant_type

        if hasattr(args, "task") and args.task:
            format_dict["task"] = args.task

        # Only format if we have placeholders to fill
        if format_dict:
            # Use SafeFormatDict to handle missing placeholders gracefully
            class SafeFormatDict(dict):
                def __missing__(self, key):
                    return "{" + key + "}"

            args.cfg["prompt_template"] = template.format_map(
                SafeFormatDict(**format_dict)
            )

            if "plant_type" in format_dict:
                print(f"Using prompt with plant type: {format_dict['plant_type']}")
            if "task" in format_dict:
                print(f"Using prompt with task: {format_dict['task']}")

    # In-context settings
    if "max_num_context" in args.cfg.get("context_options", {}):
        context_options = args.cfg["context_options"]
        max_num_context = context_options.get("max_num_context", None)
        max_num_example = context_options.get("max_num_example", 1)
        include_correct_class = context_options.get("include_correct_class", True)
        random_pool = (max_num_example == 0)  # 0 = random pool mode
        if max_num_context:
            print(f"Using max_num_context: {max_num_context}")
        if random_pool:
            print("Using random pool context (class-agnostic sampling, labels still shown)")
        else:
            print(f"Include correct class in context: {include_correct_class}")
    else:
        max_num_context = None
        max_num_example = 1
        include_correct_class = True
        random_pool = False

    if args.fold:
        # L1 cross-dataset evaluation: context from fold train split, test on each val dataset
        from tasks.classification import load_fold_split

        train_path, train_datasets = load_fold_split(args.fold, "train", args.splits_path)
        context = get_context(train_path, num_examples_per_class=max_num_example)
        include_correct_class = False  # context labels differ from test dataset labels
        print(
            f"Fold '{args.fold}': context from {len(train_datasets)} train datasets, "
            f"include_correct_class forced to False"
        )

        _, test_datasets = load_fold_split(args.fold, "val", args.splits_path)
        print(f"Fold '{args.fold}': testing on {len(test_datasets)} val datasets: {test_datasets}")

        for dataset in test_datasets:
            fold_output_dir = os.path.join(args.output_dir, args.model_type, args.fold, dataset)
            print(f"\n--- [{args.fold}] Running on val dataset: {dataset} ---")
            _dispatch_model(args, dataset, fold_output_dir, context, max_num_context, include_correct_class, random_pool)

    else:
        output_dir = os.path.join(args.output_dir, args.model_type, args.dataset)
        context = get_context(args.dataset, num_examples_per_class=max_num_example)
        _dispatch_model(args, args.dataset, output_dir, context, max_num_context, include_correct_class, random_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Name of dataset to pull from AgML."
    )
    parser.add_argument(
        "--plant-type",
        type=str,
        default=None,
        help="Plant type for open-ended prompts.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type (e.g., 'disease', 'pest/damage', 'crops/weeds') for prompt templates.",
    )
    parser.add_argument(
        "--model-type", type=str, default="yolo", help="Type of model to use."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/", help="Directory to save outputs."
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help=(
            "Fold name from splits.yaml for L1 cross-dataset evaluation "
            "(e.g. disease_fold_1, pest_damage_fold_2). "
            "Context is built from the fold's train split; inference runs over each val dataset. "
            "Mutually exclusive with --dataset."
        ),
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="splits.yaml",
        help="Path to the splits YAML file (default: splits.yaml).",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args.cfg = yaml_cfg.get(args.model_type, {})

    main(args)
