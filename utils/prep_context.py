import json
import os
import random

from typing import Optional
from tasks.classification import load_agml_dataset

def get_context(dataset: str, num_examples_per_class: int = 1, seed: int = 42) -> dict[str, list[str]]:
    
    random.seed(seed)
    dataset_path = load_agml_dataset(dataset)
    
    # we use the training set to get context examples
    dataset_path_train = os.path.join(dataset_path, "train")
    
    # go through each class folder and get example image paths
    context_examples = {}
    for class_name in os.listdir(dataset_path_train):
        class_folder = os.path.join(dataset_path_train, class_name)
        if os.path.isdir(class_folder):
            
            # randomly select images from the class folder
            image_paths = [
                os.path.join(class_folder, img)
                for img in os.listdir(class_folder)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            
            # generate random indexes for selecting examples
            selected_images = random.sample(image_paths, min(num_examples_per_class, len(image_paths)))
            context_examples[class_name] = selected_images

    return context_examples

def create_classification_message(
    task: str,
    template: str,
    query_image_path: str,
    context_examples: dict[str, list[str]],
    max_num_class_context: Optional[int] = None,
    correct_class: Optional[str] = None,
    output_path: Optional[str] = None,
    random_seed: int = 42
) -> tuple[dict, dict]:
    """Build a classification message with in-context examples.

    Returns:
        message: The conversation message dict.
        context_meta: Metadata about the context used:
            - num_context_classes: number of classes included as context
            - num_context_examples: total number of context images included
    """
    content = [
        {
            "type": "text",
            "text": "Here are examples of disease, pest, damage, or other stress from the same plant:",
        }
    ]
    
    # filter context if max_num_class_context is specified
    # remember max_num_class_context is the max number of classes, not examples
    # but keep correct class examples if correct_class is specified
    if max_num_class_context is not None:
        random.seed(random_seed)
        classes = list(context_examples.keys())
        
        if correct_class and correct_class in classes:
            classes.remove(correct_class)
            selected_classes = random.sample(classes, min(max_num_class_context - 1, len(classes)))
            selected_classes.append(correct_class)  # ensure correct class is included
        else:
            selected_classes = random.sample(classes, min(max_num_class_context, len(classes)))
        context_examples = {cls: context_examples[cls] for cls in selected_classes}

    num_context_examples = 0
    for class_name, image_paths in context_examples.items():
        content.append({"type": "text", "text": class_name})

        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
            num_context_examples += 1

    content.append({"type": "text", "text": template})
    content.append({"type": "image", "image": query_image_path})

    message = {"role": "user", "content": content}
    context_meta = {
        "num_context_classes": len(context_examples),
        "num_context_examples": num_context_examples,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(message, f, indent=2)

    return message, context_meta


def add_context_examples(
    json_path: str,
    new_examples: dict[str, list[str]],
    insert_before_template: bool = True,
) -> dict:
    with open(json_path, "r") as f:
        message = json.load(f)

    content = message["content"]

    if insert_before_template:
        insert_index = len(content) - 2
    else:
        insert_index = 1

    new_entries = []
    for class_name, image_paths in new_examples.items():
        new_entries.append({"type": "text", "text": class_name})
        for image_path in image_paths:
            new_entries.append({"type": "image", "image": image_path})

    message["content"] = content[:insert_index] + new_entries + content[insert_index:]

    with open(json_path, "w") as f:
        json.dump(message, f, indent=2)

    return message


def update_query_image(json_path: str, new_query_image: str) -> dict:
    with open(json_path, "r") as f:
        message = json.load(f)

    message["content"][-1]["image"] = new_query_image

    with open(json_path, "w") as f:
        json.dump(message, f, indent=2)

    return message
