import json
import os
import random

from typing import Optional
from tasks.classification import load_agml_dataset

def get_context(dataset: str, num_examples_per_class: int = 1, seed: int = 42) -> dict[str, list[str]]:
    """Get context examples from the training set.

    Args:
        num_examples_per_class: Number of images to sample per class (class-balanced).
            Set to 0 to build a random pool instead: 100 images are sampled uniformly
            from the entire training set (class labels are still tracked). Use this
            pool with random_pool=True in create_classification_message.
    """
    random.seed(seed)
    # Accept direct directory paths (e.g. combined fold dirs from load_fold_split)
    # as well as AgML dataset names.
    if os.path.isdir(dataset):
        dataset_path = dataset
    else:
        dataset_path = load_agml_dataset(dataset)
    
    # we use the training set to get context examples
    dataset_path_train = os.path.join(dataset_path, "train")
    
    if num_examples_per_class == 0:
        # random pool: collect all images across all classes, sample 100
        all_images = []  # list of (class_name, image_path)
        for class_name in os.listdir(dataset_path_train):
            class_folder = os.path.join(dataset_path_train, class_name)
            if os.path.isdir(class_folder):
                for img in os.listdir(class_folder):
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        all_images.append((class_name, os.path.join(class_folder, img)))
        pool_size = 100
        selected = random.sample(all_images, min(pool_size, len(all_images)))
        if len(selected) < pool_size:
            print(
                f"WARNING: random pool requested {pool_size} images but only "
                f"{len(all_images)} available. Using all {len(all_images)}."
            )
        # organise back into dict[class_name, list[str]] preserving real labels
        pool: dict[str, list[str]] = {}
        for class_name, image_path in selected:
            pool.setdefault(class_name, []).append(image_path)
        return pool
    
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
    include_correct_class: bool = True,
    random_pool: bool = False,
    output_path: Optional[str] = None,
    random_seed: int = 42
) -> tuple[dict, dict]:
    """Build a classification message with in-context examples.

    Args:
        include_correct_class: When True (default) and max_num_class_context is set,
            the correct class is always guaranteed to appear in the sampled context.
            Ignored when random_pool=True.
        random_pool: When True, context_examples is treated as a flat pool (built with
            num_examples_per_class=0 in get_context). max_num_class_context images are
            sampled randomly from the entire pool regardless of class. include_correct_class
            is ignored in this mode.

    Returns:
        message: The conversation message dict.
        context_meta: Metadata about the context used:
            - num_context_classes: number of classes included as context
            - num_context_examples: total number of context images included
    """
    content = [
        {
            "type": "text",
            "text": "Here are image and label examples of disease, pest, damage, or other stresses:",
        }
    ]
    
    # filter / sample context
    use_all = max_num_class_context == "max" or max_num_class_context is None
    if random_pool and not use_all:
        # flatten pool into (class_name, image_path) pairs and sample randomly
        random.seed(random_seed)
        all_pairs = [
            (cls, img)
            for cls, imgs in context_examples.items()
            for img in imgs
        ]
        n = min(max_num_class_context, len(all_pairs))
        selected_pairs = random.sample(all_pairs, n)
        # rebuild as dict to reuse the same rendering loop below
        sampled_context: dict[str, list[str]] = {}
        for cls, img in selected_pairs:
            sampled_context.setdefault(cls, []).append(img)
        context_examples = sampled_context
    elif not random_pool and not use_all:
        # class-balanced sampling
        random.seed(random_seed)
        classes = list(context_examples.keys())
        if include_correct_class and correct_class and correct_class in classes:
            classes.remove(correct_class)
            selected_classes = random.sample(classes, min(max_num_class_context - 1, len(classes)))
            selected_classes.append(correct_class)
        else:
            selected_classes = random.sample(classes, min(max_num_class_context, len(classes)))
        context_examples = {cls: context_examples[cls] for cls in selected_classes}
    # if use_all: context_examples is used as-is

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
