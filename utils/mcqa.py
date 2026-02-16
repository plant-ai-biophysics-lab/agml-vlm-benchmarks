"""
MCQA (Multiple Choice Question Answering) utilities.

Provides functions to generate MCQA choices for classification tasks.
Can be used by any model's test function by checking for 'mcqa_options' in config.
"""

import random
from typing import List, Tuple

# Set random seed for reproducibility
RANDOM_SEED = 42


def get_mcqa_choices(
    true_label: str,
    all_classes: List[str],
    options_within_dataset: bool = True,
    mcqa_num_choices: int = 4,
    all_dataset_classes: dict = None,
    current_dataset: str = None,
    answer_included_ratio: float = 0.7,
    sample_index: int = 0,
    print_sample: bool = False,
) -> Tuple[List[str], str, bool, int]:
    """
    Generate MCQA choices for a single sample.

    Args:
        true_label: The correct label for the sample
        all_classes: All available classes from the current dataset
        options_within_dataset: If True (mcqa_1), sample from same dataset only
        mcqa_num_choices: Total number of choices (including special option)
        all_dataset_classes: Dict mapping dataset names to their classes (for mcqa_2)
        current_dataset: Name of current dataset (for mcqa_2)
        answer_included_ratio: Ratio of samples where answer is included (default: 0.7)
        sample_index: Index of current sample (for reproducible seeding)
        print_sample: If True, print debug info for this sample

    Returns:
        Tuple of (choices, correct_answer, answer_was_included, correct_answer_index)
    """
    # Use sample index for reproducible randomization per sample
    random.seed(RANDOM_SEED + sample_index)

    # Use "None of the above" for both modes since answer is not always present
    special_option = "None of the above"

    # Determine if answer should be included (70/30 split)
    include_answer = random.random() < answer_included_ratio

    if print_sample:
        import sys

        print(f"\n{'=' * 80}", flush=True)
        print(f"MCQA Sample - Image #{sample_index + 1}", flush=True)
        print(f"{'=' * 80}", flush=True)
        print(f"True Label: {true_label}", flush=True)
        print(
            f"Mode: {'Within Dataset' if options_within_dataset else 'Cross Dataset'}",
            flush=True,
        )
        print(f"Answer Included: {include_answer}", flush=True)

    if include_answer:
        # Include correct answer in choices
        # Generate (mcqa_num_choices - 2) distractors (reserve 1 for correct, 1 for special)
        num_distractors = mcqa_num_choices - 2

        # if num_distractors > len(all_classes), make num_distractors == len(all_classes)
        if num_distractors > len(all_classes):
            num_distractors = len(all_classes)

        if options_within_dataset:
            # mcqa intra class: Sample distractors from same dataset
            available_distractors = [c for c in all_classes if c != true_label]
        else:
            # mcqa inter class: Sample distractors from any dataset
            available_distractors = []
            if all_dataset_classes and current_dataset:
                for dataset_name, classes in all_dataset_classes.items():
                    for cls in classes:
                        # Can include classes from same or different datasets
                        if cls != true_label:
                            available_distractors.append(cls)
            else:
                # Fallback to same dataset if cross-dataset classes not available
                available_distractors = [c for c in all_classes if c != true_label]

        # Sample distractors
        if len(available_distractors) >= num_distractors:
            distractors = random.sample(available_distractors, num_distractors)
        else:
            distractors = available_distractors

        # Combine: correct answer + distractors (will add special option at end after shuffling)
        choices_to_shuffle = [true_label] + distractors
        correct_answer = true_label

    else:
        # Do not include correct answer - only distractors (will add special option at end after shuffling)
        num_distractors = mcqa_num_choices - 1  # Reserve 1 for special option

        # if num_distractors > len(all_classes), make num_distractors == len(all_classes)
        if num_distractors > len(all_classes):
            num_distractors = len(all_classes)

        if options_within_dataset:
            # mcqa_1: Sample from same dataset (excluding correct answer)
            available_distractors = [c for c in all_classes if c != true_label]
        else:
            # mcqa_2: Sample from any dataset (excluding correct answer)
            available_distractors = []
            if all_dataset_classes and current_dataset:
                for dataset_name, classes in all_dataset_classes.items():
                    for cls in classes:
                        if cls != true_label:
                            available_distractors.append(cls)
            else:
                available_distractors = [c for c in all_classes if c != true_label]

        # Sample distractors
        if len(available_distractors) >= num_distractors:
            distractors = random.sample(available_distractors, num_distractors)
        else:
            distractors = available_distractors

        choices_to_shuffle = distractors
        correct_answer = special_option

    # Shuffle the choices (excluding special option)
    random.shuffle(choices_to_shuffle)

    # Add special option at the end
    choices = choices_to_shuffle + [special_option]
    
    # Find the index of the correct answer (1-indexed for display)
    correct_answer_index = choices.index(correct_answer)

    if print_sample:
        print(f"\nGenerated Choices:", flush=True)
        for i, choice in enumerate(choices, 1):
            marker = " ✓" if choice == correct_answer else ""
            print(f"  {i}. {choice}{marker}", flush=True)
        print(f"\nCorrect Answer: {correct_answer} (Option {correct_answer_index + 1})", flush=True)
        print(f"{'=' * 80}\n", flush=True)

    return choices, correct_answer, include_answer, correct_answer_index


def load_all_dataset_classes() -> dict:
    """
    Load class mappings for all available datasets.
    Used for mcqa_2 mode to sample cross-dataset distractors.

    Returns:
        Dictionary mapping dataset names to their class lists
    """
    from tasks.classification import load_agml_dataset, candidate_labels

    datasets = [
        "arabica_coffee_leaf_disease_classification",
        "banana_leaf_disease_classification",
        "bean_disease_uganda",
        "betel_leaf_disease_classification",
        "blackgram_plant_leaf_disease_classification",
        "chilli_leaf_classification",
        "coconut_tree_disease_classification",
        "corn_maize_leaf_disease",
        "crop_weeds_greece",
        "cucumber_disease_classification",
        "guava_disease_pakistan",
        "java_plum_leaf_disease_classification",
        "leaf_counting_denmark",
        "onion_leaf_classification",
        "orange_leaf_disease_classification",
        "paddy_disease_classification",
        "papaya_leaf_disease_classification",
        "plant_seedlings_aarhus",
        "plant_village_classification",
        "rangeland_weeds_australia",
        "rice_leaf_disease_classification",
        "soybean_insect_classification",
        "soybean_weed_uav_brazil",
        "sugarcane_damage_usa",
        "sunflower_disease_classification",
        "tea_leaf_disease_classification",
        "tomato_leaf_disease",
        "vine_virus_photo_dataset",
    ]

    class_mapping = {}
    for dataset_name in datasets:
        try:
            dataset_path = load_agml_dataset(dataset_name)
            classes = candidate_labels(dataset_path)
            class_mapping[dataset_name] = classes
        except Exception as e:
            print(f"Warning: Could not load classes for {dataset_name}: {e}")
            continue

    return class_mapping
