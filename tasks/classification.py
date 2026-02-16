import agml
import os
import pandas as pd
import shutil

from pathlib import Path
from ultralytics.data.split import split_classify_dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def parse_class_to_dataset(class_name: str) -> tuple[str, str]:

    if '-' in class_name:
        parts = class_name.split('-', 1)  # split on first dash only
        return parts[0], parts[1]
    else:
        # fallback for classes without prefix
        return "unknown", class_name

def is_dataset_avail(dataset: str):
    
    available_datasets = [
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
        "vine_virus_photo_dataset"
    ]
    
    if dataset in available_datasets:
        return True
    
    else:
        return False

def load_agml_dataset(dataset, split_name="combined"):
    """
    Load AgML dataset(s) and return the path to the split dataset.
    
    Args:
        dataset: Dataset name (str) or list of dataset names
        split_name: Name suffix for combined dataset directory (e.g., "train", "val", "combined")
                   This allows creating separate combined directories for train and val splits.
    
    Returns:
        Path to the dataset directory
    """

    # single dataset
    if isinstance(dataset, str):
        loader = agml.data.AgMLDataLoader(dataset)
        dataset_path = loader.dataset_root
        
        # check if _split folders exist
        if not os.path.exists(f"{dataset_path}_split"):
            split_classify_dataset(dataset_path, train_ratio=0.8)
            
        dataset_path = f"{dataset_path}_split"
        return dataset_path
    
    # multiple datasets
    elif isinstance(dataset, list):
        
        # load each dataset
        split_paths = []
        for ds in dataset:
            loader = agml.data.AgMLDataLoader(ds)
            dataset_path = loader.dataset_root
            
            # check if _split folders exist
            if not os.path.exists(f"{dataset_path}_split"):
                split_classify_dataset(dataset_path, train_ratio=0.8)
                
            split_paths.append(f"{dataset_path}_split")
        
        # combined name and temp directory with split_name suffix
        combined_name = f"all_datasets_{split_name}"
        parent_dir = Path(split_paths[0]).parent
        combined_dir = parent_dir / f"{combined_name}_combined"
        
        # remove existing combined directory
        if combined_dir.exists():
            print(f"Removing existing combined dataset directory at {combined_dir}")
            shutil.rmtree(combined_dir)
        
        # create train and val directories
        train_dir = combined_dir / "train"
        val_dir = combined_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # copy all class folders from each dataset (symlinks don't work with Path.rglob)
        for split_path in split_paths:
            split_path = Path(split_path)
            
            # get dataset name from path
            dataset_name = split_path.stem.replace('_split', '')
            
            train_src = split_path / "train"
            if train_src.exists():
                for class_dir in train_src.iterdir():
                    if class_dir.is_dir() and not class_dir.name.endswith('.cache'):
                        # prefix class name with dataset name
                        target_class_name = f"{dataset_name}-{class_dir.name}"
                        target_path = train_dir / target_class_name
                        
                        shutil.copytree(class_dir, target_path, symlinks=True)
            
            val_src = split_path / "val"
            if val_src.exists():
                for class_dir in val_src.iterdir():
                    if class_dir.is_dir() and not class_dir.name.endswith('.cache'):
                        # prefix class name with dataset name
                        target_class_name = f"{dataset_name}-{class_dir.name}"
                        target_path = val_dir / target_class_name
                        
                        shutil.copytree(class_dir, target_path, symlinks=True)
        
        return str(combined_dir)
    
    else:
        raise TypeError(f"dataset must be str or list[str], got {type(dataset)}")

def candidate_labels(dataset_path: str):
    
    # get class names from folders in path
    class_names = sorted(os.listdir(os.path.join(dataset_path, "train")))
    
    return class_names

def agml_to_df(root: str):
    
    # ensure str is Path object
    if not isinstance(root, Path):
        root = Path(root)
    
    # index dataset path
    rows = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        
        label = class_dir.name
        for p in sorted(class_dir.rglob("*")):
            
            if p.is_file() and p.suffix.lower() in IMG_EXTS:        
                rows.append({"id": p.stem, "image_path": str(p), "label": label})
                
    if not rows:
        raise ValueError(f"No images found under {root}")
    
    return pd.DataFrame(rows)