from ultralytics import YOLO
from tasks.classification import load_agml_dataset
        
def train(args: dict, model_type: str, dataset: str, output_dir: str):
    
    # get dataset path
    dataset_path = load_agml_dataset(dataset)
    
    model = YOLO(model_type)
    
    model.train(
        data = dataset_path,
        project = output_dir,
        name = "yolo11_train",
        **args
    )
    
    model.val(
        data = dataset_path,
        project = output_dir,
        name = "yolo11_val",
        **args
    )