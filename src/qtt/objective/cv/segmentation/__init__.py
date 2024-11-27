import os
from pathlib import Path
from torchvision.datasets import VOCSegmentation 
from .finetune_wrapper import finetune_script

def extract_segmentation_task_info_metafeat(
    dataset_class, 
    root: str | Path, 
    year: str = None, 
    transform=None,
    download=False
):
    root = Path(root)
    assert root.exists(), f"dataset-path: {root} does not exist."


    num_samples = 5 # + len(val_dataset)
    num_classes = 21
    num_features = 128  # Fixed for now
    num_channels = 3    # Assuming RGB
    dataset_name = "voc"

    metafeat = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "num_features": num_features,
        "num_channels": num_channels,
    }

    task_info = {
        "data-path": str(root),
        "dataset": dataset_name,
        "num-classes": num_classes,
        "num-samples": num_samples,
    }

    return task_info, metafeat


