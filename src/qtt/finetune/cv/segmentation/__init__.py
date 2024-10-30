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

    # Initialize the dataset
    if dataset_class == VOCSegmentation:
        assert year is not None, "Year must be specified for VOCSegmentation."
        dataset_name = "voc"
        train_dataset = dataset_class(root=root, year=year, image_set="train", download=download)
        # val_dataset = dataset_class(root=root, year=year, image_set="valid", download=True)
    else:
        ##TODO: need to test for other datasets
        dataset = dataset_class(root=root, split=image_set, transform=transform, download=download)

    num_samples = len(train_dataset) # + len(val_dataset)
    num_classes = len(train_dataset.class_names) if hasattr(train_dataset, 'class_names') else 0
    num_features = 128  # Fixed for now
    num_channels = 3    # Assuming RGB

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

# # # Example usage
# task_info, metafeat = extract_segmentation_task_info_metafeat(
#     dataset_class=VOCSegmentation, 
#     root='/work/dlclarge2/dasb-Camvid', 
#     year='2007', 
# )
# print(task_info)
# print(metafeat)

