import os
from datasets import Dataset, DatasetDict, Image
import json
from huggingface_hub import upload_file
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
from torchvision import transforms

default_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
        ])

class SemanticSegTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform_fn=None):
        self.dataset = hf_dataset
        self.transform = transform_fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        if self.transform:
            img, label = self.transform(data["pixel_values"], data["label"])
        else:
            img = self.transform(data["pixel_values"])
            label = self.transform(data["label"])

        return img, label



def create_hf_dataset(root_dir, train=True):
    img_dir = os.path.join(root_dir, "images", "training" if train else "validation")
    ann_dir = os.path.join(root_dir, "annotations", "training" if train else "validation")
    image_paths = sorted([os.path.join(img_dir, file) for _, _, files in os.walk(img_dir) for file in files])
    label_paths = sorted([os.path.join(ann_dir, file) for _, _, files in os.walk(ann_dir) for file in files])

    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset


if __name__ == "__main__":

    # train_dataset = create_hf_dataset(root_dir="/work/dlclarge2/dasb-Camvid/VOC_2007", train=True)
    # validation_dataset = create_hf_dataset(root_dir="/work/dlclarge2/dasb-Camvid/VOC_2007", train=False)

    # dataset = DatasetDict({
    #     "train": train_dataset,
    #     "validation": validation_dataset,
    # }
    # )

    hf = load_dataset("nielsr/ade20k-demo", split="train")
    dataset = SemanticSegTorchDataset(hf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    imgs, label = next(iter(dataloader))
    print(f"Images batch shape: {imgs.size()}")
    print(f"Labels batch shape: {label.size()}")
        


