import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
import pandas as pd
import random
from itertools import product

# Define the grid for hyperparameters
hyperparameter_grid = {
    "batch_size": [4],
    "lr": [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01],
    "momentum": [0.0, 0.8, 0.9, 0.95, 0.99],
    "weight_decay": [0, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    "lr_warmup_epochs": [0, 5, 10],
    "model": ["deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large"]
}

# Create all combinations of hyperparameters
all_configs = list(product(
    hyperparameter_grid["batch_size"],
    hyperparameter_grid["lr"],
    hyperparameter_grid["momentum"],
    hyperparameter_grid["weight_decay"],
    hyperparameter_grid["lr_warmup_epochs"],
    hyperparameter_grid["model"]
))

# Select 3 random samples from the generated grid
random.seed(42)  # For reproducibility
selected_configs = random.sample(all_configs, 3)

# Define a basic training loop
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        targets = targets.squeeze(1).long()
        optimizer.zero_grad()
        print(images.shape)
        outputs = model(images)["out"]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define the transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts PIL Image to PyTorch Tensor and scales values to [0, 1]
])

# Load the VOCSegmentation dataset with the defined transform
train_dataset = VOCSegmentation(
    root="/work/dlclarge2/dasb-Camvid",
    year="2007",
    image_set="train",
    download=False,
    transforms=lambda img, target: (transform(img), transform(target))  # Apply transform to both image and target
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataFrame to store results
results = []

# Training loop for each configuration sample
for config in selected_configs:
    batch_size, learning_rate, momentum, weight_decay, lr_warmup_epochs, model_name = config

    # Model Selection
    if model_name == "deeplabv3_mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=21)
    else:
        model = lraspp_mobilenet_v3_large(pretrained=False, num_classes=21)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Training with warmup epochs
    epochs = 1
    start_time = time.time()
    for epoch in range(epochs):
        if epoch < lr_warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * (epoch + 1) / lr_warmup_epochs
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        
        try:
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        except:
            loss = 0

    training_time = time.time() - start_time
    results.append({
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_warmup_epochs": lr_warmup_epochs,
        "model": model_name,
        "loss": loss,
        "training_time": training_time
    })

# Save results to a DataFrame and export to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("hpo_results.csv", index=False)
print("Results saved to hpo_results.csv")