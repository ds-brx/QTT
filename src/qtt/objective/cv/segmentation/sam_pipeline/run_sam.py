"""
Reference: 
Rogge, N. (2020). Transformers Tutorials" (Version 1.0) 
[Computer software]. https://doi.org/10.5281/zenodo.1234

"""
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
import time
from datasets import load_dataset
import argparse
from tqdm import tqdm
from torch.optim import Adam
import monai
from torch_pipeline import presets 
from PIL import Image
import torchvision.transforms as transforms
from statistics import mean
def get_args(args):
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default="", type=str, help="dataset path")
  parser.add_argument("--dataset_name", default="coco", type=str, help="dataset name")
  parser.add_argument(
      "-b", "--per_device_train_batch_size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
  )
  parser.add_argument("--num_train_epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

  parser.add_argument(
      "-j", "--dataloader_num_workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
  )
  parser.add_argument("--learning_rate", default=0.01, type=float, help="initial learning rate")
  parser.add_argument(
      "--wd",
      "--weight_decay",
      default=1e-4,
      type=float,
      metavar="W",
      help="weight decay (default: 1e-4)",
      dest="weight_decay",
  )
  return parser


def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  ground_truth_map = np.array(ground_truth_map)
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
        ])
mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
        ])

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = img_transform(item["pixel_values"])
    ground_truth_mask = mask_transform(item["label"]).squeeze(0)
  
    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

    
def main(args):
    train_dataset = load_dataset(args.dataset_name, split="train")
    valid_dataset = load_dataset(args.dataset_name, split="train")
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=train_dataset, processor=processor)
    valid_dataset = SAMDataset(dataset=valid_dataset, processor=processor)

    train_dataloader = DataLoader(
      train_dataset, 
      batch_size=args.per_device_train_batch_size, 
      shuffle=True,
      num_workers = args.dataloader_num_workers,
      drop_last = True
      )
    valid_dataloader = DataLoader(
      valid_dataset, 
      batch_size=args.per_device_train_batch_size, 
      shuffle=False,
      num_workers = args.dataloader_num_workers,
      drop_last = True
      )

    from transformers import SamModel 

    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
      if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
          param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(
      model.mask_decoder.parameters(), 
      lr=args.learning_rate, 
      weight_decay=args.weight_decay
      )

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    start_time = time.time()
    model.train()
    print("MODEL TRAINING...")
    for epoch in range(args.num_train_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
        # forward pass
          outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)

          # compute loss
          predicted_masks = outputs.pred_masks.squeeze(1)
          ground_truth_masks = batch["ground_truth_mask"].float().to(device)
          loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

          # backward pass (compute gradients of parameters w.r.t. loss)
          optimizer.zero_grad()
          loss.backward()

          # optimize
          optimizer.step()
          epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    
    cost = time.time() - start_time
    val_loss = 0
    print("MODEL VALIDATING...")
    for batch in tqdm(valid_dataloader):
        model.eval()
        with torch.no_grad():
          outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)
          predicted_masks = outputs.pred_masks.squeeze(1)
          ground_truth_masks = batch["ground_truth_mask"].float().to(device)
          loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
          val_loss += loss

    score = val_loss/len(valid_dataloader)

    print("FINISHED.")
    return score, time

if __name__ == "__main__":
    main(args)




        

