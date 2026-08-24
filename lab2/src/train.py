import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from utils import *
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from evaluate import validate_one_epoch
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = get_data_dir()  
SPLIT_DIR = "./kaggle/split/"
SAVE_DIR = "./saved_models/"

MODEL_NAME = "unet" # (input: "unet" or "resnet")
MODEL = UNet if MODEL_NAME == "unet" else ResNet34_UNet
TARGET_SIZE = 256
BATCH_SIZE =32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

DEVICE = get_device()
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = torch.cuda.is_available()
# ==========================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = forward_pass(model, images)
        loss = criterion(outputs, masks)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        with torch.inference_mode():
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            batch_dice = get_dice_score(preds, masks)
            epoch_dice += batch_dice
        
    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

def main():
    set_seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Current Device: {DEVICE}")

    train_transform = get_transforms("train", TARGET_SIZE)
    valid_transform = get_transforms("valid", TARGET_SIZE)

    train_dataset = OxfordPetDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR, mode="train", 
                                     model_name=MODEL_NAME, transform=train_transform)
    valid_dataset = OxfordPetDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR, mode="valid", 
                                     model_name=MODEL_NAME, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    model = MODEL(in_channels=3, out_channels=1).to(DEVICE)
    loss_func = BCEDiceLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                                    epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))

    best_dice = 0.0
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_dice = train_one_epoch(model, train_loader, loss_func, optimizer, scheduler, DEVICE)
        valid_loss, valid_dice = validate_one_epoch(model, valid_loader, loss_func, DEVICE)
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Dice: {train_dice:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"Valid Dice: {valid_dice:.4f}")
        
        if valid_dice > best_dice:
            best_dice = valid_dice
            model_name = model.__class__.__name__ 
            save_path = os.path.join(SAVE_DIR, f"best_{model_name}.pth")
            
            torch.save(model.state_dict(), save_path)
            print(f"<Found a higher dice score ({best_dice:.4f})! Model already saved to {save_path}>")
        

if __name__ == "__main__":
    main()