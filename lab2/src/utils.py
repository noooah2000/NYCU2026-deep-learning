import torch
import torch.nn as nn
import os
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def get_transforms(mode="train", target_size=256):
    if mode == "train":
        return A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_dice_score(pred_mask, gt_mask, smooth=1e-6):
    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)

    intersection = torch.sum(pred_mask_flat * gt_mask_flat)
    total = torch.sum(pred_mask_flat) + torch.sum(gt_mask_flat)

    dice = (2 * intersection + smooth) / (total + smooth)
    return dice.item()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, gt_mask):
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        gt_mask_flat = gt_mask.view(-1)
        
        intersection = torch.sum(probs_flat * gt_mask_flat)
        total = torch.sum(probs_flat) + torch.sum(gt_mask_flat)
        
        dice_score = (2 * intersection + self.smooth) / (total + self.smooth)
        
        return 1 - dice_score
    
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        return 0.7 * self.bce(logits, targets) + 0.3 * self.dice(logits, targets)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed) 

def get_data_dir():
    if os.path.exists('/content'):
        return '/content/dataset'
    elif os.path.exists('/Users/noahhuang/Desktop/DL-NYCU-2026/DL_LAB/lab2_data/dataset/oxford-iiit-pet'):
        return '/Users/noahhuang/Desktop/DL-NYCU-2026/DL_LAB/lab2_data/dataset/oxford-iiit-pet'
    else:
        return './dataset/oxford-iiit-pet/'
    
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def forward_pass(model, images, target_size=256, padding=94):
    if model.__class__.__name__ == "UNet":
        images_pad = F.pad(images, (padding, padding, padding, padding), mode='reflect') 
        outputs_pad = model(images_pad) 
        outputs = TF.center_crop(outputs_pad, [target_size, target_size]) 
        return outputs
        
    else:
        return model(images)