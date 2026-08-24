import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from utils import get_transforms, get_device, get_data_dir, set_seed, forward_pass
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

# ==========================================
# CONFIG 
# ==========================================
MODEL_NAME = "unet" # (input: "unet" or "resnet")
MODEL = UNet if MODEL_NAME == "unet" else ResNet34_UNet

DATA_DIR = get_device() 
DATA_DIR = get_data_dir()  
SPLIT_DIR = "./kaggle/split/"
SUBMISSION_DIR = f"./kaggle/submission/{MODEL_NAME}/"
SAVE_DIR = "./saved_models/"
WEIGHTS_PATH = os.path.join(SAVE_DIR, f"best_{MODEL.__name__}.pth")

TARGET_SIZE = 256
BATCH_SIZE = 16

DEVICE = get_device()
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = torch.cuda.is_available()
# ==========================================

def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

def main():
    set_seed(42)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    print(f"Current Device: {DEVICE}")
    print(f"Loading Model: {MODEL.__name__} from {WEIGHTS_PATH}")

    test_transform = get_transforms("test", TARGET_SIZE)

    test_dataset = OxfordPetDataset(data_dir=DATA_DIR, split_dir=SPLIT_DIR, mode="test", 
                                    model_name=MODEL_NAME, transform=test_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = MODEL(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    submission_data = []

    with torch.inference_mode():
        for images, filenames, ori_heights, ori_weights in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            
            outputs = forward_pass(model, images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for i in range(images.size(0)):
                file_name = filenames[i]
                heights = ori_heights[i].item()
                weights = ori_weights[i].item()
                
                single_mask = preds[i].unsqueeze(0) # origin was (C, H, W) but F.interpolate only accept (B, C, H, W)
                restored_mask = F.interpolate(single_mask, size=(heights, weights), mode='nearest')
                
                mask_rle = restored_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle_string = rle_encode(mask_rle)
                
                submission_data.append({"image_id": file_name, "encoded_mask": rle_string})

    print("\nGenerating CSV file...")
    csv_path = os.path.join(SUBMISSION_DIR, f"{MODEL.__name__}_submission.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ["image_id", "encoded_mask"] 
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(submission_data)

    print(f"Kaggle submission file successfully generated at: '{csv_path}'")

if __name__ == "__main__":
    main()