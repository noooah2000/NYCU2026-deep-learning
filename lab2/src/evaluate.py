import torch
from tqdm.auto import tqdm
from utils import get_dice_score, forward_pass

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0

    with torch.inference_mode():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = forward_pass(model, images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            batch_dice = get_dice_score(preds, masks)
            epoch_dice += batch_dice
            
    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)


