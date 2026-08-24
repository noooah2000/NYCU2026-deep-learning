import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from dataset import IClevrDataset
from model import UNet
from ddpm import DDPM
from config import Config
import os
import wandb
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project="lab6-generative-models",
        name="ddpm-cfg-optimized",
        config={
            "batch_size": Config.BATCH_SIZE,
            "epochs": Config.EPOCHS,
            "learning_rate": Config.LR,
            "timesteps": Config.TIMESTEPS,
            "time_dim": Config.TIME_DIM,
            "cond_dim": Config.COND_DIM,
            "num_workers": Config.NUM_WORKERS,
            "save_interval": Config.SAVE_INTERVAL 
        }
    )

    dataset = IClevrDataset(
        root_dir=Config.IMG_DIR, 
        json_file=f"{Config.DATA_DIR}/train.json", 
        objects_json=f"{Config.DATA_DIR}/objects.json"
    )
    
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    
    unet = UNet(config=Config).to(device)
    ddpm = DDPM(model=unet, timesteps=Config.TIMESTEPS).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=Config.LR)
    
    epochs = Config.EPOCHS 
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    os.makedirs('checkpoints', exist_ok=True)
    
    best_loss = float('inf') 
    
    for epoch in range(epochs):
        unet.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            b = images.shape[0]
            
            t = torch.randint(0, ddpm.timesteps, (b,), device=device).long()
            noise = torch.randn_like(images).to(device)
            x_t = ddpm.q_sample(images, t, noise)
            
            prob = torch.rand(b, device=device)
            mask = prob < 0.1
            labels[mask] = 0.0
            
            with autocast('cuda'):
                predicted_noise = unet(x_t, t, labels)
                loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": current_lr
        })
        
      
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(unet.state_dict(), 'checkpoints/ddpm_unet_best_loss.pth')
            print(f"Found lower Loss: {best_loss:.4f}, Updated to best_loss.pth")
        
      
        if (epoch + 1) % Config.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            checkpoint_path = f'checkpoints/ddpm_unet_epoch_{epoch+1}.pth'
            torch.save(unet.state_dict(), checkpoint_path)
            
    wandb.finish()

if __name__ == '__main__':
    train()