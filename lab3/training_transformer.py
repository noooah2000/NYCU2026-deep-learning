import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim, self.scheduler = self.configure_optimizers()
        self.scaler = torch.amp.GradScaler('cuda') if "cuda" in self.args.device else None
        self.prepare_training()
        self.writer = SummaryWriter(log_dir="runs/transformer_experiment")
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0

        device_type = "cuda" if "cuda" in self.args.device else "mps" if "mps" in self.args.device else "cpu"
        for images in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.args.device)
            self.optim.zero_grad()

            with torch.autocast(device_type=device_type):
                logits, z_indices = self.model(images)
                loss = self.loss_func(logits.view(-1, logits.size(-1)), z_indices.view(-1).long())
            epoch_loss += loss.item()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()

        return epoch_loss / len(dataloader)

    def eval_one_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0

        with torch.inference_mode():
            for images in tqdm(dataloader, desc="Validating", leave=False):
                images = images.to(self.args.device)
                logits, z_indices = self.model(images)
                loss = self.loss_func(logits.view(-1, logits.size(-1)), z_indices.view(-1).long())
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return optimizer, scheduler

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_base_data_dir():
    if os.path.exists('/content'):
        return '/content/lab3_dataset'
    elif os.path.exists('/Users/noahhuang/Desktop/DL-NYCU-2026/DL_LAB/lab3_dataset'):
        return '/Users/noahhuang/Desktop/DL-NYCU-2026/DL_LAB/lab3_dataset'
    else:
        return './lab3_dataset'


if __name__ == '__main__':
    set_seed(42)
    base_dir = get_base_data_dir()

    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default=f"{base_dir}/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default=f"{base_dir}/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Noah's Mac detected: Switching to MPS!")
        else:
            args.device = "cpu"
            print("No GPU detected, falling back to CPU.")
    else:
        print(f"Using device: {args.device}")


    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)

    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    args.steps_per_epoch = len(train_loader)
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    best_val_loss = float('inf')
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        train_transformer.scheduler.step()
        valid_loss = train_transformer.eval_one_epoch(val_loader)

        print(f"Epoch[{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f}")
        train_transformer.writer.add_scalar('Loss/Train', train_loss, epoch)
        train_transformer.writer.add_scalar('Loss/Validation', valid_loss, epoch)
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_path = os.path.join("transformer_checkpoints", "best_model.pt")
            torch.save(train_transformer.model.transformer.state_dict(), save_path)
            print(f"<New best model saved with Val Loss: {best_val_loss:.4f}>")
    
    train_transformer.writer.close()
