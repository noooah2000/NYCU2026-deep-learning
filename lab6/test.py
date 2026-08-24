import os
import json
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from model import UNet
from ddpm import DDPM, DDIM 
from evaluator import evaluation_model
from config import Config

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('images/test', exist_ok=True)
    os.makedirs('images/new_test', exist_ok=True)
    
    with open(f'{Config.DATA_DIR}/objects.json', 'r') as f:
        classes = json.load(f)
        
    unet = UNet(config=Config).to(device)
    checkpoint_path = 'checkpoints/ddpm_unet_epoch_200.pth' 
    print(f"Loading model from {checkpoint_path}...")
    unet.load_state_dict(torch.load(checkpoint_path))
    unet.eval()
    
    if Config.SAMPLE_METHOD.upper() == 'DDIM':
        print(f"Initializing DDIM Sampler ({Config.DDIM_TIMESTEPS} steps, CFG={Config.CFG_SCALE})")
        sampler = DDIM(model=unet, timesteps=Config.TIMESTEPS, ddim_timesteps=Config.DDIM_TIMESTEPS).to(device)
    else:
        print(f"Initializing DDPM Sampler (1000 steps, CFG={Config.CFG_SCALE})")
        sampler = DDPM(model=unet, timesteps=Config.TIMESTEPS).to(device)

    eval_model = evaluation_model() 
    
    test_files = [f'{Config.DATA_DIR}/test.json', f'{Config.DATA_DIR}/new_test.json']
    out_dirs = ['images/test', 'images/new_test']
    
    for test_file, out_dir in zip(test_files, out_dirs):
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            
        b_size = len(test_data)
        cond_tensor = torch.zeros((b_size, Config.COND_DIM))
        for i, labels in enumerate(test_data):
            for label in labels:
                class_idx = classes[label]
                cond_tensor[i, class_idx] = 1.0
        cond_tensor = cond_tensor.to(device)
        
        filename_only = os.path.basename(test_file)
        print(f"\nGenerating images for {filename_only}...")
        generated_images = sampler.sample(cond=cond_tensor, cfg_scale=Config.CFG_SCALE)
        
        acc = eval_model.eval(generated_images, cond_tensor)
        print(f"Accuracy for {filename_only}: {acc:.4f}")
        
        denorm_images = (generated_images + 1) / 2 
        
        for i, img in enumerate(denorm_images):
            save_image(img, os.path.join(out_dir, f"{i}.png"))
            
        grid = make_grid(denorm_images, nrow=8)
        save_image(grid, f"grid_{filename_only.split('.')[0]}.png")
        
    print("\n" + "="*50)
    print("Generating Denoising Process Grid...")
    
    target_labels = ["red sphere", "cyan cylinder", "cyan cube"]
    denoise_cond = torch.zeros((1, Config.COND_DIM))
    for label in target_labels:
        denoise_cond[0, classes[label]] = 1.0
    denoise_cond = denoise_cond.to(device)
    
    img = torch.randn((1, 3, 64, 64), device=device)
    process_images = []
    
    interval = sampler.timesteps // 10 
    for i in tqdm(reversed(range(sampler.timesteps)), desc="Denoising Process", total=sampler.timesteps):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sampler.p_sample(img, t, denoise_cond, cfg_scale=Config.CFG_SCALE) 
        
        if i % interval == 0 or i == sampler.timesteps - 1:
            process_images.append(img.squeeze(0).clone().cpu())
            
    process_images = torch.stack(process_images)
    process_images = (process_images + 1) / 2
    
    process_grid = make_grid(process_images, nrow=len(process_images))
    save_image(process_grid, "denoising_process.png")
    print("Denoising process saved to denoising_process.png")

if __name__ == '__main__':
    evaluate()