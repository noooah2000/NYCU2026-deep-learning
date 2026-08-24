import os
import json
import torch
import numpy as np
from model import UNet
from ddpm import DDPM, DDIM  
from evaluator import evaluation_model
from config import Config

def generate_cfg_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 載入物件對應表與 Evaluator
    eval_model = evaluation_model()
    with open(f'{Config.DATA_DIR}/objects.json', 'r') as f:
        classes = json.load(f)
        
    # 2. 預處理測試資料
    test_files = [f'{Config.DATA_DIR}/test.json', f'{Config.DATA_DIR}/new_test.json']
    datasets = {}
    for test_file in test_files:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        b_size = len(test_data)
        cond_tensor = torch.zeros((b_size, Config.COND_DIM))
        for i, labels in enumerate(test_data):
            for label in labels:
                cond_tensor[i, classes[label]] = 1.0
        datasets[os.path.basename(test_file)] = cond_tensor.to(device)

    # 3. 載入最佳權重 (請確保這個檔案存在，或是換成你 Epoch 200 的那個檔案)
    checkpoint_path = 'checkpoints/ddpm_unet_epoch_200.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到權重檔: {checkpoint_path}")
        return
        
    print(f"載入權重: {checkpoint_path}")
    unet = UNet(config=Config).to(device)
    unet.load_state_dict(torch.load(checkpoint_path))
    unet.eval()
    
    # 初始化 Sampler (只要初始化一次就好，推論時再傳入不同的 cfg_scale)
    if Config.SAMPLE_METHOD.upper() == 'DDIM':
        sampler = DDIM(model=unet, timesteps=Config.TIMESTEPS, ddim_timesteps=Config.DDIM_TIMESTEPS).to(device)
    else:
        sampler = DDPM(model=unet, timesteps=Config.TIMESTEPS).to(device)

    # 4. 定義要測試的 CFG Scale 列表 (可以自己增減)
    cfg_scales_to_test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    test_accs = []
    new_test_accs = []
    
    print(f"總共要測試 {len(cfg_scales_to_test)} 種 CFG Scale 數值...")

    # 5. 開始推論並收集數據
    for cfg in cfg_scales_to_test:
        print(f"\n[{'='*12} 正在測試 CFG Scale = {cfg:.1f} {'='*12}]")
        
        cond_test = datasets['test.json']
        gen_test = sampler.sample(cond=cond_test, cfg_scale=cfg)
        acc_test = eval_model.eval(gen_test, cond_test)
        
        cond_new = datasets['new_test.json']
        gen_new = sampler.sample(cond=cond_new, cfg_scale=cfg)
        acc_new = eval_model.eval(gen_new, cond_new)
        
        print(f"📊 CFG {cfg:.1f} | test: {acc_test:.4f} | new_test: {acc_new:.4f}")
        
        test_accs.append(acc_test)
        new_test_accs.append(acc_new)
        
    # 6. 將結果存為 JSON
    data_export = {
        "method": Config.SAMPLE_METHOD.upper(),
        "checkpoint_used": os.path.basename(checkpoint_path),
        "cfg_scales": cfg_scales_to_test,
        "test_accs": test_accs,
        "new_test_accs": new_test_accs
    }
    
    with open('cfg_trend_data.json', 'w') as f:
        json.dump(data_export, f, indent=4)
        
    print("\n✅ 數據已成功儲存至 cfg_trend_data.json！現在你可以使用 plot_cfg_trend_data.py 來畫圖了。")

if __name__ == '__main__':
    generate_cfg_data()