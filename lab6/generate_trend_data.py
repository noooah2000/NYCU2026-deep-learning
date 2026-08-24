import os
import re
import json
import torch
from tqdm import tqdm
from model import UNet
from ddpm import DDPM, DDIM  
from evaluator import evaluation_model
from config import Config

def generate_data():
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

    # 3. 掃描 checkpoints 資料夾並排序
    checkpoint_dir = 'checkpoints'
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        match = re.match(r'ddpm_unet_epoch_(\d+)\.pth', f)
        if match:
            checkpoints.append((int(match.group(1)), os.path.join(checkpoint_dir, f)))
            
    checkpoints.sort(key=lambda x: x[0]) 
    
    if not checkpoints:
        print("沒有找到任何符合命名規則的權重檔")
        return

    print(f"總共找到 {len(checkpoints)} 個 Checkpoints 準備進行測試...")

    unet = UNet(config=Config).to(device)
    
    epochs_list = []
    test_accs = []
    new_test_accs = []
    
    # 4. 開始推論並收集數據
    for epoch, cp_path in checkpoints:
        filename = os.path.basename(cp_path)
        print(f"\n[{'='*12} 正在測試: {filename} {'='*12}]")
        
        unet.load_state_dict(torch.load(cp_path))
        unet.eval()
        
        if Config.SAMPLE_METHOD.upper() == 'DDIM':
            sampler = DDIM(model=unet, timesteps=Config.TIMESTEPS, ddim_timesteps=Config.DDIM_TIMESTEPS).to(device)
        else:
            sampler = DDPM(model=unet, timesteps=Config.TIMESTEPS).to(device)
            
        cond_test = datasets['test.json']
        gen_test = sampler.sample(cond=cond_test, cfg_scale=Config.CFG_SCALE)
        acc_test = eval_model.eval(gen_test, cond_test)
        
        cond_new = datasets['new_test.json']
        gen_new = sampler.sample(cond=cond_new, cfg_scale=Config.CFG_SCALE)
        acc_new = eval_model.eval(gen_new, cond_new)
        
        print(f"📊 test.json: {acc_test:.4f} | new_test.json: {acc_new:.4f}")
        
        epochs_list.append(epoch)
        test_accs.append(acc_test)
        new_test_accs.append(acc_new)
        
    # 5. 將結果存為 JSON，以便後續無限次畫圖
    data_export = {
        "method": Config.SAMPLE_METHOD.upper(),
        "cfg_scale": Config.CFG_SCALE,
        "epochs": epochs_list,
        "test_accs": test_accs,
        "new_test_accs": new_test_accs
    }
    
    with open('trend_data.json', 'w') as f:
        json.dump(data_export, f, indent=4)
        
    print("\n✅ 數據已成功儲存至 trend_data.json！現在你可以使用 plot_trend_data.py 來畫圖了。")

if __name__ == '__main__':
    generate_data()