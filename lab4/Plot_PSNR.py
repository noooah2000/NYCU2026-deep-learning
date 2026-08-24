import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 從你原本的檔案中匯入必要的模組
from Trainer import VAE_Model, Generate_PSNR
from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

class PSNR_Plotter(VAE_Model):
    def __init__(self, args):
        # 🌟 繞過 VAE_Model 的 __init__，避免初始化 SummaryWriter 和 Optimizer
        super(VAE_Model, self).__init__() 
        self.args = args
        
        # 僅初始化網路架構與必要組件
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.mse_criterion = nn.MSELoss()
        self.val_vi_len = args.val_vi_len

    @torch.no_grad()
    def plot_psnr(self):
        # 🔥 使用最高級護盾：直接呼叫 nn.Module 的 eval 模式
        nn.Module.eval(self)
        
        val_loader = self.val_dataloader()
        all_psnr_seqs = []
        
        print(f"正在載入權重並計算 PSNR (對象: {self.args.ckpt_path})...")
        
        for idx, (img, label) in enumerate(tqdm(val_loader, ncols=100)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            
            B = img.size(0)
            seq_len = img.size(1) 
            
            psnr_list = []
            prev_frame = img[:, 0]
            
            # 自迴歸推論循環
            for t in range(1, seq_len):
                F_prev = self.frame_transformation(prev_frame)
                L_curr = self.label_transformation(label[:, t])
                
                # 測試階段改用隨機抽樣 z
                z = torch.randn(B, self.args.N_dim, F_prev.size(2), F_prev.size(3)).to(self.args.device)
                dec_out = self.Decoder_Fusion(F_prev, L_curr, z)
                gen_frame = torch.sigmoid(self.Generator(dec_out))
                
                # 計算與 Ground Truth 的誤差
                psnr = Generate_PSNR(gen_frame, img[:, t]).item()
                psnr_list.append(psnr)
                
                # 接龍：將預測結果作為下一幀輸入
                prev_frame = gen_frame  
                
            all_psnr_seqs.append(psnr_list)
            
        # 計算所有 Validation 資料的平均值
        avg_psnr_per_frame = np.mean(all_psnr_seqs, axis=0)
        
        # --- 繪圖邏輯 ---
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, seq_len), avg_psnr_per_frame, label='Per-frame PSNR', color='#1f77b4')
        
        # 加入平均線
        mean_val = np.mean(avg_psnr_per_frame)
        plt.axhline(y=mean_val, color='r', linestyle='--', label=f'Avg PSNR: {mean_val:.2f}')
        
        plt.title('PSNR-per-frame Diagram (Validation Set)', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 儲存
        save_name = "psnr_plot.png"
        plt.savefig(os.path.join(self.args.save_root, save_name), dpi=300)
        print(f"成功！折線圖已儲存至: {os.path.join(self.args.save_root, save_name)}")

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = PSNR_Plotter(args).to(args.device)
    
    # 載入權重
    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"已成功載入最佳模型權重。")
    
    model.plot_psnr()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 核心必要參數
    parser.add_argument('--DR',            type=str, default="../lab_data/lab4_data", help="資料集路徑")
    parser.add_argument('--save_root',     type=str, default="./checkpoints", help="圖片儲存路徑")
    parser.add_argument('--ckpt_path',     type=str, default="./checkpoints/best_model.ckpt", help="權重路徑")
    parser.add_argument('--device',        type=str, default="cuda")
    
    # 資料與架構參數 (必須與訓練時一致)
    parser.add_argument('--batch_size',    type=int, default=1)
    parser.add_argument('--val_vi_len',    type=int, default=630)
    parser.add_argument('--frame_H',       type=int, default=32)
    parser.add_argument('--frame_W',       type=int, default=64)
    parser.add_argument('--num_workers',   type=int, default=4)
    
    parser.add_argument('--F_dim',         type=int, default=128)
    parser.add_argument('--L_dim',         type=int, default=32)
    parser.add_argument('--N_dim',         type=int, default=12)
    parser.add_argument('--D_out_dim',     type=int, default=192)

    args = parser.parse_args()
    main(args)