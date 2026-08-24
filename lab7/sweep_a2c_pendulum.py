import os
import wandb
import argparse
import gymnasium as gym
import torch
import numpy as np
import random

# 直接從你的作業檔案中匯入 A2CAgent 與設定種子的函式
from a2c_pendulum import A2CAgent, seed_torch

def sweep_train():
    # 1. 讓 WandB 接管初始化，它會自動發派一組實驗參數
    wandb.init()
    config = wandb.config

    if wandb.run:
        sweep_num = wandb.run.name.split('-')[-1]
        custom_run_name = f"Sweep-{sweep_num}" 
        wandb.run.name = custom_run_name
    else:
        sweep_num = "0"

    # 2. 建立參數物件
    parser = argparse.ArgumentParser()
    args = parser.parse_args([]) # 產生一個空的 args 物件
    args.save_dir = "task1_checkpoints_sweep"
    os.makedirs(args.save_dir, exist_ok=True)
    args.model_name = f"LAB7_314551161_task1_a2c_pendulum_sweep{sweep_num}.pt"

    # 3. 綁定你要搜索的超參數 (從 config 讀取)
    args.actor_lr = config.actor_lr
    args.critic_lr = config.critic_lr
    args.entropy_weight = config.entropy_weight
    
    # 🌟 新增：從 config 讀取 discount_factor 與 seed
    args.discount_factor = config.discount_factor
    args.seed = config.seed

    # 4. 固定那些不需要搜索的參數
    args.eval_freq = 20
    args.num_episodes = 1000 

    # 5. 環境與隨機種子設定
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # 🌟 修改：確保所有的環境和庫都使用 Sweep 派發下來的種子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # 6. 建立 Agent 並開始訓練
    agent = A2CAgent(env, args)
    agent.train()

if __name__ == "__main__":
    import sys
    
    # 定義搜索空間 (Sweep Configuration)
    sweep_config = {
        'method': 'bayes', # 繼續維持 bayes，讓它在限縮後的健康區間內做精準微調
        'metric': {
            'name': 'eval_score', # 嚴格對齊考場的平均分數
            'goal': 'maximize'
        },
        'parameters': {
            'actor_lr': {'values': [0.0001]},
            'critic_lr': {'values': [0.0008]},
            'entropy_weight': {'values': [0, 0.0001, 0.00003]},
            
            # 🌟 新增：將 discount_factor 加入搜尋
            # 助教的 PDF 講義有特別提到 Pendulum 預設 0.9 是故意的，可以多測 0.9, 0.93, 0.95
            'discount_factor': {'values': [0.9, 0.93, 0.95]}, 
            
            # 🌟 新增：將 seed 加入搜尋 (通常測試幾個具代表性的質數或幸運數字)
            'seed': {'values': [0, 42, 77, 707, 1024]}
        }
    }

    # 判斷執行指令是否帶有 sweep_id 參數
    if len(sys.argv) > 1:
        # 模式 B：我是打工人 (Agent)
        target_sweep_id = sys.argv[1]
        print(f"🚀 啟動 Agent，加入 Sweep: {target_sweep_id}")
        
        # 這裡的 count=5 代表「這個打工人」最多做 5 份工作就下班
        wandb.agent(target_sweep_id, sweep_train, count=5, project="DLP-Lab7-A2C-Pendulum-Sweep")
        
    else:
        # 模式 A：我是主控台 (創建 Sweep)
        sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-A2C-Pendulum-Sweep")
        print("="*60)
        print("🎉 成功在 W&B 雲端創建 Sweep！")
        print("請打開 4 個終端機分頁，並在每個分頁分別輸入以下指令來啟動平行運算：")
        print(f"python sweep_a2c_pendulum.py {sweep_id}")
        print("="*60)