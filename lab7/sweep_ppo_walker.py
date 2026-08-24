import os
import sys
import wandb
import argparse
import gymnasium as gym
import torch
import numpy as np
import random

# 🌟 從你的 Task 3 完美版 ppo_walker 匯入
from ppo_walker import PPOAgent, seed_torch

def sweep_train():
    # 1. 讓 WandB 接管初始化
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
    args = parser.parse_args([]) 
    
    # 🌟 重要防撞機制：因為 ppo_walker 裡面的檔名寫死了 (例如 _best.pt, _1m.pt)
    # 我們把每個 Sweep 存到不同的獨立資料夾，避免平行運算時互相覆蓋
    args.save_dir = f"task3_checkpoints_sweep/sweep_{sweep_num}"
    os.makedirs(args.save_dir, exist_ok=True)

    # 3. 綁定你要搜索的超參數 (助教規定必測的 epsilon 與 entropy)
    args.actor_lr = config.actor_lr
    args.critic_lr = config.critic_lr
    args.entropy_weight = config.entropy_weight
    args.epsilon = config.epsilon
    args.seed = config.seed

    # 4. 固定不需要搜索的參數 (完全對齊 Task 3 需求)
    args.discount_factor = 0.99  # 🌟 助教 PDF 規定 Walker 必須鎖定 0.99
    args.tau = 0.95              
    args.batch_size = 64
    args.rollout_len = 2048      # Walker 經典配置
    args.update_epoch = 10       
    args.eval_freq = 10      # 每 50 次 update (約 10 萬步) 嚴格評估一次 20 局
    args.num_episodes = 1500     # 確保總步數大於 3,000,000，才能產出助教要的 5 個 milestones

    # 5. 環境與隨機種子設定
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    
    # 🌟 絕對不能忘記的神級外掛
    env = gym.wrappers.NormalizeObservation(env)
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # 6. 建立 Agent 並開始訓練
    agent = PPOAgent(env, args)
    agent.train()


if __name__ == "__main__":
    
    # 定義 PPO Task 3 專屬的搜索空間
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval_score', 
            'goal': 'maximize'
        },
        'parameters': {
            # MuJoCo 的 Learning Rate 通常不需要太大，這三個是黃金區間
            'actor_lr': {'values': [1e-4, 3e-4]},
            'critic_lr': {'values': [3e-4, 5e-4]},
            
            # 🌟 助教指定實驗 1: Entropy Coefficient (連續控制通常設極小)
            'entropy_weight': {'values': [0.0, 0.001, 0.01]}, 
            
            # 🌟 助教指定實驗 2: Clipping Parameter
            'epsilon': {'values': [0.1, 0.2, 0.3]}, 
            
            # 隨機選幾個 Seed，看哪個天賦異稟
            'seed': {'values': [0, 77, 707]}
        }
    }

    # 判斷執行指令是否帶有 sweep_id 參數
    if len(sys.argv) > 1:
        # 模式 B：我是打工人 (Agent)
        target_sweep_id = sys.argv[1]
        print(f"🚀 啟動 PPO Walker Agent，加入 Sweep: {target_sweep_id}")
        
        # 這裡的 count 可以自己設定，Walker 跑很慢，建議每個 worker 跑 2-3 個就好
        wandb.agent(target_sweep_id, sweep_train, count=3, project="DLP-Lab7-PPO-Walker-Sweep")
        
    else:
        # 模式 A：我是主控台 (創建 Sweep)
        sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-PPO-Walker-Sweep")
        print("="*60)
        print("🎉 成功在 W&B 雲端創建 Task 3 PPO Sweep！")
        print("請打開幾個 tmux 窗格中，並分別輸入：")
        print(f"python sweep_ppo_walker.py {sweep_id}")
        print("="*60)