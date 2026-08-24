import os
import sys
import wandb
import argparse
import gymnasium as gym
import torch
import numpy as np
import random

# 從 Task 2 的 ppo_pendulum 匯入 Agent 與設定種子的函式
from ppo_pendulum import PPOAgent, seed_torch

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
    
    # Task 2 專屬的存檔資料夾與檔名
    args.save_dir = "task2_checkpoints_sweep"
    os.makedirs(args.save_dir, exist_ok=True)
    args.model_name = f"LAB7_314551161_task2_ppo_pendulum_sweep{sweep_num}.pt"

    # 3. 綁定你要搜索的超參數 (加入 PPO 特有的 epsilon)
    args.actor_lr = config.actor_lr
    args.critic_lr = config.critic_lr
    args.entropy_weight = config.entropy_weight
    args.epsilon = config.epsilon
    
    # 🌟 修正：從 W&B config 讀取你的幸運參數，不要寫死！
    args.discount_factor = config.discount_factor
    args.seed = config.seed

    # 4. 固定不需要搜索的參數
    args.tau = 0.95              # GAE 的衰減係數 (0.95 是實務上最穩定的預設值)
    args.batch_size = 64
    args.rollout_len = 2000
    args.update_epoch = 10       # 每次 rollout 收集完後，重複訓練網路 10 次
    args.eval_freq = 2           # 每 5 次 update (約 10,000 步) 就評估並存檔一次
    args.num_episodes = 100      # 設定為 100 次 update，對應助教的 200k 步

    # 5. 環境與隨機種子設定
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # 🌟 修正：確保這裡吃的是 args.seed，而不是寫死的 77
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # 6. 建立 Agent 並開始訓練
    agent = PPOAgent(env, args)
    agent.train()


if __name__ == "__main__":
    
    # 定義 PPO 專屬的搜索空間
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval_score', 
            'goal': 'maximize'
        },
        'parameters': {
            # 🌟 融入 Task 1 的黃金參數，讓 Agent 贏在起跑點
            'actor_lr': {'values': [5e-4, 1e-4, 3e-4]},
            'critic_lr': {'values': [3e-3, 8e-4, 1e-3]},
            
            # 既然 0 已經證明是最完美的，我們直接鎖死 0，節省搜尋時間
            'entropy_weight': {'values': [0.0]}, 
            
            # PPO 的核心靈魂：限制 Actor 更新幅度
            'epsilon': {'values': [0.1, 0.2, 0.3]}, 
            
            # 🌟 放入你驗證過最強的 0.95
            'discount_factor': {'values': [0.90, 0.95, 0.99]},
            
            # 🌟 放入你的幸運數字 707
            'seed': {'values': [77, 707, 1024]}
        }
    }

    # 判斷執行指令是否帶有 sweep_id 參數
    if len(sys.argv) > 1:
        # 模式 B：我是打工人 (Agent)
        target_sweep_id = sys.argv[1]
        print(f"🚀 啟動 PPO Agent，加入 Sweep: {target_sweep_id}")
        
        wandb.agent(target_sweep_id, sweep_train, count=5, project="DLP-Lab7-PPO-Pendulum-Sweep")
        
    else:
        # 模式 A：我是主控台 (創建 Sweep)
        sweep_id = wandb.sweep(sweep_config, project="DLP-Lab7-PPO-Pendulum-Sweep")
        print("="*60)
        print("🎉 成功在 W&B 雲端創建 Task 2 PPO Sweep！")
        print("請打開 4 個 tmux 窗格中，並分別輸入：")
        print(f"python sweep_ppo_pendulum.py {sweep_id}")
        print("="*60)