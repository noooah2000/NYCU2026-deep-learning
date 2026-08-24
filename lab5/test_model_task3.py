import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        features = self.backbone(x / 255.0)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        cropped = gray[34:194, :]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        _, thresholded = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return thresholded

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame.copy() for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        step_count = 0

        record_video = (ep < 3)
        video_writer = None
        if record_video:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            video_writer = imageio.get_writer(out_path, fps=30, macro_block_size=1)

        while not done:
            if record_video:
                frame = env.render()
                video_writer.append_data(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

            step_count += 1

        if video_writer:
            video_writer.close()

        all_rewards.append(total_reward)

        print(f"[environment steps: {step_count:8d}], [seed: {args.seed + ep:2d}], [eval reward: {total_reward}]")
    env.close()
    print("\n" + "="*30)
    avg_reward = np.mean(all_rewards)
    print(f"Average: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../LAB5_314551161_task3_best.pt", help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
