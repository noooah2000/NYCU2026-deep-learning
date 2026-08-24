#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def initialize_kaiming(layer: nn.Linear):
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.5))

        initialize_kaiming(self.fc1)
        initialize_kaiming(self.fc2)
        initialize_uniformly(self.mu_layer)
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_layer(x)) * 2.0
        std = self.log_std.clamp(-10, 0).exp().expand_as(mu)

        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

        initialize_kaiming(self.fc1)
        initialize_kaiming(self.fc2)
        initialize_uniformly(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.eval_env = gym.make("Pendulum-v1")
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.eval_freq = args.eval_freq
        self.save_dir = args.save_dir
        self.model_name = args.model_name
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=400, gamma=0.5)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=400, gamma=0.5)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        scaled_state = np.array(state, dtype=np.float32)
        scaled_state[2] = scaled_state[2] / 8.0
        return torch.FloatTensor(scaled_state).to(self.device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_t = self.preprocess_state(state)
        action, dist = self.actor(state_t)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, terminated])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, terminated = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - int(terminated)
        
        ############TODO#############
        reward_scaled = (reward + 8.0) / 8.0
        state_t = self.preprocess_state(state)
        next_state_t = self.preprocess_state(next_state)

        reward_t = torch.FloatTensor([reward_scaled]).to(self.device)
        mask_t = torch.FloatTensor([mask]).to(self.device)

        current_value = self.critic(state_t)
        with torch.no_grad():
            next_value = self.critic(next_state_t)

        target_value = reward_t + self.gamma * next_value * mask_t
        value_loss = F.mse_loss(current_value, target_value)
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        advantage = (target_value - current_value).detach()
        policy_loss = - (log_prob * advantage)
        
        _, dist = self.actor(state_t)
        entropy = dist.entropy().sum(dim=-1)
        policy_loss = policy_loss - self.entropy_weight * entropy
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_eval_score = -float('inf')
        
        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes)):
            actor_losses, critic_losses, scores = [], [], []
            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    tqdm.write(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "step": step_count,
                        "episode": ep,
                        "return": score,
                        "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
                        "critic_lr": self.critic_optimizer.param_groups[0]['lr']
                        })  
                    
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            if ep % self.eval_freq == 0:
                avg_eval_score = self.evaluate(num_eval_episodes=20)
                tqdm.write(f"--- (Episode {ep}, Step {step_count}) --- Avg Score: {avg_eval_score:.2f}")
                
                wandb.log({
                    "eval_score": avg_eval_score,
                    "step": step_count
                })
                
                if avg_eval_score > best_eval_score:
                    best_eval_score = avg_eval_score
                    
                    save_path = os.path.join(self.save_dir, self.model_name)
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'step': step_count,
                        'best_score': best_eval_score
                    }, save_path)
                    
                    tqdm.write(f"Best Score: {best_eval_score:.2f} (Saved to {save_path})")

    def test(self, model_path: str, num_test_episodes: int = 20):
        """Test the agent for TA evaluation."""
        self.is_test = True
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        global_train_steps = checkpoint.get('step', 'Unknown')
        print(f"Successfully loaded model from: {model_path}")
        print("-" * 50)
        
        total_scores = []
        
        for i in range(num_test_episodes):
            test_seed = i  
            state, _ = self.env.reset(seed=test_seed)
            done = False
            score = 0
            step_count = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                step_count += 1
                
            total_scores.append(score)
            print(f"Environment steps: {global_train_steps}, seed: {test_seed}, eval reward: {score:.0f}")
            
        print("-" * 50)
        print(f"Average reward: {np.mean(total_scores):.2f}")
        print("-" * 50)
        
        self.is_test = False

    def evaluate(self, num_eval_episodes: int = 20) -> float:
        self.is_test = True 
        total_scores = []
        for i in range(num_eval_episodes):
            test_seed = i
            state, _ = self.eval_env.reset(seed=test_seed)
            done = False
            score = 0

            while not done:
                with torch.no_grad():
                    action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                state = next_state
                score += reward
                
            total_scores.append(score)
            
        self.is_test = False 
        return float(np.mean(total_scores))

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=0.0001)
    parser.add_argument("--critic-lr", type=float, default=0.0008)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--entropy-weight", type=float, default=0) # entropy can be disabled by setting this to 0

    parser.add_argument("--save-dir", type=str, default="task1_checkpoints", help="Directory to save model weights")
    parser.add_argument("--eval-freq", type=int, default=20, help="Evaluate and save model every N episodes")
    parser.add_argument("--model-name", type=str, default="LAB7_314551161_task1_a2c_pendulum.pt", help="Name of the saved model file")
    parser.add_argument("--test-only", action="store_true", help="Enable this flag to run the TA evaluation mode")
    parser.add_argument("--model-path", type=str, default="LAB7_314551161_task1_a2c_pendulum.pt", help="Path to the trained .pt model checkpoint for testing")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    agent = A2CAgent(env, args)
    if args.test_only:
        if not args.model_path:
            print("Error: Please provide the model checkpoint path using --model-path!")
        else:
            agent.test(model_path=args.model_path, num_test_episodes=20)
    else:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
        os.makedirs(args.save_dir, exist_ok=True)
        agent.train()