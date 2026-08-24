#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple
import os
import gymnasium as gym
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

def initialize_kaiming(layer: nn.Linear):
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -10,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

        initialize_kaiming(self.fc1)
        initialize_kaiming(self.fc2)
        init_layer_uniform(self.mu_layer)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_layer(x)) 
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mu)
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
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)

        initialize_kaiming(self.fc1)
        initialize_kaiming(self.fc2)
        init_layer_uniform(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.eval_env = gym.make("Walker2d-v5")
        self.eval_env = gym.wrappers.NormalizeObservation(self.eval_env)
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.save_dir = args.save_dir
        self.eval_freq = args.eval_freq
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=400, gamma=0.5)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=400, gamma=0.5)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

        # Checkpoints
        self.milestones = {
            1000000: "1m",
            1500000: "1p5m",
            2000000: "2m",
            2500000: "2p5m",
            3000000: "3m"
        }

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        scaled_state = np.copy(state).astype(np.float32)
        return torch.FloatTensor(scaled_state).to(self.device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action
        selected_action = selected_action.clamp(-1.0, 1.0)

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(
                dist.log_prob(selected_action).sum(dim=-1, keepdim=True).detach()
            )

        return selected_action.cpu().detach().numpy().reshape(self.action_dim)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = bool(terminated or truncated)

        next_state_arr = np.array(next_state, dtype=np.float32)
        reward_float = float(reward)

        if not self.is_test:
            self.rewards.append(torch.tensor([[reward_float]], dtype=torch.float32, device=self.device))
            self.masks.append(torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=self.device))

        return next_state_arr, reward_float, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy.mean()
            #############################

            # critic_loss
            ############TODO#############
            current_value = self.critic(state)
            critic_loss = F.mse_loss(current_value, return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False
        best_eval_score = -float('inf')
        self.best_saved_score = -float('inf')

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        score = 0         
        episode_count = 0
        
        for update_idx in tqdm(range(1, self.num_episodes + 1)):
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = np.expand_dims(next_state, axis=0)
                score += reward 

                if self.total_step in self.milestones:
                    m_name = self.milestones[self.total_step]
                    save_path = os.path.join(self.save_dir, f"LAB7_314551161_task3_ppo_{m_name}.pt")
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'step': self.total_step,
                        'obs_rms': copy.deepcopy(self.env.obs_rms) if hasattr(self.env, "obs_rms") else None
                    }, save_path)
                    tqdm.write(f"\n[Milestone Reached] {m_name} steps! Saved to {save_path}\n")

                if done:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    
                    tqdm.write(f"Episode {episode_count} (Step {self.total_step}): Total Reward = {score:.2f}")
                    if wandb.run:
                        wandb.log({"step": self.total_step, "episode": episode_count, "return": score})
                    score = 0 

            actor_loss, critic_loss = self.update_model(next_state)
            
            if wandb.run:
                wandb.log({
                    "step": self.total_step,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "actor lr": self.actor_optimizer.param_groups[0]['lr'],
                    "critic lr": self.critic_optimizer.param_groups[0]['lr']
                })

            self.actor_scheduler.step()
            self.critic_scheduler.step()
                
            if update_idx % self.eval_freq == 0:
                avg_eval_score = self.evaluate() 
                tqdm.write(f"--- (Update {update_idx}, Step {self.total_step}) --- Avg Eval Score: {avg_eval_score:.2f}")
                
                if wandb.run:
                    wandb.log({"eval_score": avg_eval_score, "step": self.total_step})
                
                if avg_eval_score > best_eval_score:
                    best_eval_score = avg_eval_score 

                    if self.total_step <= 1000000:
                        self.best_saved_score = best_eval_score
                        save_path = os.path.join(self.save_dir, "LAB7_314551161_task3_best.pt")
                        torch.save({
                            'actor_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'step': self.total_step,
                            'best_score': best_eval_score,
                            'obs_rms': copy.deepcopy(self.env.obs_rms) if hasattr(self.env, "obs_rms") else None
                        }, save_path)
                        tqdm.write(f"Best Score: {best_eval_score:.2f} (Saved to {save_path})")
                    else:
                        tqdm.write(f" Found New Hightest: {best_eval_score:.2f}")

                if self.total_step >= 1000000:
                    if not hasattr(self, 'early_stop_checked'):
                        self.early_stop_checked = True
                        if self.best_saved_score < 2500:
                            tqdm.write(f"\n🛑 [Early Stopping] {self.best_saved_score:.2f} < 2500。\n")
                            self.env.close()
                            return
                        else:
                            tqdm.write(f"\n✅ [Keep Going] {self.best_saved_score:.2f} \n")
        # termination
        self.env.close()

    def evaluate(self, num_eval_episodes: int = 20) -> float:
        self.is_test = True 
        total_scores = []

        if hasattr(self.env, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(self.env.obs_rms)

        for i in range(num_eval_episodes):
            test_seed = i
            state, _ = self.eval_env.reset(seed=test_seed)
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                state = np.expand_dims(next_state, axis=0)
                score += float(reward)

            total_scores.append(score)
            
        self.is_test = False 
        return float(np.mean(total_scores))

    def test(self, model_path: str, num_test_episodes: int = 20):
        """Test the agent for TA Evaluation."""
        self.is_test = True
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        global_train_steps = checkpoint.get('step', 'Unknown')
        
        print(f"Successfully loaded model from: {model_path}")
        print("-" * 50)
        
        if hasattr(self.env, "obs_rms") and 'obs_rms' in checkpoint and checkpoint['obs_rms'] is not None:
            self.eval_env.obs_rms = copy.deepcopy(checkpoint['obs_rms'])
        elif hasattr(self.env, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(self.env.obs_rms)

        total_scores = []
        for i in range(num_test_episodes):
            test_seed = i  
            state, _ = self.eval_env.reset(seed=test_seed)
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                state = np.expand_dims(next_state, axis=0)
                score += reward
                
            total_scores.append(score)
            print(f"Environment steps: {global_train_steps}, seed: {test_seed}, eval reward: {score:.0f}")
            
        print("-" * 50)
        print(f"Average reward: {np.mean(total_scores):.2f}")
        print("-" * 50)
        self.is_test = False
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=0.0001)
    parser.add_argument("--critic-lr", type=float, default=0.0005)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--update-epoch", type=int, default=10)

    parser.add_argument("--save-dir", type=str, default="task3_checkpoints", help="Directory to save model weights")
    parser.add_argument("--eval-freq", type=int, default=10, help="Evaluate and save model every N updates")
    parser.add_argument("--test-only", action="store_true", help="Enable this flag to run the TA evaluation mode")
    parser.add_argument("--model-path", type=str, default="LAB7_314551161_task3_ppo_1m.pt", help="Path to the trained .pt model checkpoint for testing")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = gym.wrappers.NormalizeObservation(env)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    agent = PPOAgent(env, args)
    if args.test_only:
        if not args.model_path:
            print("Error: Please provide the model checkpoint path using --model-path!")
        else:
            agent.test(model_path=args.model_path, num_test_episodes=20)
    else:
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
        os.makedirs(args.save_dir, exist_ok=True)
        agent.train()