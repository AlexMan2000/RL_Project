import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import random

class PolicyBasedAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize policy and value networks
        self.policy_network = PolicyNetwork(
            input_dim=16,  # 4x4 board
            hidden_dim=config.hidden_dim,
            output_dim=4  # 4 possible actions
        ).to(self.device)
        
        self.value_network = ValueNetwork(
            input_dim=16,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.memory = []
        
        # Training statistics
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "episode_rewards": []
        }

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using the policy network."""
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            
        return action, log_prob.item()

    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool, log_prob: float):
        """Store experience for later update."""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
        
        if done:
            self._update_networks()
            self.memory = []

    def _update_networks(self):
        """Update policy and value networks using PPO."""
        if not self.memory:
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor(np.array([m['state'] for m in self.memory])).to(self.device)
        actions = torch.LongTensor([m['action'] for m in self.memory]).to(self.device)
        rewards = torch.FloatTensor([m['reward'] for m in self.memory]).to(self.device)
        next_states = torch.FloatTensor(np.array([m['next_state'] for m in self.memory])).to(self.device)
        dones = torch.FloatTensor([m['done'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        
        # Compute returns
        returns = self._compute_returns(rewards, dones)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy network
        for _ in range(self.config.ppo_epochs):
            # Get new action probabilities
            action_probs = self.policy_network(states)
            new_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate loss
            advantages = returns - self.value_network(states).squeeze()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip,
                              1 + self.config.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Add entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
            policy_loss = policy_loss - self.config.entropy_coef * entropy
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Store statistics
            self.stats["policy_loss"].append(policy_loss.item())
            self.stats["entropy"].append(entropy.item())
        
        # Update value network
        for _ in range(self.config.ppo_epochs):
            value_pred = self.value_network(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            self.stats["value_loss"].append(value_loss.item())

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.config.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(self.device)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state) 