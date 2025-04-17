"""
Policy based agent using Monte Carlo sampling, called Policy Gradient Monte Carlo
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from config import RLConfig, BoardConfig, ModelConfig
from typing import Optional

class PolicyNetwork(nn.Module):
    """
    Policy network for the Policy Gradient Monte Carlo algorithm, it parametrizes the policy π(a|s;θ)
    """
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

class PolicyBasedAgent:
    def __init__(self, rl_config: RLConfig, board_config: BoardConfig, model_config: Optional[ModelConfig] = None):
        self.rl_config = rl_config
        self.board_config = board_config
        self.device = torch.device(rl_config.device)
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(
            input_dim=self.board_config.board_size ** 2,
            hidden_dim=rl_config.hidden_dim,
            output_dim=4  # Four possible actions: up, right, down, left
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=rl_config.learning_rate
        )
        
        # Episode memory for storing trajectories
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training statistics
        self.stats = {
            "policy_loss": [],
            "episode_rewards": [],
            "episode_lengths": []
        }
        
        self.steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using the policy network."""
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Store transition and update policy if episode is done."""
        # Store the transition
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        if done:
            self._update_policy()
            
            # Clear episode memory
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []

    def _update_policy(self):
        """Update policy network using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return
            
        # Convert to tensors
        states = torch.FloatTensor(np.array([s.flatten() for s in self.episode_states])).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        
        # Calculate discounted rewards
        returns = self._compute_returns(self.episode_rewards)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get action probabilities
        action_probs = self.policy_network(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        # Calculate loss
        policy_loss = -(returns * torch.log(selected_probs.squeeze(1))).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Store statistics
        self.stats["policy_loss"].append(policy_loss.item())
        self.stats["episode_rewards"].append(sum(self.episode_rewards))
        self.stats["episode_lengths"].append(len(self.episode_rewards))
        
        self.steps += 1

    def _compute_returns(self, rewards: list) -> list:
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.rl_config.gamma * G
            returns.insert(0, G)
        return returns 