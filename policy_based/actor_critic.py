"""
Actor-Critic algorithm, page 20 of lecture 7 slides, it learns the policy and the value function simultaneously
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from config import RLConfig, BoardConfig, ModelConfig

class ActorNetwork(nn.Module):
    """
    Actor network for the Actor-Critic algorithm, it parametrizes the policy π(a|s;θ)
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

class CriticNetwork(nn.Module):
    """
    Critic network for the Actor-Critic algorithm, it estimates the value function V(s;w)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single value estimate
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ActorCriticAgent:
    def __init__(self, rl_config: RLConfig, board_config: BoardConfig, model_config: Optional[ModelConfig] = None):
        self.rl_config = rl_config
        self.board_config = board_config
        self.device = torch.device(rl_config.device)
        
        # Initialize dimensions
        self.input_dim = board_config.board_size ** 2
        self.hidden_dim = rl_config.hidden_dim
        self.output_dim = 4  # Four possible actions: up, right, down, left
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)
        
        self.critic = CriticNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=rl_config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=rl_config.learning_rate)
        
        # Training statistics
        self.stats = {
            "actor_loss": [],
            "critic_loss": [],
            "episode_rewards": []
        }

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor and apply log2 transformation."""
        # Flatten the state and convert to tensor
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        # Apply log2 transformation (adding 1 to avoid log(0))
        return torch.log2(state_tensor + 1)

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using the current policy."""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            # Get action probabilities from actor
            action_probs = self.actor(state_tensor)
            # Get value estimate from critic
            value = self.critic(state_tensor)
        
        # Sample action from the probability distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        
        return action.item(), value.item()

    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """Update both actor and critic networks."""
        # Convert everything to tensors
        state_tensor = self.preprocess_state(state)
        next_state_tensor = self.preprocess_state(next_state)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        # Compute advantage
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
            target_value = reward_tensor + self.rl_config.gamma * next_value * (1 - done_tensor)
            advantage = target_value - self.critic(state_tensor)

        # Update Critic
        value = self.critic(state_tensor)
        critic_loss = nn.MSELoss()(value, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        action_probs = self.actor(state_tensor)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_prob = action_distribution.log_prob(action_tensor)
        
        # Add entropy term for exploration
        entropy = action_distribution.entropy()
        
        # Compute actor loss with entropy regularization
        actor_loss = -(log_prob * advantage.detach() + 0.01 * entropy)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Store statistics
        self.stats["actor_loss"].append(actor_loss.item())
        self.stats["critic_loss"].append(critic_loss.item())

    def save_model(self, path: str) -> None:
        """Save actor and critic models."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        """Load actor and critic models."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
