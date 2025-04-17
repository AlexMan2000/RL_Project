"""
Proximal Policy Optimization, page 24 of lecture 7 slides, it uses the policy gradient theorem to update the policy parameters θ,
it learns the policy and the value function simultaneously
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from config import RLConfig, BoardConfig, ModelConfig

class PPOMemory:
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches

class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim  # Store input dimension for reshaping
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Handle both single state and batch of states
        if state.dim() == 1:
            # Single state case
            state = state.view(1, self.input_dim)
        else:
            # Batch case - ensure proper shape
            state = state.view(-1, self.input_dim)
        return self.network(state)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim  # Store input dimension for reshaping
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Handle both single state and batch of states
        if state.dim() == 1:
            # Single state case
            state = state.view(1, self.input_dim)
        else:
            # Batch case - ensure proper shape
            state = state.view(-1, self.input_dim)
        return self.network(state)

class PPOAgent:
    def __init__(self, rl_config: RLConfig, board_config: BoardConfig, model_config: Optional[ModelConfig] = None):
        self.rl_config = rl_config
        self.board_config = board_config
        self.device = torch.device(rl_config.device)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95
        self.value_clip_range = 0.2
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Initialize dimensions
        self.input_dim = board_config.board_size ** 2
        self.hidden_dim = rl_config.hidden_dim
        self.output_dim = 4  # Four possible actions: up, right, down, left
        
        # Initialize networks
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
        
        # Initialize memory
        self.memory = PPOMemory(rl_config.batch_size)
        
        # Store last log probability for update
        self.last_log_prob = None
        
        # Training statistics
        self.stats = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": []
        }

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor and apply log2 transformation."""
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        return torch.log2(state_tensor + 1)

    def select_action(self, state: np.ndarray) -> int:
        """Select action using the current policy."""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            self.last_log_prob = action_distribution.log_prob(action).item()
        
        return action.item()

    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
                
            delta = rewards[step] + self.rl_config.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.rl_config.gamma * self.gae_lambda * next_non_terminal * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return returns, advantages

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience and update policy if enough samples are collected."""
        # Get value estimate for the next state
        with torch.no_grad():
            next_state_tensor = self.preprocess_state(next_state)
            next_value = self.critic(next_state_tensor).item()
            
        # Store experience in memory
        self.memory.store(state, action, self.last_log_prob, next_value, reward, done)
        
        # Only update if we have enough samples
        if len(self.memory.states) >= self.rl_config.batch_size:
            self._update_networks(next_value)
            
    def _update_networks(self, next_value: float = 0) -> None:
        """Update policy and value networks using PPO."""
        states, actions, old_probs, values, rewards, dones, batches = \
            self.memory.generate_batches()
            
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # Ensure proper state tensor shape for network input
        states = torch.FloatTensor(np.log2(states + 1)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]
                batch_values = values[batch]
                
                # Get current action probabilities and values
                action_probs = self.actor(batch_states)  # Shape will be handled in forward pass
                current_values = self.critic(batch_states)  # Shape will be handled in forward pass
                
                # Get distribution
                dist = torch.distributions.Categorical(action_probs)
                current_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute probability ratio
                prob_ratio = torch.exp(current_probs - batch_old_probs)
                
                # Compute PPO policy loss
                surr1 = prob_ratio * batch_advantages
                surr2 = torch.clamp(prob_ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss with clipping
                value_pred_clipped = batch_values + \
                    torch.clamp(current_values - batch_values, 
                              -self.value_clip_range, self.value_clip_range)
                value_losses = (current_values - batch_returns) ** 2
                value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss = actor_loss - self.entropy_coef * entropy  # Add entropy bonus
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Store statistics
                self.stats["actor_loss"].append(actor_loss.item())
                self.stats["critic_loss"].append(critic_loss.item())
                self.stats["entropy"].append(entropy.item())
        
        self.memory.clear()

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
