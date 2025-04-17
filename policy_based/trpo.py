"""
Trust Region Policy Optimization, page 23 of lecture 7 slides, it uses the policy gradient theorem to update the policy parameters θ,
it learns the policy and the value function simultaneously
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from config import RLConfig, BoardConfig, ModelConfig

class TRPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batch(self):
        return np.array(self.states), np.array(self.actions), \
               np.array(self.rewards), np.array(self.values), \
               np.array(self.log_probs), np.array(self.dones)

class ActorNetwork(nn.Module):
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

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)

    def get_kl(self, states: torch.Tensor, old_probs: torch.Tensor) -> torch.Tensor:
        new_probs = self.forward(states)
        old_probs = old_probs.detach()
        kl = torch.sum(old_probs * (torch.log(old_probs) - torch.log(new_probs)), dim=1)
        return kl.mean()

class CriticNetwork(nn.Module):
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

class TRPOAgent:
    def __init__(self, rl_config: RLConfig, board_config: BoardConfig, model_config: Optional[ModelConfig] = None):
        self.rl_config = rl_config
        self.board_config = board_config
        self.device = torch.device(rl_config.device)
        
        # TRPO hyperparameters
        self.max_kl = 0.01
        self.damping = 0.1
        self.value_train_iters = 10
        self.gae_lambda = 0.95
        
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
        
        # Initialize critic optimizer (actor uses natural gradient)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=rl_config.learning_rate)
        
        # Initialize memory
        self.memory = TRPOMemory()
        
        # Store last log probability for update
        self.last_log_prob = None
        
        # Training statistics
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "kl_div": [],
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
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            self.last_log_prob = dist.log_prob(action).item()
        
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
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
                
            delta = rewards[step] + self.rl_config.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.rl_config.gamma * self.gae_lambda * next_non_terminal * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return returns, advantages

    def conjugate_gradient(self, states: torch.Tensor, grad: torch.Tensor, nsteps: int = 10) -> torch.Tensor:
        """Conjugate gradient algorithm to compute x = H^(-1)g."""
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        
        for i in range(nsteps):
            Hp = self.hessian_vector_product(states, p)
            alpha = r.dot(r) / (p.dot(Hp) + self.damping)
            x += alpha * p
            r_new = r - alpha * Hp
            beta = r_new.dot(r_new) / r.dot(r)
            r = r_new
            p = r + beta * p
            
        return x

    def hessian_vector_product(self, states: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian-vector product using the policy's KL divergence."""
        kl = self.actor.get_kl(states, self.actor(states))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
        
        grad_vector_product = kl_grad.dot(vector)
        grad_grad = torch.autograd.grad(grad_vector_product, self.actor.parameters())
        grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grad_grad])
        
        return grad_grad

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience and update policy if enough samples are collected."""
        # Get value estimate for the next state
        with torch.no_grad():
            next_state_tensor = self.preprocess_state(next_state)
            next_value = self.critic(next_state_tensor).item()
            
        # Store experience in memory
        self.memory.store(state, action, reward, next_value, self.last_log_prob, done)
        
        # Only update if we have enough samples
        if len(self.memory.states) >= self.rl_config.batch_size:
            self._update_networks(next_value)
            
    def _update_networks(self, next_value: float = 0) -> None:
        """Update policy using TRPO and value function using standard gradient descent."""
        # Get batch data
        states, actions, rewards, values, log_probs, dones = self.memory.get_batch()
        
        # Convert to tensors
        states = torch.FloatTensor(np.log2(states + 1)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        for _ in range(self.value_train_iters):
            value_pred = self.critic(states)
            value_loss = 0.5 * (returns - value_pred.squeeze()).pow(2).mean()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
            
            self.stats["value_loss"].append(value_loss.item())
        
        # Compute policy gradient
        new_log_probs = self.actor.get_log_prob(states, actions)
        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -(policy_ratio * advantages).mean()
        
        # Compute gradient of policy loss
        self.actor.zero_grad()
        policy_loss.backward()
        policy_grad = torch.cat([param.grad.view(-1) for param in self.actor.parameters()])
        
        # Compute natural gradient using conjugate gradient
        natural_gradient = self.conjugate_gradient(states, policy_grad)
        
        # Compute step size using line search
        step_size = torch.sqrt(2 * self.max_kl / (natural_gradient.dot(self.hessian_vector_product(states, natural_gradient)) + 1e-8))
        
        # Update actor parameters
        old_params = torch.cat([param.data.view(-1) for param in self.actor.parameters()])
        params_update = step_size * natural_gradient
        
        # Line search
        for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            new_params = old_params + fraction * params_update
            
            # Update model parameters
            index = 0
            for param in self.actor.parameters():
                param_size = param.numel()
                param.data = new_params[index:index + param_size].view(param.shape)
                index += param_size
            
            # Check KL divergence
            kl = self.actor.get_kl(states, self.actor(states))
            
            if kl <= self.max_kl:
                self.stats["policy_loss"].append(policy_loss.item())
                self.stats["kl_div"].append(kl.item())
                break
            
            if fraction == 0.03125:
                # If we get here, revert to old parameters
                index = 0
                for param in self.actor.parameters():
                    param_size = param.numel()
                    param.data = old_params[index:index + param_size].view(param.shape)
                    index += param_size
        
        self.memory.clear()

    def save_model(self, path: str) -> None:
        """Save actor and critic models."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        """Load actor and critic models."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
