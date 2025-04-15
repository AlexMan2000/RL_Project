import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import random

class ModelBasedAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize dynamics model
        self.dynamics_model = DynamicsModel(
            input_dim=16,  # 4x4 board
            hidden_dim=config.hidden_dim,
            output_dim=16  # Next state prediction
        ).to(self.device)
        
        # Initialize value model
        self.value_model = ValueModel(
            input_dim=16,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.dynamics_optimizer = optim.Adam(
            self.dynamics_model.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_model.parameters(),
            lr=config.learning_rate
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=config.memory_size)
        
        # Training statistics
        self.stats = {
            "dynamics_loss": [],
            "value_loss": [],
            "planning_steps": []
        }

    def select_action(self, state: np.ndarray) -> int:
        """Select action using model-based planning."""
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        
        # Perform planning using the learned model
        best_action = self._plan(state_tensor)
        return best_action

    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update the models using the collected experience."""
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update dynamics model
        self._update_dynamics_model(states, actions, next_states)
        
        # Update value model
        self._update_value_model(states, rewards, next_states, dones)

    def _plan(self, state: torch.Tensor) -> int:
        """Plan actions using the learned model."""
        best_value = float('-inf')
        best_action = 0
        
        for action in range(4):  # 4 possible actions
            # Simulate next state using dynamics model
            next_state_pred = self.dynamics_model(state, action)
            
            # Predict value of next state
            value = self.value_model(next_state_pred)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

    def _update_dynamics_model(self, states: torch.Tensor, actions: torch.Tensor,
                             next_states: torch.Tensor):
        """Update the dynamics model using supervised learning."""
        self.dynamics_optimizer.zero_grad()
        
        # Predict next states
        next_states_pred = self.dynamics_model(states, actions)
        
        # Compute loss
        loss = nn.MSELoss()(next_states_pred, next_states)
        
        # Backpropagate
        loss.backward()
        self.dynamics_optimizer.step()
        
        self.stats["dynamics_loss"].append(loss.item())

    def _update_value_model(self, states: torch.Tensor, rewards: torch.Tensor,
                          next_states: torch.Tensor, dones: torch.Tensor):
        """Update the value model using TD learning."""
        self.value_optimizer.zero_grad()
        
        # Predict current and next state values
        current_values = self.value_model(states)
        next_values = self.value_model(next_states)
        
        # Compute TD targets
        targets = rewards + self.config.gamma * (1 - dones) * next_values
        
        # Compute loss
        loss = nn.MSELoss()(current_values, targets.detach())
        
        # Backpropagate
        loss.backward()
        self.value_optimizer.step()
        
        self.stats["value_loss"].append(loss.item())

class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Convert action to one-hot encoding
        action_one_hot = torch.zeros(4).to(state.device)
        action_one_hot[action] = 1
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.network(x)

class ValueModel(nn.Module):
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