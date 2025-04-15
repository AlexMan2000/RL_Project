import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import random

class ValueBasedAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize Q-network and target network
        self.q_network = QNetwork(
            input_dim=16,  # 4x4 board
            hidden_dim=config.hidden_dim,
            output_dim=4  # 4 possible actions
        ).to(self.device)
        
        self.target_network = QNetwork(
            input_dim=16,
            hidden_dim=config.hidden_dim,
            output_dim=4
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=config.memory_size)
        
        # Training statistics
        self.stats = {
            "q_loss": [],
            "epsilon": [],
            "episode_rewards": []
        }
        
        self.epsilon = config.epsilon
        self.steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action
        
        state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update the Q-network using experience replay."""
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
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.config.gamma * (1 - dones) * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_min,
                         self.epsilon * self.config.epsilon_decay)
        
        # Store statistics
        self.stats["q_loss"].append(loss.item())
        self.stats["epsilon"].append(self.epsilon)

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state) 