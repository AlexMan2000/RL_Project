"""
This file contains the MDP planning agent introduced in slide 17 of lecture 8
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from config import RLConfig, BoardConfig, ModelConfig

class MDPModel:
    """
    Model M of the environment that predicts (r, s') given (s, a)
    """
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.state_space_size = board_size * board_size
        self.action_space_size = 4  # up, right, down, left
        
        # Initialize transition dynamics and reward models
        # For each state-action pair, store counts and rewards
        self.transition_counts = {}  # (s,a) -> {s' -> count}
        self.reward_sums = {}       # (s,a) -> total_reward
        self.visit_counts = {}      # (s,a) -> count
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update model with new experience"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        sa_key = (state_key, action)
        
        # Initialize if not seen before
        if sa_key not in self.transition_counts:
            self.transition_counts[sa_key] = {}
            self.reward_sums[sa_key] = 0.0
            self.visit_counts[sa_key] = 0
            
        # Update transition counts
        if next_state_key not in self.transition_counts[sa_key]:
            self.transition_counts[sa_key][next_state_key] = 0
        self.transition_counts[sa_key][next_state_key] += 1
        
        # Update reward statistics
        self.reward_sums[sa_key] += reward
        self.visit_counts[sa_key] += 1
        
    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predict next state and reward for a given state-action pair"""
        state_key = self._get_state_key(state)
        sa_key = (state_key, action)
        
        # If never seen this state-action pair, return current state and 0 reward
        if sa_key not in self.transition_counts:
            return state, 0.0
            
        # Get most likely next state
        next_state_counts = self.transition_counts[sa_key]
        most_likely_next_state = max(next_state_counts.items(), key=lambda x: x[1])[0]
        
        # Get average reward
        avg_reward = self.reward_sums[sa_key] / self.visit_counts[sa_key]
        
        return self._get_state_from_key(most_likely_next_state), avg_reward
    
    def _get_state_key(self, state: np.ndarray) -> tuple:
        """Convert state array to hashable tuple"""
        return tuple(state.flatten())
    
    def _get_state_from_key(self, state_key: tuple) -> np.ndarray:
        """Convert state key back to array"""
        return np.array(state_key).reshape(self.board_size, self.board_size)

class ValueFunction:
    """
    Value function v that estimates the value of each state
    """
    def __init__(self, board_size: int, gamma: float = 0.99):
        self.board_size = board_size
        self.gamma = gamma
        self.values = {}  # state -> value
        
    def update(self, state: np.ndarray, reward: float, next_state: np.ndarray):
        """Update value function using TD learning"""
        state_key = tuple(state.flatten())
        next_state_key = tuple(next_state.flatten())
        
        # Initialize values if not seen before
        if state_key not in self.values:
            self.values[state_key] = 0.0
        if next_state_key not in self.values:
            self.values[next_state_key] = 0.0
            
        # TD update
        td_target = reward + self.gamma * self.values[next_state_key]
        td_error = td_target - self.values[state_key]
        self.values[state_key] += 0.1 * td_error  # learning rate = 0.1
        
    def get_value(self, state: np.ndarray) -> float:
        """Get value of a state"""
        state_key = tuple(state.flatten())
        return self.values.get(state_key, 0.0)

class Policy:
    """
    Policy π that maps states to action probabilities
    """
    def __init__(self, board_size: int, action_space_size: int = 4):
        self.board_size = board_size
        self.action_space_size = action_space_size
        self.policy = {}  # state -> action probabilities
        
    def update(self, state: np.ndarray, action_values: np.ndarray):
        """Update policy using softmax over action values"""
        state_key = tuple(state.flatten())
        # Softmax with temperature = 1.0
        exp_values = np.exp(action_values - np.max(action_values))
        self.policy[state_key] = exp_values / exp_values.sum()
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action according to policy"""
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            # If state not seen before, return random action
            return np.random.randint(self.action_space_size)
        return np.random.choice(self.action_space_size, p=self.policy[state_key])

class MDPAgent:
    """
    MDP Planning Agent that follows the algorithm in the image
    """
    def __init__(self, rl_config: RLConfig, board_config: BoardConfig, model_config: Optional[ModelConfig] = None):
        self.rl_config = rl_config
        self.board_config = board_config
        
        # Initialize M, v, and π arbitrarily
        self.model = MDPModel(board_config.board_size)
        self.value_function = ValueFunction(board_config.board_size, rl_config.gamma)
        self.policy = Policy(board_config.board_size)
        
        # Statistics
        self.stats = {
            "episode_rewards": [],
            "model_updates": 0,
            "value_updates": 0,
            "policy_updates": 0
        }
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action following π(s)"""
        return self.policy.select_action(state)
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Update agent following the MDP planning algorithm:
        1. Send (s,a) to model M and receive (r,s')
        2. Update v for π at s
        3. Update π given value function v
        4. s = s'
        """
        # Update model M with new experience
        self.model.update(state, action, reward, next_state)
        self.stats["model_updates"] += 1
        
        # Update value function v for π at s
        self.value_function.update(state, reward, next_state)
        self.stats["value_updates"] += 1
        
        # Update policy π using updated value function
        action_values = np.zeros(4)  # For each action
        for a in range(4):
            # Query model to predict next state and reward
            pred_next_state, pred_reward = self.model.predict(state, a)
            # Compute action value using current value function
            action_values[a] = pred_reward + self.rl_config.gamma * self.value_function.get_value(pred_next_state)
        
        # Update policy with new action values
        self.policy.update(state, action_values)
        self.stats["policy_updates"] += 1
        
        # Note: s = s' is handled by the training loop
