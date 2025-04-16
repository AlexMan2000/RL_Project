from enum import Enum
from typing import Optional
import numpy as np
import torch

class RLMethod(Enum):
    MODEL_BASED = "model_based"
    VALUE_BASED = "value_based"
    POLICY_BASED = "policy_based"
    



class BoardConfig:
    def __init__(self, board_size: int = 4, init_board: Optional[np.ndarray] = None):
        self.board_size = board_size
        self.init_board = init_board


class RLConfig:
    def __init__(
        self,
        method: RLMethod = RLMethod.VALUE_BASED,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update: int = 1000,
        hidden_dim: int = 128,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.method = method
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.hidden_dim = hidden_dim
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.device = device