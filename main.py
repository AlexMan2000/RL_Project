import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Dict, Any, Union
from enum import Enum
from value_based import ValueBasedAgent
from model_based import ModelBasedAgent
from policy_based import PolicyBasedAgent
import random

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

class Game2048Env(gym.Env):
    def __init__(self, board_size: int = 4, init_board: Optional[np.ndarray] = None):
        super().__init__()
        self.board_size = board_size
        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = gym.spaces.Box(
            low=0, high=2**16, shape=(self.board_size, self.board_size), dtype=np.int32
        )
        if init_board is None:
            self.board = self._random_init_board()
        else:
            self.board = init_board
        self.reset()

    def _random_init_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        # Randomly choose two distinct empty cells, first is (x1, y1), second is (x2, y2)
        [first, second] = self._sample_empty_positions(2)

        # Place either a 2 (90% chance) or 4 (10% chance)
        board[first] = 2 if random.random() < 0.9 else 4
        board[second] = 2 if random.random() < 0.9 else 4

        return board
    

    def _sample_empty_positions(self, k=1):
        empty_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
        if not empty_positions:
            return
        return random.sample(empty_positions, k)


    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.board = self._random_init_board()
        self._add_new_tile()
        self._add_new_tile()
        return self.board, {}

    def step(self, action: int):
        reward = self._move(action)
        done = self._is_game_over()
        if not done:
            self._add_new_tile()
        return self.board, reward, done, False, {}

    def _move(self, action: int):
        # Implement 2048 game logic here,
        # Rotate board so we can treat all moves as 'left'
        # 0: up, 1: right, 2: down, 3: left
        if action == 0:
            self.board = np.rot90(self.board, k=1)
        elif action == 1:
            self.board = np.rot90(self.board, k=2)
        elif action == 2:
            self.board = np.rot90(self.board, k=3)
        elif action == 3:
            pass

        new_board = np.zeros_like(self.board)
        total_reward = 0
        for i in range(self.board_size):
            row = self.board[i]
            # Step 1: remove zeros (slide left)
            non_zero = row[row != 0]
            
            # Step 2: merge
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged.append(non_zero[j] * 2)
                    total_reward += non_zero[j] * 2
                    skip = True
                else:
                    merged.append(non_zero[j])
            
            # Step 3: pad with zeros to the right
            merged += [0] * (self.board_size - len(merged))
            new_board[i] = merged

        # Restore board orientation
        if action == 0:
            self.board = np.rot90(new_board, k=3)
        elif action == 1:
            self.board = np.rot90(new_board, k=2)
        elif action == 2:
            self.board = np.rot90(new_board, k=1)
        elif action == 3:
            self.board = new_board

        return total_reward

    def _add_new_tile(self):
        # Add a new tile (2 or 4) to a random empty position

        i, j = self._sample_empty_positions(1)[0]
        self.board[i, j] = 2 if random.random() < 0.9 else 4
            

    def _is_game_over(self) -> bool:
        """
        Check:
        1. If any tile is 0 → not over
        2. If any mergeable horizontal or vertical pair exists → not over
        3. Otherwise, game is over.
        """
        # Check if the game is over
         # If there's any empty tile
        if np.any(self.board == 0):
            return False

        # Check for horizontal merges
        for i in range(self.board_size):
            for j in range(self.board_size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check for vertical merges
        for i in range(self.board_size - 1):
            for j in range(self.board_size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        return True

def create_env(config: BoardConfig) -> Game2048Env:
    """Create and return the 2048 environment."""
    return Game2048Env(config.board_size, config.init_board)

def create_agent(config: RLConfig):
    """Create and return the appropriate RL agent based on the configuration."""
    if config.method == RLMethod.MODEL_BASED:
        return ModelBasedAgent(config)
    elif config.method == RLMethod.VALUE_BASED:
        return ValueBasedAgent(config)
    elif config.method == RLMethod.POLICY_BASED:
        return PolicyBasedAgent(config)
    else:
        raise ValueError(f"Unknown RL method: {config.method}")

def train(
    rl_config: RLConfig,
    board_config: BoardConfig,
    env: Optional[Game2048Env] = None,
    agent: Optional[Any] = None,
    render: bool = False
) -> Dict[str, Any]:
    """
    Train the RL agent on the 2048 environment.
    
    Args:
        config: RL configuration
        env: Optional environment instance
        agent: Optional agent instance
        render: Whether to render the environment during training
    
    Returns:
        Dictionary containing training statistics
    """
    if env is None:
        env = create_env(board_config)
    if agent is None:
        agent = create_agent(rl_config)
    
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "best_score": 0
    }
    
    for episode in range(rl_config.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(rl_config.max_steps):
            if render:
                env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(step + 1)
        stats["best_score"] = max(stats["best_score"], episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Best Score: {stats['best_score']}")
    
    return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent on 2048 game")
    parser.add_argument("--method", type=str, default="value_based",
                       choices=["model_based", "value_based", "policy_based"],
                       help="RL method to use")
    parser.add_argument("--num_episodes", type=int, default=1000,
                       help="Number of episodes to train")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during training")
    
    args = parser.parse_args()
    

    # Config for RL Algorithm
    rl_config = RLConfig(
        method=RLMethod(args.method),
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )

    # Config for Board
    board_config = BoardConfig(board_size=4)


    stats = train(rl_config, board_config, render=args.render)
    print(f"Training completed. Best score: {stats['best_score']}")

if __name__ == "__main__":
    main() 