import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Dict, Any, Union
from enum import Enum

class RLMethod(Enum):
    MODEL_BASED = "model_based"
    VALUE_BASED = "value_based"
    POLICY_BASED = "policy_based"

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
    def __init__(self):
        super().__init__()
        self.board_size = 4
        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = gym.spaces.Box(
            low=0, high=2**16, shape=(self.board_size, self.board_size), dtype=np.int32
        )
        self.reset()

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self._add_new_tile()
        self._add_new_tile()
        return self.board, {}

    def step(self, action: int):
        old_board = self.board.copy()
        self._move(action)
        reward = self._calculate_reward(old_board)
        done = self._is_game_over()
        if not done:
            self._add_new_tile()
        return self.board, reward, done, False, {}

    def _move(self, action: int):
        # Implement 2048 game logic here
        pass

    def _add_new_tile(self):
        # Add a new tile (2 or 4) to a random empty position
        pass

    def _calculate_reward(self, old_board: np.ndarray) -> float:
        # Calculate reward based on the game state
        pass

    def _is_game_over(self) -> bool:
        # Check if the game is over
        pass

def create_env(config: RLConfig) -> Game2048Env:
    """Create and return the 2048 environment."""
    return Game2048Env()

def create_agent(config: RLConfig):
    """Create and return the appropriate RL agent based on the configuration."""
    if config.method == RLMethod.MODEL_BASED:
        from agents.model_based import ModelBasedAgent
        return ModelBasedAgent(config)
    elif config.method == RLMethod.VALUE_BASED:
        from agents.value_based import ValueBasedAgent
        return ValueBasedAgent(config)
    elif config.method == RLMethod.POLICY_BASED:
        from agents.policy_based import PolicyBasedAgent
        return PolicyBasedAgent(config)
    else:
        raise ValueError(f"Unknown RL method: {config.method}")

def train(
    config: RLConfig,
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
        env = create_env(config)
    if agent is None:
        agent = create_agent(config)
    
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "best_score": 0
    }
    
    for episode in range(config.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(config.max_steps):
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
    
    config = RLConfig(
        method=RLMethod(args.method),
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    stats = train(config, render=args.render)
    print(f"Training completed. Best score: {stats['best_score']}")

if __name__ == "__main__":
    main() 