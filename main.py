import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, List
from enum import Enum
# from value_based import ValueBasedAgent   # This is a more enhanced version of the ValueBasedAgent
from value_based.mlp_model import MLPValueBasedAgent
from value_based.cnn_model import CNNValueBasedAgent
from model_based import ModelBasedAgent
from policy_based.pgmc import PGMCAgent
from policy_based.actor_critic import ActorCriticAgent
from policy_based.trpo import TRPOAgent
from policy_based.ppo import PPOAgent
import random
from config import RLConfig, BoardConfig, RLMethod, ModelConfig
import json
import os
from datetime import datetime


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (RLConfig, BoardConfig)):
            return {k: v for k, v in vars(obj).items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class Game2048Env(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, board_size: int = 4, init_board: Optional[np.ndarray] = None, render_mode: str = None):
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
        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {}

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

def create_agent(
        rl_config: RLConfig, 
        board_config: BoardConfig, 
        model_config: Optional[ModelConfig] = None, 
        policy_based_model: Optional[str] = None, 
        value_based_model: Optional[str] = None):
    """Create and return the appropriate RL agent based on the configuration."""
    if isinstance(rl_config.method, str):
        rl_config.method = RLMethod(rl_config.method)
    
    if rl_config.method == RLMethod.MODEL_BASED:
        print("Model based agent")
        raise NotImplementedError("Model based agent is not implemented")
    elif rl_config.method == RLMethod.VALUE_BASED:
        print("Value based agent")
        if value_based_model == "cnn":
            return CNNValueBasedAgent(rl_config, board_config, model_config)
        elif value_based_model == "mlp":
            return MLPValueBasedAgent(rl_config, board_config, model_config)
        else:
            raise ValueError(f"Unknown value based model: {value_based_model}")
    elif rl_config.method == RLMethod.POLICY_BASED:
        print("Policy based agent")
        if policy_based_model == "ppo":
            return PPOAgent(rl_config, board_config, model_config)
        elif policy_based_model == "trpo":
            return TRPOAgent(rl_config, board_config, model_config)
        elif policy_based_model == "pgmc":
            return PGMCAgent(rl_config, board_config, model_config)
        elif policy_based_model == "actor_critic":
            return ActorCriticAgent(rl_config, board_config, model_config)
        else:
            raise ValueError(f"Unknown policy based model: {policy_based_model}")
    else:
        raise ValueError(f"Unknown RL method: {rl_config.method}")

def train(
    rl_config: RLConfig,
    board_config: BoardConfig,
    model_config: Optional[ModelConfig] = None,
    policy_based_model: Optional[str] = None,
    value_based_model: Optional[str] = None,
    env: Optional[Game2048Env] = None,
    agent: Optional[Any] = None,
    render: bool = False,
    save_dir: str = "results",
    save_every: int = 10,
    no_save: bool = False
) -> Dict[str, Any]:
    """
    Train the RL agent on the 2048 environment.
    
    Args:
        config: RL configuration
        env: Optional environment instance
        agent: Optional agent instance
        render: Whether to render the environment during training
        save_dir: Directory to save training results
    
    Returns:
        Dictionary containing training statistics
    """
    if env is None:
        env = create_env(board_config)
    if agent is None:
        agent = create_agent(rl_config, board_config, model_config, policy_based_model, value_based_model)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "best_score": 0,
        "config": {
            "rl_config": rl_config,
            "board_config": board_config
        }
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
        
        if episode % save_every == 0 and not no_save:
            print(f"Episode {episode}, Reward: {episode_reward}, Best Score: {stats['best_score']}")
            
            # Save intermediate results every 10 episodes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"training_stats_{timestamp}.json")
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=4, cls=EnhancedJSONEncoder)
    
    # Save final results
    if not no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = os.path.join(save_dir, f"final_training_stats_{timestamp}.json")
        with open(final_save_path, 'w') as f:
            json.dump(stats, f, indent=4, cls=EnhancedJSONEncoder)
        print(f"Results saved in directory: {save_dir}")

    
    return stats

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from dictionary."""
    return ModelConfig(**config_dict)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent on 2048 game")
    parser.add_argument("--method", type=str, default="value_based",
                       choices=[e.value for e in RLMethod],
                       help="RL method to use")
    parser.add_argument("--policy_based_model", type=str, default="pgmc",
                       choices=["pgmc", "actor_critic", "trpo", "ppo"],
                       help="Policy based model to use")
    parser.add_argument("--value_based_model", type=str, default="mlp",
                       choices=["mlp", "cnn"],
                       help="Value based model to use")
    parser.add_argument("--board-size", type=int, default=4,
                       help="Board size")
    parser.add_argument("--num-episodes", type=int, default=1000,
                       help="Number of episodes to train")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during training")
    parser.add_argument("--rl-config-file", type=str,
                       help="Path to RL configuration JSON file")
    parser.add_argument("--board-config-file", type=str,
                       help="Path to board configuration JSON file")
    parser.add_argument("--model-config-file", type=str,
                       help="Path to model architecture configuration JSON file")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save training results")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save training results every N episodes")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save training results")
    
    args = parser.parse_args()

    # Load configurations from files if provided
    rl_config_dict = {}
    board_config_dict = {}
    model_config_dict = {}
    
    if args.rl_config_file:
        rl_config_dict = load_config(args.rl_config_file)
    else:
        rl_config_dict = {
            "method": args.method,
            "num_episodes": args.num_episodes,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma
        }

    if args.board_config_file:
        board_config_dict = load_config(args.board_config_file)
    else:
        board_config_dict = {
            "board_size": args.board_size
        }

    if args.model_config_file:
        model_config_dict = load_config(args.model_config_file)
        model_config = create_model_config(model_config_dict)
    else:
        model_config = None

    # Create configs
    rl_config = RLConfig(**rl_config_dict)
    board_config = BoardConfig(**board_config_dict)

    stats = train(
        rl_config, 
        board_config,
        model_config,
        policy_based_model=args.policy_based_model,
        value_based_model=args.value_based_model,
        render=args.render, 
        save_dir=args.save_dir, 
        save_every=args.save_every, 
        no_save=args.no_save)
    print(f"Training completed. Best score: {stats['best_score']}")
    
if __name__ == "__main__":
    main() 