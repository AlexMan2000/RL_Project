import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, List
from enum import Enum
from value_based.mlp_model import MLPValueBasedAgent
from value_based.cnn_model import CNNValueBasedAgent
from model_based.mdp import MDPAgent
from policy_based.pgmc import PGMCAgent
from policy_based.actor_critic import ActorCriticAgent
from policy_based.trpo import TRPOAgent
from policy_based.ppo import PPOAgent
import random
from config import RLConfig, BoardConfig, RLMethod, ModelConfig
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


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


    def __init__(self, 
        board_size: int = 4, 
        init_board: Optional[np.ndarray] = None, 
        render_mode: str = None, 
        reward_config: Optional[dict] = None
    ):
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
        # Reward config
        self.reward_config = reward_config or {
            'merge_bonus': False,
            'fullness_penalty': False,
            'smoothness_bonus': False,
            'corner_bonus': False,
            'no_progress_penalty': False
        }

        print(self.reward_config)


    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {}

    def _random_init_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        # Randomly choose two distinct empty cells, first is (x1, y1), second is (x2, y2)
        [first, second] = self._sample_empty_positions(2, board)

        # Place either a 2 (90% chance) or 4 (10% chance)
        board[first] = 2 if random.random() < 0.9 else 4
        board[second] = 2 if random.random() < 0.9 else 4

        return board
    

    def _sample_empty_positions(self, k=1, board=None):
        if board is None:
            board = self.board
        empty_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if board[i, j] == 0]
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
        board_before_action = self.board.copy()
        total_score, total_reward = self._move(action)
        board_after_action = self.board.copy()
        is_valid_move = not np.array_equal(board_before_action, board_after_action)
        done = self._is_game_over()
        if not done and is_valid_move:
            self._add_new_tile()
        return self.board, total_score, total_reward, done, False, {}

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
        total_score = 0
        total_reward = 0
        merged_this_move = False
        largest_merge = 0
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
                    merged_value = non_zero[j] * 2
                    merged.append(merged_value)
                    total_score += merged_value
                    merged_this_move = True
                    if merged_value > largest_merge:
                        largest_merge = merged_value
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


        # --- Reward Shaping ---
        total_reward += total_score
        # 1. Merge bonus (encourage merging high-value tiles)
        if self.reward_config.get('merge_bonus', True):
            # Extra bonus for large merges
            if largest_merge >= 32:
                total_reward += (np.log2(largest_merge) - 4) * 10  # e.g., 32->10, 64->20, etc.
        # 2. Fullness penalty (encourage survival)
        if self.reward_config.get('fullness_penalty', True):
            empty_cells = np.sum(new_board == 0)
            total_reward += empty_cells * 0.5  # more empty = better
        # 3. Smoothness bonus (encourage similar tiles adjacent)
        if self.reward_config.get('smoothness_bonus', True):
            smoothness = 0
            for i in range(self.board_size):
                for j in range(self.board_size - 1):
                    if new_board[i, j] == new_board[i, j + 1] and new_board[i, j] != 0:
                        smoothness += 1
                    if new_board[j, i] == new_board[j + 1, i] and new_board[j, i] != 0:
                        smoothness += 1
            total_reward += smoothness
        # 4. Corner bonus (encourage largest tile in a corner)
        if self.reward_config.get('corner_bonus', True):
            max_tile = np.max(new_board)
            corners = [new_board[0, 0], new_board[0, -1], new_board[-1, 0], new_board[-1, -1]]
            if max_tile in corners:
                total_reward += 10
        # 5. No progress penalty (discourage no merges or no displacement)
        if self.reward_config.get('no_progress_penalty', True):
            if not merged_this_move and np.array_equal(self.board, new_board):
                total_reward -= 100
        return total_score, total_reward

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

        # If the board is full, check if any mergeable horizontal or vertical pair exists
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

    def display_board(self):
        for row in self.board:
            print("\t".join(f"{int(val):4d}" if val != 0 else "   ." for val in row))
        print()

def create_env(config: BoardConfig, reward_config: Optional[dict] = None) -> Game2048Env:
    """Create and return the 2048 environment."""
    return Game2048Env(config.board_size, config.init_board, reward_config=reward_config)

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
        return MDPAgent(rl_config, board_config, model_config)
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
    save_dir: str = "trained_models",
    log_every: int = 10,
    no_save: bool = False,
    reward_config: Optional[dict] = None
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
        env = create_env(board_config, reward_config)
    if agent is None:
        agent = create_agent(rl_config, board_config, model_config, policy_based_model, value_based_model)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Record start time
    start_time = datetime.now()
    
    stats = {
        "episode_rewards": [],
        "episode_scores": [],
        "episode_lengths": [],
        "best_score": 0,
        "best_board": None,  # Store the board state of the best score
        "avg_scores": [],  # Track average scores per log_every episodes
        "config": {
            "rl_config": rl_config,
            "board_config": board_config
        }
    }

    algorithm_name = rl_config.method.value
    model_name = value_based_model if rl_config.method == RLMethod.VALUE_BASED else policy_based_model
    subfolder = os.path.join(save_dir, f"{algorithm_name}_{model_name}")
    os.makedirs(subfolder, exist_ok=True)


    stats_log_every = {
        "episode_rewards": [],
        "episode_scores": [],
        "episode_lengths": [],
        "best_score": 0
    }
    
    # See if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for episode in range(1, rl_config.num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_score = 0
        done = False

        # for step in range(rl_config.max_steps):
        step = 0
        if rl_config.max_steps is not None:
            while step < rl_config.max_steps:
                if render:
                    env.render()
                
                action = agent.select_action(state)
                next_state, step_score, step_reward, done, _, _ = env.step(action)
                agent.update(state, action, step_reward, next_state, done)
                
                state = next_state
                episode_score += step_score
                episode_reward += step_reward
                
                if done:
                    break
                step += 1
        else:    
            print("Training until game over")
            while not done:
                if render:
                    env.render()
                
                action = agent.select_action(state)
                next_state, step_score, step_reward, done, _, _ = env.step(action)
                agent.update(state, action, step_reward, next_state, done)
                
                state = next_state
                episode_score += step_score
                episode_reward += step_reward
                
                if done:
                    break
                step += 1

        
        stats["episode_rewards"].append(episode_reward)
        stats["episode_scores"].append(episode_score)
        stats["episode_lengths"].append(step + 1)
        
        # Update best score and store the board state if it's a new best
        if episode_score > stats["best_score"]:
            stats["best_score"] = episode_score
            stats["best_board"] = state.copy()  # Store the final board state
        
        stats_log_every["episode_rewards"].append(episode_reward)
        stats_log_every["episode_scores"].append(episode_score)
        stats_log_every["episode_lengths"].append(step + 1)
        stats_log_every["best_score"] = max(stats_log_every["best_score"], episode_score)
        
        if episode % log_every == 0:
            avg_score = sum(stats_log_every['episode_scores']) / log_every
            stats["avg_scores"].append(avg_score)  # Store the average score

            print(f"Episode {episode - log_every} ~ {episode}, avg score: {avg_score}, best score ever: {stats['best_score']}")
            # Save intermediate results every log_every episodes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_log_every = {
                "episode_rewards": [],
                "episode_scores": [],
                "episode_lengths": [],
                "best_score": 0
            }
            if not no_save:
                save_path = os.path.join(subfolder, f"checkpoint_{episode}_{timestamp}.pt")
                model_state = {}

                # Value-based: DQN, etc.
                if hasattr(agent, 'q_network'):
                    model_state['q_network'] = agent.q_network.state_dict()
                if hasattr(agent, 'target_network'):
                    model_state['target_network'] = agent.target_network.state_dict()

                # Policy-based: PPO, Actor-Critic, TRPO
                if hasattr(agent, 'actor'):
                    model_state['actor'] = agent.actor.state_dict()
                if hasattr(agent, 'critic'):
                    model_state['critic'] = agent.critic.state_dict()
                if hasattr(agent, 'policy_network'):
                    model_state['policy_network'] = agent.policy_network.state_dict()

                # Model-based: MDPAgent (example: save model, value_function, policy if they have state_dict)
                if hasattr(agent, 'model') and hasattr(agent.model, 'state_dict'):
                    model_state['model'] = agent.model.state_dict()
                if hasattr(agent, 'value_function') and hasattr(agent.value_function, 'state_dict'):
                    model_state['value_function'] = agent.value_function.state_dict()
                if hasattr(agent, 'policy') and hasattr(agent.policy, 'state_dict'):
                    model_state['policy'] = agent.policy.state_dict()
                if model_state:
                    torch.save(model_state, save_path)
                    print(f"Model parameters saved to: {save_path}")
                else:
                    print("No model parameters found to save for this agent.")
    
    # Save final results
    if not no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = os.path.join(subfolder, f"final_training_stats_{timestamp}.json")
        with open(final_save_path, 'w') as f:
            json.dump(stats, f, indent=4, cls=EnhancedJSONEncoder)
        print(f"Results saved in directory: {subfolder}")

    # Calculate training duration
    end_time = datetime.now()
    training_duration = end_time - start_time
    hours = training_duration.seconds // 3600
    minutes = (training_duration.seconds % 3600) // 60
    seconds = training_duration.seconds % 60
    
    # Plot training progress with both raw scores and moving average
    plt.figure(figsize=(12, 6))
    
    # Plot raw scores with lower alpha for better visibility
    plt.plot(range(len(stats["episode_scores"])), stats["episode_scores"], 
             label='Episode Score', alpha=0.3, color='blue')
    
    # Plot average scores
    avg_x = range(log_every-1, len(stats["episode_scores"]), log_every)
    plt.plot(avg_x, stats["avg_scores"], 
             label=f'Average Score (per {log_every} episodes)', 
             color='red', linewidth=2)
    
    # Add horizontal line for overall average
    overall_avg = np.mean(stats["episode_scores"])
    plt.axhline(y=overall_avg, color='green', linestyle='--', 
                label=f'Overall Average Score: {overall_avg:.2f}')
    
    # Mark highest score episode with a red star
    max_score_idx = np.argmax(stats["episode_scores"])
    max_score = stats["episode_scores"][max_score_idx]
    plt.plot(max_score_idx, max_score, 'r*', markersize=15,
             label=f'Highest Score: {max_score:.2f} (Episode {max_score_idx+1})')
    
    plt.xlabel('Episode')
    plt.ylabel('Score/Avg Score')
    plt.title(f'Training Progress Statistics (Duration: {hours:02d}:{minutes:02d}:{seconds:02d})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot in the same subfolder as checkpoints
    plot_path = os.path.join(subfolder, f'training_progress_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training progress plot saved to: {plot_path}")
    
    # Display the final board state of the best score
    if stats["best_board"] is not None:
        print("\nFinal Board State of Highest Score Episode:")
        print("Score:", stats["best_score"])
        print("Board:")
        for row in stats["best_board"]:
            print("\t".join(f"{int(val):4d}" if val != 0 else "   ." for val in row))
        print()
    
    return stats

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from dictionary."""
    return ModelConfig(**config_dict)

def load_reward_config(config_file: str) -> dict:
    if not os.path.exists(config_file):
        # Default config if file does not exist
        return {
            'merge_bonus': True,
            'fullness_penalty': True,
            'smoothness_bonus': True,
            'corner_bonus': True,
            'no_progress_penalty': True
        }
    with open(config_file, 'r') as f:
        return json.load(f)

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
    parser.add_argument("--max-steps-per-episode", type=int, default=1000,
                       help="Maximum number of steps per episode")
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
    parser.add_argument("--save-dir", type=str, default="trained_models",
                       help="Directory to save training results")
    parser.add_argument("--log-every", type=int, default=10,
                       help="Log training results every N episodes")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save training results")
    parser.add_argument("--reward-config-file", type=str, default="config_files/reward_config.json",
                       help="Path to reward configuration JSON file")
    
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
            "max_steps": args.max_steps_per_episode,
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

    reward_config = load_reward_config(args.reward_config_file)

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
        log_every=args.log_every, 
        no_save=args.no_save,
        reward_config=reward_config)
    print(f"Training completed. Best score: {stats['best_score']}")
    
if __name__ == "__main__":
    main() 