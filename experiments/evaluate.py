import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import time
from datetime import datetime
import json

from train import train, RLConfig, BoardConfig
from config import RLMethod

def run_experiment(
    method: str,
    model_type: str,
    num_episodes: int,
    num_runs: int = 5,
    save_dir: str = "experiment_results"
) -> Dict:
    """Run experiment for a specific method and model multiple times"""
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "best_scores": [],
        "highest_tiles": [],
        "training_times": [],
        "updates_per_episode": []
    }
    
    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs} for {method}-{model_type}")
        
        # Configure the run
        rl_config = RLConfig(
            method=method,
            num_episodes=num_episodes,
            learning_rate=0.001,
            gamma=0.99
        )
        board_config = BoardConfig(board_size=4)
        
        # Time the training
        start_time = time.time()
        stats = train(
            rl_config=rl_config,
            board_config=board_config,
            policy_based_model=model_type if method == "policy_based" else None,
            value_based_model=model_type if method == "value_based" else None,
            save_dir=os.path.join(save_dir, f"{method}_{model_type}_run{run}"),
            no_save=True
        )
        training_time = time.time() - start_time
        
        # Store results
        results["episode_rewards"].append(stats["episode_rewards"])
        results["episode_lengths"].append(stats["episode_lengths"])
        results["best_scores"].append(stats["best_score"])
        results["training_times"].append(training_time)
        
    return results

def plot_learning_curves(results: Dict[str, Dict], save_dir: str):
    """Plot learning curves comparing different methods"""
    plt.figure(figsize=(15, 10))
    
    # Plot average rewards over episodes
    plt.subplot(2, 2, 1)
    for method, data in results.items():
        rewards = np.array(data["episode_rewards"])
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        episodes = np.arange(len(mean_rewards))
        plt.plot(episodes, mean_rewards, label=method)
        plt.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        alpha=0.2)
    plt.title("Learning Curves")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    
    # Plot best scores distribution
    plt.subplot(2, 2, 2)
    data = []
    labels = []
    for method, result in results.items():
        data.append(result["best_scores"])
        labels.extend([method] * len(result["best_scores"]))
    plt.boxplot(data, labels=list(results.keys()))
    plt.title("Distribution of Best Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    
    # Plot training times
    plt.subplot(2, 2, 3)
    times = [np.mean(result["training_times"]) for result in results.values()]
    std_times = [np.std(result["training_times"]) for result in results.values()]
    plt.bar(list(results.keys()), times, yerr=std_times)
    plt.title("Average Training Time")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    
    # Plot episode lengths
    plt.subplot(2, 2, 4)
    for method, data in results.items():
        lengths = np.array(data["episode_lengths"])
        mean_lengths = np.mean(lengths, axis=0)
        std_lengths = np.std(lengths, axis=0)
        episodes = np.arange(len(mean_lengths))
        plt.plot(episodes, mean_lengths, label=method)
        plt.fill_between(episodes, 
                        mean_lengths - std_lengths, 
                        mean_lengths + std_lengths, 
                        alpha=0.2)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"))
    plt.close()

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table of different metrics"""
    data = []
    for method, result in results.items():
        row = {
            "Method": method,
            "Avg Best Score": np.mean(result["best_scores"]),
            "Std Best Score": np.std(result["best_scores"]),
            "Avg Training Time": np.mean(result["training_times"]),
            "Avg Episode Length": np.mean([np.mean(x) for x in result["episode_lengths"]]),
            "Final Avg Reward": np.mean([x[-1] for x in result["episode_rewards"]])
        }
        data.append(row)
    return pd.DataFrame(data)

def main():
    # Configuration
    num_episodes = 1000
    num_runs = 5
    base_save_dir = "experiment_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define experiments
    experiments = [
        ("value_based", "mlp"),
        ("value_based", "cnn"),
        ("policy_based", "pgmc"),
        ("policy_based", "actor_critic"),
        ("policy_based", "trpo"),
        ("policy_based", "ppo"),
        ("model_based", None)
    ]
    
    # Run experiments
    results = {}
    for method, model_type in experiments:
        model_name = f"{method}_{model_type if model_type else 'mdp'}"
        results[model_name] = run_experiment(
            method=method,
            model_type=model_type,
            num_episodes=num_episodes,
            num_runs=num_runs,
            save_dir=save_dir
        )
    
    # Generate visualizations and analysis
    plot_learning_curves(results, save_dir)
    comparison_table = create_comparison_table(results)
    
    # Save results
    comparison_table.to_csv(os.path.join(save_dir, "comparison_table.csv"))
    with open(os.path.join(save_dir, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print("\nResults Summary:")
    print(comparison_table)
    print(f"\nDetailed results saved to: {save_dir}")

if __name__ == "__main__":
    main() 