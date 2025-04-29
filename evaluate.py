import argparse
import torch
import numpy as np
import os
import json
from config import RLConfig, BoardConfig, RLMethod, ModelConfig
from train import Game2048Env, create_env, create_agent, load_config, create_model_config, load_reward_config

# Import agent classes
from value_based.mlp_model import MLPValueBasedAgent
from value_based.cnn_model import CNNValueBasedAgent
from model_based.mdp import MDPAgent
from policy_based.pgmc import PGMCAgent
from policy_based.actor_critic import ActorCriticAgent
from policy_based.trpo import TRPOAgent
from policy_based.ppo import PPOAgent

def load_agent_state(agent, checkpoint):
    # Value-based
    if hasattr(agent, 'q_network') and 'q_network' in checkpoint:
        agent.q_network.load_state_dict(checkpoint['q_network'])
    if hasattr(agent, 'target_network') and 'target_network' in checkpoint:
        agent.target_network.load_state_dict(checkpoint['target_network'])
    # Policy-based
    if hasattr(agent, 'actor') and 'actor' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor'])
    if hasattr(agent, 'critic') and 'critic' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic'])
    if hasattr(agent, 'policy_network') and 'policy_network' in checkpoint:
        agent.policy_network.load_state_dict(checkpoint['policy_network'])
    # Model-based
    if hasattr(agent, 'model') and 'model' in checkpoint:
        agent.model.load_state_dict(checkpoint['model'])
    if hasattr(agent, 'value_function') and 'value_function' in checkpoint:
        agent.value_function.load_state_dict(checkpoint['value_function'])
    if hasattr(agent, 'policy') and 'policy' in checkpoint:
        agent.policy.load_state_dict(checkpoint['policy'])

def evaluate_agent(agent, env, num_episodes=100, render=False):
    scores = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(state)
            next_state, step_score, _, done, _, _ = env.step(action)
            state = next_state
            score += step_score
            if render:
                env.display_board()
        scores.append(score)
        print(f"Episode {ep+1}: Score = {score}")
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Std Score: {std_score:.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    return scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained 2048 RL agent.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--algorithm', type=str, required=True, choices=[e.value for e in RLMethod], help='Algorithm type (value_based, policy_based, model_based)')
    parser.add_argument('--policy_based_model', type=str, default=None, choices=["pgmc", "actor_critic", "trpo", "ppo"], help='Policy based model type')
    parser.add_argument('--value_based_model', type=str, default=None, choices=["mlp", "cnn"], help='Value based model type')
    parser.add_argument('--rl-config-file', type=str, help='Path to RL config JSON')
    parser.add_argument('--board-config-file', type=str, help='Path to board config JSON')
    parser.add_argument('--model-config-file', type=str, help='Path to model config JSON')
    parser.add_argument('--reward-config-file', type=str, default="config_files/reward_config.json", help='Path to reward config JSON')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the board during evaluation')
    args = parser.parse_args()

    # Load configs
    rl_config_dict = load_config(args.rl_config_file) if args.rl_config_file else {}
    board_config_dict = load_config(args.board_config_file) if args.board_config_file else {}
    model_config_dict = load_config(args.model_config_file) if args.model_config_file else None
    model_config = create_model_config(model_config_dict) if model_config_dict else None
    reward_config = load_reward_config(args.reward_config_file)

    # Set up RLConfig and BoardConfig
    rl_config = RLConfig(**rl_config_dict) if rl_config_dict else RLConfig(method=RLMethod(args.algorithm))
    board_config = BoardConfig(**board_config_dict) if board_config_dict else BoardConfig()

    # Create environment
    env = create_env(board_config, reward_config)

    # Create agent
    agent = create_agent(
        rl_config,
        board_config,
        model_config,
        policy_based_model=args.policy_based_model,
        value_based_model=args.value_based_model
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    load_agent_state(agent, checkpoint)

    # Evaluate
    evaluate_agent(agent, env, num_episodes=args.num_episodes, render=args.render)

if __name__ == "__main__":
    main()
