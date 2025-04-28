import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame
import time
from typing import Optional, Tuple

from train import Game2048Env, create_agent, RLConfig, BoardConfig

class GameVisualizer:  
    """Visualize the 2048 game and agent's decision making"""
    
    COLORS = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46)
    }
    
    def __init__(self, board_size: int = 4, cell_size: int = 100):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 10
        
        # Calculate window size
        window_size = board_size * (cell_size + self.margin) + self.margin
        self.window = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("2048 RL Agent Visualization")
        
        # Font for rendering numbers
        self.font = pygame.font.Font(None, 36)
        
    def draw_board(self, board: np.ndarray, action_probs: Optional[np.ndarray] = None):
        """Draw the game board and optionally show action probabilities"""
        self.window.fill((187, 173, 160))  # Background color
        
        # Draw cells
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = int(board[i, j])
                color = self.COLORS.get(value, (237, 194, 46))  # Default to 2048 color for higher values
                
                x = j * (self.cell_size + self.margin) + self.margin
                y = i * (self.cell_size + self.margin) + self.margin
                
                pygame.draw.rect(self.window, color, 
                               (x, y, self.cell_size, self.cell_size))
                
                if value != 0:
                    text = self.font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x + self.cell_size/2,
                                                    y + self.cell_size/2))
                    self.window.blit(text, text_rect)
        
        # Draw action probabilities if provided
        if action_probs is not None:
            directions = ["↑", "→", "↓", "←"]
            for i, (prob, direction) in enumerate(zip(action_probs, directions)):
                text = self.font.render(f"{direction}: {prob:.2f}", True, (0, 0, 0))
                self.window.blit(text, (10, self.board_size * self.cell_size + 20 + i*25))
        
        pygame.display.flip()

def visualize_gameplay(method: str, model_type: Optional[str] = None, 
                      num_episodes: int = 5, delay: float = 0.5):
    """Visualize the agent playing the game"""
    
    # Initialize environment and agent
    board_config = BoardConfig(board_size=4)
    rl_config = RLConfig(
        method=method,
        num_episodes=1000,  # Pre-train episodes
        learning_rate=0.001,
        gamma=0.99
    )
    
    env = Game2048Env(board_config.board_size)
    agent = create_agent(
        rl_config=rl_config,
        board_config=board_config,
        policy_based_model=model_type if method == "policy_based" else None,
        value_based_model=model_type if method == "value_based" else None
    )
    
    # Initialize visualizer
    visualizer = GameVisualizer()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"\nStarting Episode {episode + 1}")
        
        while not done:
            # Get action probabilities (if available)
            action_probs = None
            if hasattr(agent, "get_action_probs"):
                action_probs = agent.get_action_probs(state)
            
            # Visualize current state
            visualizer.draw_board(state, action_probs)
            time.sleep(delay)
            
            # Get and perform action
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    
    pygame.quit()

def main():
    # Example usage
    print("Available methods:")
    print("1. Value-based (MLP)")
    print("2. Value-based (CNN)")
    print("3. Policy-based (PGMC)")
    print("4. Policy-based (Actor-Critic)")
    print("5. Policy-based (TRPO)")
    print("6. Policy-based (PPO)")
    print("7. Model-based (MDP)")
    
    choice = input("Select method (1-7): ")
    
    methods = {
        "1": ("value_based", "mlp"),
        "2": ("value_based", "cnn"),
        "3": ("policy_based", "pgmc"),
        "4": ("policy_based", "actor_critic"),
        "5": ("policy_based", "trpo"),
        "6": ("policy_based", "ppo"),
        "7": ("model_based", None)
    }
    
    if choice in methods:
        method, model_type = methods[choice]
        visualize_gameplay(method, model_type)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main() 