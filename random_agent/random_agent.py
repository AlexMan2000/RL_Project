import random
import numpy as np
class RandomAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]

    def select_action(self, state: np.ndarray):
        return random.choice(self.action_space) 