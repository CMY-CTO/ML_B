
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QLearningConfig:
    alpha: float = 0.1          # learning rate
    gamma: float = 0.95         # discount factor
    n_actions: int = 4          # UP/DOWN/LEFT/RIGHT


class QLearningAgent:
    def __init__(self, n_states: int, config: QLearningConfig = QLearningConfig(), seed: int = 0):
        self.n_states = n_states
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        # Q-table: shape (n_states, n_actions)
        self.Q = np.zeros((n_states, self.cfg.n_actions), dtype=np.float32)

    def act(self, state_id: int, epsilon: float) -> int:
        """ε-greedy action selection."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.cfg.n_actions))
        # greedy (break ties randomly)
        q = self.Q[state_id]
        max_q = np.max(q)
        best_actions = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """Q-learning update."""
        q_sa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.cfg.gamma * float(np.max(self.Q[s2]))
        self.Q[s, a] = q_sa + self.cfg.alpha * (target - q_sa)

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path)
        # minimal sanity check
        if self.Q.shape != (self.n_states, self.cfg.n_actions):
            raise ValueError(f"Loaded Q has shape {self.Q.shape}, expected {(self.n_states, self.cfg.n_actions)}")