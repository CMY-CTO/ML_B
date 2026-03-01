
from __future__ import annotations

import os
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from env_maze import MazeEnv
from qlearning_agent import QLearningAgent, QLearningConfig


@dataclass
class TrainConfig:
    episodes: int = 1000
    max_steps: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # multiply per episode

    seed: int = 0

    results_dir: str = "results"
    qtable_path: str = "results/qtable.npy"
    log_path: str = "results/train_log.csv"
    curve_path: str = "results/reward_curve.png"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train(cfg: TrainConfig) -> None:
    ensure_dir(cfg.results_dir)

    env = MazeEnv(max_steps=cfg.max_steps)
    n_states = env.width * env.height

    agent = QLearningAgent(
        n_states=n_states,
        config=QLearningConfig(alpha=0.1, gamma=0.95, n_actions=4),
        seed=cfg.seed,
    )

    epsilon = cfg.epsilon_start

    rewards = []
    steps_list = []
    success_list = []

    # CSV log
    with open(cfg.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "epsilon", "total_reward", "steps", "success"])
        visit_counts = np.zeros((n_states,), dtype=np.int32)

        for ep in range(1, cfg.episodes + 1):
            env.reset()
            s = env.state_id()

            total_reward = 0.0
            success = 0
            steps = 0

            for t in range(cfg.max_steps):
                a = agent.act(s, epsilon)
                _, r, done, info = env.step(a)
                s2 = env.state_id()

                agent.update(s, a, r, s2, done)

                s = s2
                total_reward += float(r)
                visit_counts[s] += 1
                steps = t + 1

                if done:
                    success = 1 if info.get("reached_goal", False) else 0
                    break

            # record
            rewards.append(total_reward)
            steps_list.append(steps)
            success_list.append(success)

            writer.writerow([ep, epsilon, total_reward, steps, success])

            # decay epsilon
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

            # lightweight console progress every 50 episodes
            if ep % 50 == 0:
                avg_r = float(np.mean(rewards[-50:]))
                sr = float(np.mean(success_list[-50:])) * 100.0
                print(f"Episode {ep:4d}/{cfg.episodes} | eps={epsilon:.3f} | avg_reward(50)={avg_r:.2f} | success_rate(50)={sr:.1f}%")

    # Save Q-table
    agent.save(cfg.qtable_path)
    print(f"Saved Q-table to: {cfg.qtable_path}")

    np.save("results/visit_counts.npy", visit_counts)
    print("Saved visit counts to: results/visit_counts.npy")
          
    # Plot reward curve (moving average)
    rewards_np = np.array(rewards, dtype=np.float32)
    window = 50
    if len(rewards_np) >= window:
        ma = np.convolve(rewards_np, np.ones(window) / window, mode="valid")
    else:
        ma = rewards_np

    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards_np, linewidth=1, alpha=0.35, label="Episode reward")
    plt.plot(np.arange(len(ma)) + (window - 1), ma, linewidth=2, label=f"Moving avg ({window})")
    plt.title("Q-learning Training Reward (Maze 10x10)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.curve_path, dpi=150)
    plt.close()
    print(f"Saved reward curve to: {cfg.curve_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)