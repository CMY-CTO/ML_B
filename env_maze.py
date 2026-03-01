from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


Action = int  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT


@dataclass
class StepInfo:
    hit_wall: bool
    reached_goal: bool
    step_count: int


class MazeEnv:
    """
    Minimal GridWorld maze environment (no external deps like gym).
    - Grid symbols: S (start), G (goal), # (wall), . (free)
    - State: agent (x, y)
    - Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    - Rewards:
        step to free cell: -1
        hit wall / out of bounds: -2 (agent stays)
        reach goal: +20 and done=True
    """

    ACTIONS = {
        0: (0, -1),  # UP
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (1, 0),   # RIGHT
    }

    def __init__(self, grid_lines: List[str] | None = None, max_steps: int = 200):
        if grid_lines is None:
            grid_lines = [
                "S.........",
                ".#######..",
                ".......#..",
                "..#####.#.",
                "..#...#.#.",
                "..#.#.#.#.",
                "..#.#...#.",
                "..#.#####.",
                "..#.......",
                "..#######G",
                ]

        # Basic validation
        if len(grid_lines) == 0:
            raise ValueError("grid_lines cannot be empty")
        width = len(grid_lines[0])
        if any(len(row) != width for row in grid_lines):
            raise ValueError("All grid lines must have the same length")

        self.grid_lines = grid_lines
        self.height = len(grid_lines)
        self.width = width
        self.max_steps = max_steps

        self.start_pos = self._find_char("S")
        self.goal_pos = self._find_char("G")

        self.agent_pos = self.start_pos
        self.step_count = 0

    def _find_char(self, ch: str) -> Tuple[int, int]:
        for y, row in enumerate(self.grid_lines):
            x = row.find(ch)
            if x != -1:
                return (x, y)
        raise ValueError(f"Grid must contain '{ch}'")

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos
        self.step_count = 0
        return self.agent_pos

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid_lines[y][x] == "#"

    def is_goal(self, x: int, y: int) -> bool:
        return (x, y) == self.goal_pos

    def state_id(self, pos: Tuple[int, int] | None = None) -> int:
        if pos is None:
            pos = self.agent_pos
        x, y = pos
        return y * self.width + x

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        if action not in self.ACTIONS:
            raise ValueError("Invalid action. Must be 0=UP,1=DOWN,2=LEFT,3=RIGHT")

        dx, dy = self.ACTIONS[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        self.step_count += 1

        hit_wall = False
        reached_goal = False
        done = False
        reward = 0.0

        # Check collision / bounds
        if (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny):
            hit_wall = True
            # stay in place
            nx, ny = x, y
            reward = -2.0
        else:
            # valid move
            self.agent_pos = (nx, ny)
            reward = -1.0

        # Goal check (after movement)
        if self.is_goal(nx, ny):
            reached_goal = True
            reward = 20.0
            done = True

        # Timeout
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "hit_wall": hit_wall,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
        }
        return self.agent_pos, reward, done, info

    def as_text(self) -> str:
        """Text rendering for quick debugging in terminal."""
        ax, ay = self.agent_pos
        lines = []
        for y, row in enumerate(self.grid_lines):
            chars = list(row)
            if y == ay:
                # Don't overwrite the goal symbol if standing on it
                if chars[ax] not in ("G",):
                    chars[ax] = "A"
            lines.append("".join(chars))
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-test (terminal)
    env = MazeEnv(max_steps=50)
    s = env.reset()
    print("Initial state:", s, "state_id:", env.state_id())
    print(env.as_text())
    print("\nTry moving RIGHT, RIGHT, DOWN ...\n")

    for a in [3, 3, 1, 1, 1]:
        ns, r, done, info = env.step(a)
        print(f"action={a} -> next_state={ns}, reward={r}, done={done}, info={info}")
        print(env.as_text())
        print("-" * 20)
        if done:
            break