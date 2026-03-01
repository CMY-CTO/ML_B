
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pygame

from env_maze import MazeEnv

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DIRS = {
    0: (0, -1),
    1: (0,  1),
    2: (-1, 0),
    3: (1,  0),
}

# Colors
COL_BG = (18, 18, 22)
COL_PANEL = (28, 28, 35)
COL_GRID = (55, 55, 70)

COL_WALL = (35, 35, 42)
COL_FREE = (235, 235, 240)
COL_START = (110, 190, 255)
COL_GOAL = (140, 235, 140)
COL_AGENT = (255, 120, 120)

COL_TEXT = (240, 240, 245)
COL_TEXT_DIM = (180, 180, 195)

COL_TRACE = (255, 200, 90)  # path trace


@dataclass
class UIConfig:
    cell_size: int = 60
    margin: int = 2
    top_panel_h: int = 130
    fps: int = 60
    step_delay: float = 0.06

    qtable_path: str = "results/qtable.npy"
    visit_path: str = "results/visit_counts.npy"  # optional


def load_npy_if_exists(path: str):
    return np.load(path) if os.path.exists(path) else None


def greedy_action(Q: np.ndarray, s: int, rng: np.random.Generator) -> int:
    row = Q[s]
    m = np.max(row)
    best = np.flatnonzero(row == m)
    return int(rng.choice(best))


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def lerp(a, b, t: float):
    return a + (b - a) * t


def heat_color(t: float):
    """
    Map t in [0,1] to a nice blue->green->yellow->red palette.
    """
    t = clamp01(t)
    # piecewise
    if t < 0.33:
        # blue -> cyan
        u = t / 0.33
        r = int(lerp(40, 60, u))
        g = int(lerp(90, 200, u))
        b = int(lerp(220, 240, u))
    elif t < 0.66:
        # cyan -> yellow
        u = (t - 0.33) / 0.33
        r = int(lerp(60, 240, u))
        g = int(lerp(200, 230, u))
        b = int(lerp(240, 80, u))
    else:
        # yellow -> red
        u = (t - 0.66) / 0.34
        r = int(lerp(240, 255, u))
        g = int(lerp(230, 80, u))
        b = int(lerp(80, 80, u))
    return (r, g, b)


def draw_text(surface, font, text, x, y, color=COL_TEXT):
    surface.blit(font.render(text, True, color), (x, y))


def action_arrow_points(cx, cy, a, size):
    # triangle pointing in action direction
    if a == 0:  # up
        return [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    if a == 1:  # down
        return [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    if a == 2:  # left
        return [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]
    if a == 3:  # right
        return [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]


def main():
    cfg = UIConfig()

    env = MazeEnv(max_steps=200)
    if not os.path.exists(cfg.qtable_path):
        raise FileNotFoundError(f"Missing {cfg.qtable_path}. Run train_qtable.py first.")
    Q = np.load(cfg.qtable_path).astype(np.float32)
    visits = load_npy_if_exists(cfg.visit_path)
    rng = np.random.default_rng(0)

    pygame.init()
    pygame.display.set_caption("Q-learning Maze - Advanced Viewer")

    grid_w = env.width * cfg.cell_size
    grid_h = env.height * cfg.cell_size
    win_w = grid_w
    win_h = cfg.top_panel_h + grid_h
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 18)
    font_big = pygame.font.SysFont("Arial", 22, bold=True)

    # UI toggles
    auto_run = True
    show_policy = True
    show_trace = True
    heat_mode = "value"  # "value" or "visit"
    show_heat = True

    epsilon = 0.0  # greedy by default
    step_delay = cfg.step_delay
    last_step_time = 0.0

    episode = 1
    total_reward = 0.0
    last_reward = 0.0
    last_action = None
    done = False

    trace = []  # list of (x,y) visited this episode

    env.reset()
    trace.append(tuple(env.agent_pos))

    def new_episode():
        nonlocal episode, total_reward, last_reward, last_action, done, trace
        env.reset()
        total_reward = 0.0
        last_reward = 0.0
        last_action = None
        done = False
        trace = [tuple(env.agent_pos)]

    def state_id_xy(x, y):
        return y * env.width + x

    def compute_V():
        return np.max(Q, axis=1)

    V = compute_V()
    V_free = V.copy()

    # Normalize helpers for heatmaps
    def heat_value_at(x, y):
        s = state_id_xy(x, y)
        return float(V_free[s])

    def heat_visit_at(x, y):
        if visits is None:
            return 0.0
        s = state_id_xy(x, y)
        return float(visits[s])

    def draw():
        screen.fill(COL_BG)

        # panel
        pygame.draw.rect(screen, COL_PANEL, pygame.Rect(0, 0, win_w, cfg.top_panel_h))
        draw_text(screen, font_big, "Advanced Q-learning Viewer", 12, 10)
        draw_text(
            screen, font,
            f"Episode={episode} | step={env.step_count}/{env.max_steps} | total_reward={total_reward:.2f} | last_r={last_reward:.2f}",
            12, 40, COL_TEXT_DIM
        )
        la = "None" if last_action is None else f"{last_action}({ACTIONS[last_action]})"
        draw_text(
            screen, font,
            f"auto={auto_run} | eps={epsilon:.2f} | heat={show_heat}({heat_mode}) | policy={show_policy} | trace={show_trace} | last_action={la}",
            12, 64, COL_TEXT_DIM
        )
        draw_text(
            screen, font,
            "Keys: Space auto | Enter reset | N next | P policy | T trace | H heat on/off | Tab heat mode | +/- speed | Esc quit",
            12, cfg.top_panel_h - 26, COL_TEXT_DIM
        )

        origin_y = cfg.top_panel_h

        # prepare heat normalization (only on free cells)
        vals = []
        for y in range(env.height):
            for x in range(env.width):
                ch = env.grid_lines[y][x]
                if ch == "#":
                    continue
                if show_heat:
                    if heat_mode == "value":
                        vals.append(heat_value_at(x, y))
                    else:
                        vals.append(heat_visit_at(x, y))
        vmin = float(np.min(vals)) if vals else 0.0
        vmax = float(np.max(vals)) if vals else 1.0
        denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0

        # draw cells
        for y in range(env.height):
            for x in range(env.width):
                ch = env.grid_lines[y][x]
                rect = pygame.Rect(
                    x * cfg.cell_size + cfg.margin,
                    origin_y + y * cfg.cell_size + cfg.margin,
                    cfg.cell_size - 2 * cfg.margin,
                    cfg.cell_size - 2 * cfg.margin,
                )

                if ch == "#":
                    base = COL_WALL
                elif ch == "S":
                    base = COL_START
                elif ch == "G":
                    base = COL_GOAL
                else:
                    base = COL_FREE

                # heat overlay on free-ish cells
                if show_heat and ch != "#":
                    raw = heat_value_at(x, y) if heat_mode == "value" else heat_visit_at(x, y)
                    t = (raw - vmin) / denom
                    hc = heat_color(t)
                    # blend base with heat
                    blend = (
                        int(0.45 * base[0] + 0.55 * hc[0]),
                        int(0.45 * base[1] + 0.55 * hc[1]),
                        int(0.45 * base[2] + 0.55 * hc[2]),
                    )
                    pygame.draw.rect(screen, blend, rect, border_radius=10)
                else:
                    pygame.draw.rect(screen, base, rect, border_radius=10)

                # policy arrow
                if show_policy and ch != "#":
                    s = state_id_xy(x, y)
                    a = int(np.argmax(Q[s]))
                    cx = x * cfg.cell_size + cfg.cell_size // 2
                    cy = origin_y + y * cfg.cell_size + cfg.cell_size // 2
                    pts = action_arrow_points(cx, cy, a, size=cfg.cell_size // 6)
                    pygame.draw.polygon(screen, (30, 30, 30), pts)
                    pygame.draw.polygon(screen, (255, 255, 255), pts, width=2)

        # grid lines
        for x in range(env.width + 1):
            pygame.draw.line(screen, COL_GRID, (x * cfg.cell_size, origin_y), (x * cfg.cell_size, origin_y + grid_h), 1)
        for y in range(env.height + 1):
            pygame.draw.line(screen, COL_GRID, (0, origin_y + y * cfg.cell_size), (grid_w, origin_y + y * cfg.cell_size), 1)

        # trace
        if show_trace and len(trace) >= 2:
            pts = []
            for (tx, ty) in trace:
                pts.append((
                    tx * cfg.cell_size + cfg.cell_size // 2,
                    origin_y + ty * cfg.cell_size + cfg.cell_size // 2
                ))
            pygame.draw.lines(screen, COL_TRACE, False, pts, width=4)

        # agent
        ax, ay = env.agent_pos
        cx = ax * cfg.cell_size + cfg.cell_size // 2
        cy = origin_y + ay * cfg.cell_size + cfg.cell_size // 2
        pygame.draw.circle(screen, COL_AGENT, (cx, cy), cfg.cell_size // 4)

        pygame.display.flip()

    def do_step(action: int):
        nonlocal total_reward, last_reward, last_action, done
        if done:
            return
        _, r, d, info = env.step(action)
        last_reward = float(r)
        total_reward += float(r)
        last_action = action
        done = bool(d)
        trace.append(tuple(env.agent_pos))

    running = True
    while running:
        clock.tick(cfg.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_SPACE:
                    auto_run = not auto_run

                elif event.key == pygame.K_RETURN:
                    new_episode()

                elif event.key == pygame.K_n:
                    episode += 1
                    new_episode()

                elif event.key == pygame.K_p:
                    show_policy = not show_policy

                elif event.key == pygame.K_t:
                    show_trace = not show_trace

                elif event.key == pygame.K_h:
                    show_heat = not show_heat

                elif event.key == pygame.K_TAB:
                    heat_mode = "visit" if heat_mode == "value" else "value"

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    step_delay = max(0.0, step_delay - 0.02)

                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    step_delay = min(1.0, step_delay + 0.02)

        now = time.time()
        if auto_run and (now - last_step_time) >= step_delay:
            last_step_time = now
            s = env.state_id()
            a = greedy_action(Q, s, rng) if epsilon <= 0 else (
                a if (rng.random() < epsilon) else greedy_action(Q, s, rng)
            )
            do_step(a)

            if done:
                time.sleep(0.35)
                episode += 1
                new_episode()

        draw()

    pygame.quit()


if __name__ == "__main__":
    main()