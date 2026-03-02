# Env
pip install -r requirements.txt

# Run
python play_ui.py

# Environment & Rewards (Summary)
- Grid symbols:
- - `S` start, `G` goal, `#` wall, `.` free
- Actions: `0`=UP, `1`=DOWN, `2`=LEFT, `3`=RIGHT
- Rewards (from env_maze.py):
- - Move to free cell: -1
- - Hit wall/out of bounds (stay in place): -2
- - Reach goal: +20 and done=True
- Episode ends when:
- - goal reached, or
- - step count reaches max_steps

# UI Controls
- `Space` — toggle auto-run
- `Enter` — reset current episode
- `N` — next episode
- `P` — toggle policy arrows
- `T` — toggle trajectory trace
- `H` — toggle heatmap on/off
- `Tab` — switch heatmap mode (value / visit)
- `+ / -` — speed up / slow down
- `Esc` or `Q` — quit

<img width="83" height="147" alt="Screen Shot 2026-03-01 at 17 13 32" src="https://github.com/user-attachments/assets/09bd0520-36bc-4796-9126-a70ee8010a5b" />

<img width="575" height="340" alt="Screen Shot 2026-03-01 at 16 50 11" src="https://github.com/user-attachments/assets/ebe357f9-abf1-422d-a97e-2a9d7213e5d1" />


<img width="593" height="752" alt="Screen Shot 2026-03-01 at 16 57 27" src="https://github.com/user-attachments/assets/df30ca70-6118-42ff-a883-812350f0cae6" />

# References
- [1] Sutton & Barto (2018), Reinforcement Learning: An Introduction 
- [2] Watkins & Dayan (1992), Q-learning
- [3] Pygame Docs: https://www.pygame.org/docs/
