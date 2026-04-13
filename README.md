# OBELIX – CS780 Capstone

Implementation and evaluation code for the OBELIX warehouse robot RL task.

## Contents
- `obelix.py` – Environment
- `rppo.py` – Recurrent PPO agent
- `rppo_weights.pth` – Trained weights
- `evaluate.py` – Local evaluation script
- `evaluate_on_codabench.py` – Codabench-style evaluation
- `play_rppo.py` – Visualize agent behavior
- `manual_play.py` – Manual keyboard control
- `training.ipynb` – Training notebook

## Quick Start
Run trained agent with rendering:
```bash
python play_rppo.py --wall_obstacles --difficulty 3
