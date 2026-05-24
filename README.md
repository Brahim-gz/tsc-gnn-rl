#  GNN Policy Network for Traffic Signal Control

> Deep Reinforcement Learning × Graph Representation Learning  
> Tools: CityFlow · PyTorch Geometric · Stable-Baselines3 · WandB

---

## Team

| Role | Owns |
|---|---|
| **Person A** — Environment & RL Engineer | `env/` · `agents/` · `training/train_*.py` · `evaluation/eval.py` |
| **Person B** — GNN & Model Engineer | `models/` · `training/sweep.py` · `evaluation/attention_viz.py` · `evaluation/generalization.py` |

---

## Project structure

```
project04_traffic/
│
├── interfaces.py              ← shared contract (STATE_DIM, model signatures)
│
├── env/
│   ├── cityflow_wrapper.py    ← CityFlow Gymnasium wrapper       [Person A]
│   └── graph_builder.py       ← .json → PyG Data builder         [Person A]
│
├── models/
│   ├── gat_encoder.py         ← 2-layer GAT encoder               [Person B]
│   ├── gnn_qnetwork.py        ← DQN Q-network                     [Person B]
│   ├── gnn_actor_critic.py    ← PPO actor-critic                  [Person B]
│   └── mlp_qnetwork.py        ← MLP ablation (no GNN)             [Person B]
│
├── agents/
│   └── baselines.py           ← Fixed-time, Max-Pressure, Random  [Person A]
│
├── training/
│   ├── replay_buffer.py       ← Graph replay buffer               [Person A]
│   ├── train_single.py        ← Week 1: single intersection        [Person A]
│   ├── train_multi.py         ← Week 2: multi-agent DQN + PPO     [Person A]
│   ├── sweep.py               ← GNN hyperparameter sweep          [Person B]
│   └── config.yaml            ← all hyperparameters
│
├── evaluation/
│   ├── eval.py                ← baseline comparison harness       [Person A]
│   ├── attention_viz.py       ← GAT attention visualisation       [Person B]
│   ├── generalization.py      ← zero-shot transfer test          [Person B]
│   └── plots.py               ← shared plot utilities             [Person B]
│
├── data/                      ← CityFlow roadnet.json + flow.json files
├── checkpoints/               ← saved model .pt files
├── logs/                      ← WandB CSV exports
└── results/                   ← plots, CSVs, final metrics
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/project04-traffic.git
cd project04-traffic

# 2. Environment
conda create -n tsc-gnn python=3.10 && conda activate tsc-gnn

# 3. CityFlow (needs cmake)
sudo apt-get install cmake build-essential   # Linux
pip install git+https://github.com/cityflow-project/CityFlow.git

# 4. PyG + RL stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install stable-baselines3[extra] wandb networkx matplotlib pandas tqdm
```

---

## Quick start

```bash
# Unit test all models (Person B)
python models/gat_encoder.py
python models/gnn_qnetwork.py
python models/gnn_actor_critic.py
python models/mlp_qnetwork.py

# Week 1: single intersection sanity check (Person A)
python training/train_single.py --no-wandb

# Week 2: full multi-agent training (Person A, after B's models are ready)
python training/train_multi.py --algo dqn
python training/train_multi.py --algo ppo

# Hyperparameter sweep (Person B, Week 2 Day 9)
python training/sweep.py

# Evaluate all baselines + trained model (Person A, Week 2 Day 10)
python evaluation/eval.py --checkpoint checkpoints/dqn_multi_final.pt

# Attention visualisation (Person B, Week 2 Day 11)
python evaluation/attention_viz.py --checkpoint checkpoints/dqn_multi_final.pt

# Generalization test (Person B, Week 2 Day 10)
python evaluation/generalization.py --checkpoint checkpoints/dqn_multi_final.pt
```

---

## Kaggle training (recommended)

```python
# Cell 1 — at the start of every Kaggle session
import os, sys
PROJECT_DIR = "/kaggle/working/project04"
if not os.path.exists(PROJECT_DIR):
    !git clone https://github.com/YOUR_USERNAME/project04-traffic.git {PROJECT_DIR}
else:
    !git -C {PROJECT_DIR} pull origin main
sys.path.insert(0, PROJECT_DIR)
```

---

## Interface contract

```python
# interfaces.py — never change alone
STATE_DIM  = 9     # [queue×4, phase_oh×3, elapsed×1]
EMBED_DIM  = 64    # GATEncoder output per node
NUM_PHASES = 4     # action space
EDGE_DIM   = 2     # [road_length_norm, num_lanes_norm]

# DQN interface
q_values = model(x, edge_index, edge_attr)          # → [N, NUM_PHASES]

# PPO interface
action_logits, state_value = model(x, edge_index, edge_attr)
# action_logits → [N, NUM_PHASES]
# state_value   → [N, 1]
```

---

## Prior art

| Paper | Venue | Code |
|---|---|---|
| CoLight | CIKM 2019 | [GitHub](https://github.com/traffic-signal-control/RL_signals) |
| PressLight | KDD 2019 | [GitHub](https://github.com/wingsweihua/presslight) |
| MPLight | AAAI 2020 | Paper |
| Advanced-XLight | AAAI 2022 | [GitHub](https://github.com/LiangZhang1996/Advanced_XLight) |
| RESCO benchmark | NeurIPS 2021 | [GitHub](https://github.com/Pi-Star-Lab/RESCO) |
