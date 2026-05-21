# training/train_single.py
# ============================================================
# Purpose: sanity check that the GNN plugs in correctly
# before scaling to multi-agent.
#
# Run:
#     python training/train_single.py
# ============================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import DEFAULT_CONFIG, STATE_DIM, NUM_PHASES
from env.cityflow_wrapper import CityFlowEnv, make_cityflow_config
from env.graph_builder import build_grid_edge_index
from training.replay_buffer import GraphReplayBuffer
from agents.baselines import FixedTimeBaseline, RandomBaseline

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed — logging to console only.")


def get_epsilon(step: int, cfg: dict) -> float:
    """Linear epsilon decay from epsilon_start to epsilon_end."""
    progress = min(step / cfg["epsilon_decay"], 1.0)
    return cfg["epsilon_start"] + progress * (cfg["epsilon_end"] - cfg["epsilon_start"])


def load_model(cfg: dict, device: torch.device):
    """
    Import  GNN_QNetwork.
    Falls back to a stub MLP if B's file doesn't exist yet.
    """
    try:
        from models.gnn_qnetwork import GNN_QNetwork
        model = GNN_QNetwork(
            state_dim  = STATE_DIM,
            embed_dim  = cfg["embed_dim"],
            num_phases = NUM_PHASES,
            gat_heads  = cfg["gat_heads"],
            hidden_dim = cfg["hidden_dim"],
        ).to(device)
        print("✓ Loaded GNN_QNetwork from models/gnn_qnetwork.py")
    except ImportError:
        print("⚠ models/gnn_qnetwork.py not found — using MLP stub (Day 4 only)")
        import torch.nn as nn
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(STATE_DIM, 64), nn.ReLU(), nn.Linear(64, NUM_PHASES)
                )
            def forward(self, x, edge_index, edge_attr=None):
                return self.fc(x)
        model = _Stub().to(device)

    return model


def train_single(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── WandB ────────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project = "project04-traffic-gnn",
            name    = f"dqn-single-{time.strftime('%m%d-%H%M')}",
            config  = cfg,
            resume  = "allow",
        )

    # ── Environment (1 intersection, 1×1 grid) ───────────────────────────────
    config_path = make_cityflow_config(
        roadnet_path = args.roadnet,
        flow_path    = args.flow,
        save_dir     = "data/single",
    )
    env = CityFlowEnv(
        config_path       = config_path,
        num_intersections = 1,
        episode_seconds   = cfg["episode_seconds"],
        delta_time        = cfg["delta_time"],
        reward_fn         = args.reward,
    )

    # ── Graph topology (single node, no real edges — acts like MLP) ──────────
    graph = build_grid_edge_index(n=1, side=1)
    graph = graph.to(device)

    # ── Models ───────────────────────────────────────────────────────────────
    q_net    = load_model(cfg, device)
    target   = load_model(cfg, device)
    target.load_state_dict(q_net.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])

    # ── Replay buffer ────────────────────────────────────────────────────────
    buffer = GraphReplayBuffer(
        capacity   = cfg["replay_capacity"],
        edge_index = graph.edge_index,
        edge_attr  = graph.edge_attr,
        num_nodes  = 1,
        device     = device,
    )

    # ── Baselines (for comparison) ───────────────────────────────────────────
    fixed_time   = FixedTimeBaseline(num_intersections=1)
    random_agent = RandomBaseline(num_intersections=1)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path    = "checkpoints/single_latest.pt"
    start_step   = 0

    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt       = torch.load(ckpt_path, map_location=device)
        q_net.load_state_dict(ckpt["q_net"])
        target.load_state_dict(ckpt["target"])
        start_step = ckpt["step"]
        print(f"✓ Resumed from step {start_step:,}")

    # ── Training loop ────────────────────────────────────────────────────────
    obs, _    = env.reset()
    ep_reward = 0.0
    ep_steps  = 0
    ep_num    = 0
    t0        = time.time()

    print(f"\nTraining single-intersection GNN-DQN for {cfg['total_steps']:,} steps...")
    print(f"{'Step':>8}  {'Ep':>5}  {'Reward':>8}  {'Loss':>8}  {'ε':>6}  {'Steps/s':>8}")
    print("-" * 60)

    for step in range(start_step, cfg["total_steps"]):
        epsilon = get_epsilon(step, cfg)

        # ε-greedy action
        if np.random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            x = torch.tensor(obs, dtype=torch.float32).to(device)  # [1, STATE_DIM]
            with torch.no_grad():
                q_vals  = q_net(x, graph.edge_index, graph.edge_attr)  # [1, NUM_PHASES]
                actions = q_vals.argmax(dim=1).cpu().numpy()

        next_obs, reward, done, _, info = env.step(actions)
        buffer.push(obs, actions, reward, next_obs, done)
        obs        = next_obs
        ep_reward += reward
        ep_steps  += 1

        # ── Learn ────────────────────────────────────────────────────────────
        loss_val = 0.0
        if buffer.is_ready(cfg["batch_size"]) and step % cfg["train_freq"] == 0:
            batch = buffer.sample(cfg["batch_size"])
            loss_val = _dqn_update(q_net, target, optimizer, batch, cfg)

        # ── Target network ───────────────────────────────────────────────────
        if step % cfg["target_update_freq"] == 0:
            target.load_state_dict(q_net.state_dict())

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            ep_num    += 1
            sps        = ep_steps / max(time.time() - t0, 1e-6)
            log        = {
                "episode":        ep_num,
                "episode_reward": ep_reward,
                "episode_steps":  ep_steps,
                "epsilon":        epsilon,
                "steps_per_sec":  sps,
                "step":           step,
            }
            if WANDB_AVAILABLE and not args.no_wandb:
                wandb.log(log)
            if ep_num % 5 == 0:
                print(f"{step:>8,}  {ep_num:>5}  {ep_reward:>8.1f}  "
                      f"{loss_val:>8.4f}  {epsilon:>6.3f}  {sps:>8.1f}")

            ep_reward = 0.0
            ep_steps  = 0
            t0        = time.time()
            obs, _    = env.reset()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % cfg["save_freq"] == 0 and step > start_step:
            torch.save({"step": step, "q_net": q_net.state_dict(),
                        "target": target.state_dict()}, ckpt_path)
            print(f"  → Checkpoint saved at step {step:,}")

    env.close()
    print("\nTraining complete.")
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


def _dqn_update(q_net, target, optimizer, batch, cfg) -> float:
    """One DQN gradient step. Returns scalar loss."""
    B, N = batch["B"], batch["N"]

    q_curr = q_net(
        batch["obs"], batch["edge_index"], batch["edge_attr"]
    ).view(B, N, NUM_PHASES)                    # [B, N, phases]

    with torch.no_grad():
        q_next = target(
            batch["next_obs"], batch["edge_index"], batch["edge_attr"]
        ).view(B, N, NUM_PHASES).max(dim=2).values   # [B, N]

    target_q = (
        batch["rewards"].unsqueeze(1) +
        cfg["gamma"] * q_next * (1.0 - batch["dones"].unsqueeze(1))
    )                                            # [B, N]

    # Gather Q-values for the chosen actions
    actions_exp = batch["actions"].unsqueeze(2) # [B, N, 1]
    q_chosen    = q_curr.gather(2, actions_exp).squeeze(2)  # [B, N]

    loss = F.smooth_l1_loss(q_chosen, target_q.detach())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet",   default="data/synthetic_4x4/roadnet.json")
    parser.add_argument("--flow",      default="data/synthetic_4x4/flow.json")
    parser.add_argument("--reward",    default="queue", choices=["queue","pressure","combined"])
    parser.add_argument("--no-wandb",  action="store_true")
    parser.add_argument("--fresh",     action="store_true", help="Ignore existing checkpoint")
    args = parser.parse_args()
    train_single(args)