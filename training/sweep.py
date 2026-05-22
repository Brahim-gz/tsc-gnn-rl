# training/sweep.py
# ============================================================
# Hyperparameter sweep for the GNN architecture.

# Sweeps: GAT heads, layers, hidden dim.
# Uses a short 5K-step proxy training run per config.
# Reports best config for Person A to use in the full run.
#
# Run:
#   python training/sweep.py --roadnet data/synthetic_4x4/roadnet.json
# ============================================================

import os
import sys
import time
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, DEFAULT_CONFIG
from env.cityflow_wrapper import CityFlowEnv, make_cityflow_config
from env.graph_builder import build_grid_edge_index, build_graph_from_roadnet, update_graph_features, obs_matrix_to_dict
from training.replay_buffer import GraphReplayBuffer

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_GRID = {
    "gat_heads":  [1, 4, 8],
    "gat_layers": [1, 2, 3],
    "hidden_dim": [64, 128, 256],
}

PROXY_STEPS = 5_000    # steps per config (fast proxy — full run is 300K)
PROXY_N     = 16       # 4×4 grid


# ─────────────────────────────────────────────────────────────────────────────
# Proxy training run
# ─────────────────────────────────────────────────────────────────────────────

def proxy_train(
    roadnet:    str,
    flow:       str,
    gat_heads:  int,
    gat_layers: int,
    hidden_dim: int,
    device:     torch.device,
    seed:       int = 0,
) -> dict:
    """
    Run PROXY_STEPS steps of DQN with the given GNN config.
    Returns a dict with the config and final performance metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "gat_heads":  gat_heads,
        "hidden_dim": hidden_dim,
        "total_steps": PROXY_STEPS,
        "epsilon_decay": PROXY_STEPS // 2,
    })

    # ── Dynamic GATEncoder with configurable depth ────────────────────────────
    from models.gat_encoder import GATEncoder
    import torch.nn as nn

    class FlexGATEncoder(nn.Module):
        """GATEncoder with variable depth (for sweep)."""
        def __init__(self, in_dim, hidden_dim, out_dim, heads, edge_dim, n_layers, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList()
            self.norms  = nn.ModuleList()
            cur_dim = in_dim
            for i in range(n_layers):
                is_last  = (i == n_layers - 1)
                out      = out_dim if is_last else hidden_dim
                h        = 1 if is_last else heads
                concat   = not is_last
                self.layers.append(
                    torch.torch_geometric_GATConv_placeholder := None  # ← see below
                )
            # Build properly with torch_geometric
            from torch_geometric.nn import GATConv
            self.layers = nn.ModuleList()
            self.norms  = nn.ModuleList()
            cur_dim = in_dim
            for i in range(n_layers):
                is_last = (i == n_layers - 1)
                out     = out_dim if is_last else hidden_dim
                h       = 1      if is_last else heads
                concat  = not is_last
                self.layers.append(GATConv(cur_dim, out, heads=h, edge_dim=edge_dim,
                                           dropout=dropout, concat=concat,
                                           add_self_loops=False))
                self.norms.append(nn.LayerNorm(out if is_last else out * h))
                cur_dim = out if is_last else out * h
            self.dropout_p = dropout
            self.out_dim   = out_dim

        def forward(self, x, edge_index, edge_attr=None):
            import torch.nn.functional as F
            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                x = layer(x, edge_index, edge_attr=edge_attr)
                x = norm(x)
                if i < len(self.layers) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
            return x

    class FlexQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FlexGATEncoder(
                in_dim=STATE_DIM, hidden_dim=hidden_dim, out_dim=cfg["embed_dim"],
                heads=gat_heads, edge_dim=2, n_layers=gat_layers,
            )
            self.q_head = nn.Sequential(
                nn.Linear(cfg["embed_dim"], hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, NUM_PHASES),
            )
        def forward(self, x, edge_index, edge_attr=None):
            return self.q_head(self.encoder(x, edge_index, edge_attr))

    q_net   = FlexQNet().to(device)
    target  = FlexQNet().to(device)
    target.load_state_dict(q_net.state_dict())
    opt     = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])

    # ── Environment ───────────────────────────────────────────────────────────
    config_path = make_cityflow_config(roadnet, flow, f"data/sweep/seed{seed}", seed=seed)
    env = CityFlowEnv(config_path, num_intersections=PROXY_N,
                      episode_seconds=cfg["episode_seconds"],
                      delta_time=cfg["delta_time"], reward_fn="queue")

    try:
        graph, tl_ids = build_graph_from_roadnet(roadnet)
    except Exception:
        graph  = build_grid_edge_index(n=PROXY_N, side=4)
        tl_ids = env.intersection_ids
    graph = graph.to(device)

    buffer = GraphReplayBuffer(
        capacity=min(cfg["replay_capacity"], PROXY_STEPS * 2),
        edge_index=graph.edge_index, edge_attr=graph.edge_attr,
        num_nodes=PROXY_N, device=device,
    )

    obs, _ = env.reset()
    ep_rewards, ep_travel_times = [], []
    ep_r, ep_steps = 0.0, 0
    t0 = time.time()

    for step in range(PROXY_STEPS):
        eps = max(0.05, 1.0 - step / (PROXY_STEPS // 2) * 0.95)

        if np.random.random() < eps:
            actions = env.action_space.sample()
        else:
            graph = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
            with torch.no_grad():
                q     = q_net(graph.x.to(device), graph.edge_index, graph.edge_attr)
                actions = q.argmax(dim=1).cpu().numpy()

        next_obs, reward, done, _, info = env.step(actions)
        buffer.push(obs, actions, reward, next_obs, done)
        obs    = next_obs
        ep_r  += reward
        ep_steps += 1

        if buffer.is_ready(cfg["batch_size"]) and step % 4 == 0:
            batch  = buffer.sample(cfg["batch_size"])
            B, N_  = batch["B"], batch["N"]
            q_curr = q_net(batch["obs"], batch["edge_index"], batch["edge_attr"]).view(B, N_, NUM_PHASES)
            with torch.no_grad():
                q_next = target(batch["next_obs"], batch["edge_index"], batch["edge_attr"]).view(B, N_, NUM_PHASES).max(dim=2).values
            target_q = batch["rewards"].unsqueeze(1) + cfg["gamma"] * q_next * (1 - batch["dones"].unsqueeze(1))
            q_chosen = q_curr.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)
            loss     = F.smooth_l1_loss(q_chosen, target_q.detach())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()

        if step % cfg["target_update_freq"] == 0:
            target.load_state_dict(q_net.state_dict())

        if done:
            ep_rewards.append(ep_r)
            ep_travel_times.append(env.avg_travel_time)
            ep_r, ep_steps = 0.0, 0
            obs, _ = env.reset()

    env.close()
    elapsed = time.time() - t0

    # Metric: mean travel time over last 3 episodes (lower = better)
    last_travel = float(np.mean(ep_travel_times[-3:])) if ep_travel_times else 9999.0
    last_reward = float(np.mean(ep_rewards[-3:]))       if ep_rewards     else -9999.0

    return {
        "gat_heads":       gat_heads,
        "gat_layers":      gat_layers,
        "hidden_dim":      hidden_dim,
        "avg_travel_time": last_travel,
        "avg_reward":      last_reward,
        "n_params":        sum(p.numel() for p in q_net.parameters()),
        "wall_time_s":     elapsed,
        "n_episodes":      len(ep_rewards),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Proxy steps per config: {PROXY_STEPS:,}")
    print(f"Sweep grid: {SWEEP_GRID}")

    configs = list(itertools.product(
        SWEEP_GRID["gat_heads"],
        SWEEP_GRID["gat_layers"],
        SWEEP_GRID["hidden_dim"],
    ))
    total = len(configs)
    print(f"\nTotal configs: {total}  (≈ {total * PROXY_STEPS / 1000:.0f}K total steps)")

    if WANDB and not args.no_wandb:
        wandb.init(project="project04-traffic-gnn", name="sweep", config=SWEEP_GRID)

    results = []
    for i, (heads, layers, hdim) in enumerate(configs):
        print(f"\n[{i+1}/{total}] heads={heads}  layers={layers}  hidden={hdim}")
        r = proxy_train(
            roadnet    = args.roadnet,
            flow       = args.flow,
            gat_heads  = heads,
            gat_layers = layers,
            hidden_dim = hdim,
            device     = device,
            seed       = args.seed,
        )
        results.append(r)
        print(f"  travel_time={r['avg_travel_time']:.1f}s  "
              f"reward={r['avg_reward']:.1f}  "
              f"params={r['n_params']:,}  "
              f"time={r['wall_time_s']:.0f}s")

        if WANDB and not args.no_wandb:
            wandb.log(r)

    # ── Results table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("avg_travel_time")
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/sweep_results.csv", index=False)

    print("\n── Sweep Results (sorted by avg travel time, lower = better) ──")
    print(df[["gat_heads", "gat_layers", "hidden_dim",
              "avg_travel_time", "avg_reward", "n_params"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\n🏆 Best config:")
    print(f"   gat_heads  = {int(best['gat_heads'])}")
    print(f"   gat_layers = {int(best['gat_layers'])}")
    print(f"   hidden_dim = {int(best['hidden_dim'])}")
    print(f"   travel_time = {best['avg_travel_time']:.1f}s")
    print(f"\n→ Update DEFAULT_CONFIG in interfaces.py with these values.")
    print(f"→ Full results saved to results/sweep_results.csv")

    # Plot sweep heatmap
    _plot_sweep_heatmap(df)

    if WANDB and not args.no_wandb:
        wandb.finish()


def _plot_sweep_heatmap(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SWEEP_GRID["gat_layers"]),
                              figsize=(14, 4), sharey=True)

    for ax, n_layers in zip(axes, SWEEP_GRID["gat_layers"]):
        sub = df[df["gat_layers"] == n_layers].pivot(
            index="hidden_dim", columns="gat_heads", values="avg_travel_time"
        )
        im = ax.imshow(sub.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([f"{h} heads" for h in sub.columns])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([f"h={d}" for d in sub.index])
        ax.set_title(f"{n_layers} GAT layer{'s' if n_layers > 1 else ''}", fontsize=11)
        ax.set_xlabel("Attention heads")

        # Annotate cells
        for ri in range(sub.values.shape[0]):
            for ci in range(sub.values.shape[1]):
                v = sub.values[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="black")

    axes[0].set_ylabel("Hidden dim")
    fig.colorbar(im, ax=axes[-1], label="Avg travel time (s) — lower better", shrink=0.8)
    fig.suptitle("GNN Hyperparameter Sweep — Proxy 5K steps",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved sweep heatmap → results/sweep_heatmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet",  default="data/synthetic_4x4/roadnet.json")
    parser.add_argument("--flow",     default="data/synthetic_4x4/flow.json")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(args)
