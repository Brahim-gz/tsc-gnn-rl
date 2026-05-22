# evaluation/generalization.py
# ============================================================
# Tests whether the trained GNN transfers zero-shot to:
#   (a) larger grids (6×6) — same topology family
#   (b) real city maps (Hangzhou, Jinan) — irregular topology
#
# Key insight being tested:
#   GNNs are structurally equivariant — the same learned
#   message-passing weights work on any graph topology.
#   An MLP with hardcoded neighbour positions cannot do this.
#
# Run:
#   python evaluation/generalization.py \
#       --checkpoint checkpoints/dqn_multi_final.pt
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, DEFAULT_CONFIG
from env.cityflow_wrapper import CityFlowEnv, make_cityflow_config
from env.graph_builder import (
    build_graph_from_roadnet,
    build_grid_edge_index,
    update_graph_features,
    obs_matrix_to_dict,
)
from models.gnn_qnetwork import GNN_QNetwork
from models.mlp_qnetwork import MLP_QNetwork
from agents.baselines import FixedTimeBaseline, MaxPressureBaseline


# ─────────────────────────────────────────────────────────────────────────────
# Network registry — all test scenarios
# ─────────────────────────────────────────────────────────────────────────────

NETWORKS = {
    # ── Training distribution ─────────────────────────────────────────────────
    "synthetic_4x4": {
        "roadnet":       "data/synthetic_4x4/roadnet.json",
        "flow":          "data/synthetic_4x4/flow.json",
        "n":             16,
        "side":          4,
        "label":         "Synthetic 4×4 (train)",
        "train_domain":  True,
    },
    # ── Same topology, larger scale ───────────────────────────────────────────
    "synthetic_6x6": {
        "roadnet":       "data/synthetic_6x6/roadnet.json",
        "flow":          "data/synthetic_6x6/flow.json",
        "n":             36,
        "side":          6,
        "label":         "Synthetic 6×6 (scaled)",
        "train_domain":  False,
    },
    # ── Real-world irregular topologies ──────────────────────────────────────
    "hangzhou_4x4": {
        "roadnet":       "data/hangzhou_4x4/roadnet.json",
        "flow":          "data/hangzhou_4x4/flow.json",
        "n":             16,
        "side":          4,
        "label":         "Hangzhou 4×4 (real)",
        "train_domain":  False,
    },
    "jinan_3x4": {
        "roadnet":       "data/jinan_3x4/roadnet.json",
        "flow":          "data/jinan_3x4/flow.json",
        "n":             12,
        "side":          4,
        "label":         "Jinan 3×4 (real)",
        "train_domain":  False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-scenario evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scenario(
    net_name:   str,
    net_cfg:    dict,
    gnn_model:  torch.nn.Module,
    mlp_model:  torch.nn.Module,
    device:     torch.device,
    cfg:        dict,
    n_seeds:    int = 3,
) -> list[dict]:
    """
    Run GNN, MLP, and baselines on one network for n_seeds seeds.
    Returns list of result dicts.
    """
    results = []
    N       = net_cfg["n"]

    # Check if data exists
    if not os.path.exists(net_cfg["roadnet"]):
        print(f"  ⚠ {net_name}: data not found at {net_cfg['roadnet']} — skipping")
        return []

    # Build graph template
    try:
        graph, tl_ids = build_graph_from_roadnet(net_cfg["roadnet"])
        graph = graph.to(device)
        print(f"  ✓ Graph: {graph.num_nodes} nodes, {graph.num_edges} edges "
              f"(irregular: {graph.num_nodes != net_cfg['n']})")
    except Exception as e:
        side  = net_cfg["side"]
        graph = build_grid_edge_index(n=N, side=side).to(device)
        tl_ids = [f"I{i}" for i in range(N)]
        print(f"  ⚠ Fallback to synthetic grid ({e})")

    for seed in range(n_seeds):
        config_path = make_cityflow_config(
            roadnet_path = net_cfg["roadnet"],
            flow_path    = net_cfg["flow"],
            save_dir     = f"data/gen_eval/{net_name}/seed{seed}",
            seed         = seed * 42,
        )
        env = CityFlowEnv(
            config_path       = config_path,
            num_intersections = N,
            episode_seconds   = cfg["episode_seconds"],
            delta_time        = cfg["delta_time"],
            reward_fn         = "queue",
        )

        # ── Run each agent ────────────────────────────────────────────────────
        for agent_name, policy_fn in _build_policies(
            gnn_model, mlp_model, graph, tl_ids, env, device, N
        ).items():
            obs, _ = env.reset()
            total_r = 0.0
            queues  = []
            done    = False

            while not done:
                actions = policy_fn(obs)
                obs, r, done, _, info = env.step(actions)
                total_r += r
                queues.append(info.get("avg_queue", 0.0))

            results.append({
                "network":        net_name,
                "label":          net_cfg["label"],
                "train_domain":   net_cfg["train_domain"],
                "agent":          agent_name,
                "seed":           seed,
                "avg_travel_time":env.avg_travel_time,
                "avg_queue":      float(np.mean(queues)),
                "total_reward":   total_r,
                "n_intersections":N,
            })

        env.close()

    return results


def _build_policies(gnn_model, mlp_model, graph, tl_ids, env, device, N) -> dict:
    """Build all policy callables for one scenario."""
    ei = graph.edge_index
    ea = graph.edge_attr

    def _gnn_policy(obs):
        g = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
        with torch.no_grad():
            q = gnn_model(g.x.to(device), ei, ea)
        return q.argmax(dim=1).cpu().numpy()

    def _mlp_policy(obs):
        g = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
        with torch.no_grad():
            q = mlp_model(g.x.to(device), ei, ea)
        return q.argmax(dim=1).cpu().numpy()

    fixed  = FixedTimeBaseline(N)
    maxp   = MaxPressureBaseline(N)

    return {
        "GNN-DQN":     _gnn_policy,
        "MLP-DQN":     _mlp_policy,
        "Fixed-time":  lambda obs: fixed.act(obs=obs, env=env),
        "Max-Pressure":lambda obs: maxp.act(obs=obs, env=env),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization_bars(df: pd.DataFrame, save_path: str = "results/generalization.png"):
    """
    Grouped bar chart: avg travel time per agent × network.
    Highlights the train domain network with a different colour.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    agents   = df["agent"].unique()
    networks = df["label"].unique()
    n_nets   = len(networks)
    n_agents = len(agents)

    summary = (
        df.groupby(["label", "agent"])["avg_travel_time"]
        .agg(["mean", "std"])
        .reset_index()
    )

    x      = np.arange(n_nets)
    width  = 0.7 / n_agents
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))

    fig, ax = plt.subplots(figsize=(max(10, n_nets * 2.5), 6))

    for i, agent in enumerate(agents):
        agent_data = summary[summary["agent"] == agent]
        means = [agent_data[agent_data["label"] == net]["mean"].values[0]
                 if len(agent_data[agent_data["label"] == net]) else 0
                 for net in networks]
        stds  = [agent_data[agent_data["label"] == net]["std"].values[0]
                 if len(agent_data[agent_data["label"] == net]) else 0
                 for net in networks]
        offset = (i - n_agents / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width * 0.9,
                        label=agent, color=colors[i],
                        yerr=stds, capsize=3, alpha=0.85)

    # Mark training domain
    for j, net_label in enumerate(networks):
        net_rows = df[df["label"] == net_label]
        if net_rows["train_domain"].any():
            ax.axvspan(j - 0.4, j + 0.4, alpha=0.08, color="green",
                       label="Train domain" if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Avg Travel Time (seconds/vehicle)", fontsize=11)
    ax.set_title("Zero-shot Generalization: GNN vs MLP vs Baselines",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved generalization plot → {save_path}")


def plot_generalization_table(df: pd.DataFrame, save_path: str = "results/generalization_table.png"):
    """Render the results as a styled table image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pivot = (
        df.groupby(["label", "agent"])["avg_travel_time"]
        .mean()
        .round(1)
        .unstack("agent")
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), len(pivot) * 0.7 + 1))
    ax.axis("off")

    table = ax.table(
        cellText   = pivot.values,
        rowLabels  = pivot.index.tolist(),
        colLabels  = pivot.columns.tolist(),
        cellLoc    = "center",
        loc        = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Colour the GNN-DQN column green if it is the best
    for row_idx in range(len(pivot)):
        row_vals    = pivot.iloc[row_idx].values
        best_col    = np.argmin(row_vals)
        col_labels  = list(pivot.columns)
        gnn_col_idx = col_labels.index("GNN-DQN") if "GNN-DQN" in col_labels else -1
        for col_idx in range(len(col_labels)):
            cell = table[row_idx + 1, col_idx]
            if col_idx == gnn_col_idx:
                cell.set_facecolor("#d4edda")   # green: our model
            if col_idx == best_col:
                cell.set_text_props(fontweight="bold")

    ax.set_title("Avg Travel Time (s/vehicle) — lower is better",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved generalization table → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # ── Load GNN model ────────────────────────────────────────────────────────
    gnn_model = GNN_QNetwork(
        state_dim  = STATE_DIM,
        embed_dim  = cfg["embed_dim"],
        num_phases = NUM_PHASES,
        gat_heads  = cfg["gat_heads"],
        hidden_dim = cfg["hidden_dim"],
    ).to(device)
    gnn_model.eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        gnn_model.load_state_dict(ckpt.get("model", ckpt))
        print(f"✓ Loaded GNN checkpoint: {args.checkpoint}")
    else:
        print("⚠ No checkpoint — using random GNN weights (for testing only)")

    # ── Load MLP ablation model ───────────────────────────────────────────────
    mlp_model = MLP_QNetwork(aggregate="mean").to(device)
    mlp_model.eval()

    if args.mlp_checkpoint and os.path.exists(args.mlp_checkpoint):
        ckpt = torch.load(args.mlp_checkpoint, map_location=device)
        mlp_model.load_state_dict(ckpt.get("model", ckpt))
        print(f"✓ Loaded MLP checkpoint: {args.mlp_checkpoint}")

    # ── Run all scenarios ─────────────────────────────────────────────────────
    all_results = []
    for net_name, net_cfg in NETWORKS.items():
        print(f"\n── {net_cfg['label']} ──────────────────────────────────")
        results = evaluate_scenario(
            net_name   = net_name,
            net_cfg    = net_cfg,
            gnn_model  = gnn_model,
            mlp_model  = mlp_model,
            device     = device,
            cfg        = cfg,
            n_seeds    = args.num_seeds,
        )
        all_results.extend(results)

    if not all_results:
        print("\n⚠ No results collected. Ensure at least one network's data is available.")
        return

    # ── Save and plot ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv("results/generalization.csv", index=False)
    print(f"\n✓ Full results saved → results/generalization.csv")

    summary = (
        df.groupby(["label", "agent"])[["avg_travel_time", "avg_queue"]]
        .mean()
        .round(2)
    )
    print("\n── Generalization results (mean across seeds) ─────────────────")
    print(summary.to_string())

    plot_generalization_bars(df)
    plot_generalization_table(df)
    print("\n✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=None, help="GNN model checkpoint .pt")
    parser.add_argument("--mlp-checkpoint", default=None, help="MLP model checkpoint .pt")
    parser.add_argument("--num-seeds",      type=int, default=3)
    args = parser.parse_args()
    main(args)
