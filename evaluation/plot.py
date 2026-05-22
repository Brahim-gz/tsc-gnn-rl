# evaluation/plots.py
# ============================================================
# Shared plotting utilities used by both eval.py and
# attention_viz.py. Produces the final comparison figures
# used in the write-up.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


AGENT_COLORS = {
    "Random":        "#d62728",
    "Fixed-time":    "#ff7f0e",
    "Max-Pressure":  "#9467bd",
    "MLP-DQN":       "#8c564b",
    "GNN-DQN":       "#2ca02c",
    "GNN-PPO":       "#1f77b4",
}

AGENT_ORDER = [
    "Random", "Fixed-time", "Max-Pressure",
    "MLP-DQN", "GNN-DQN", "GNN-PPO",
]


def plot_learning_curves(
    log_path:  str = "logs/",
    save_path: str = "results/learning_curves.png",
):
    """
    Plot reward and travel time learning curves for all runs.
    Reads WandB-exported CSV files from logs/.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csvs = [f for f in os.listdir(log_path) if f.endswith(".csv")] if os.path.exists(log_path) else []

    if not csvs:
        print("⚠ No log CSV files found in logs/. Export from WandB first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for csv_file in csvs:
        df   = pd.read_csv(os.path.join(log_path, csv_file))
        name = csv_file.replace(".csv", "")
        color = AGENT_COLORS.get(name, "gray")

        if "episode_reward" in df.columns and "step" in df.columns:
            # Smooth with rolling mean
            smoothed = df["episode_reward"].rolling(window=10, min_periods=1).mean()
            ax1.plot(df["step"], smoothed, label=name, color=color, linewidth=1.8)

        if "avg_travel_time" in df.columns and "step" in df.columns:
            smoothed = df["avg_travel_time"].rolling(window=10, min_periods=1).mean()
            ax2.plot(df["step"], smoothed, label=name, color=color, linewidth=1.8)

    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Episode reward (higher = better)", fontsize=11)
    ax1.set_title("Reward learning curve", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Avg travel time (s) — lower = better", fontsize=11)
    ax2.set_title("Travel time during training", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved learning curves → {save_path}")


def plot_final_comparison(
    metrics_csv: str = "results/metrics.csv",
    save_path:   str = "results/final_comparison.png",
):
    """
    Grouped bar chart comparing all agents on all 4 metrics.
    Uses the output of evaluation/eval.py.
    """
    if not os.path.exists(metrics_csv):
        print(f"⚠ {metrics_csv} not found. Run eval.py first.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df      = pd.read_csv(metrics_csv)
    agents  = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    metrics = {
        "avg_travel_time": ("Avg Travel Time (s)", False),   # lower is better
        "avg_queue":       ("Avg Queue Length",    False),
        "total_reward":    ("Total Reward",        True),    # higher is better
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    x         = np.arange(len(agents))
    width     = 0.6

    for ax, (metric, (ylabel, higher_better)) in zip(axes, metrics.items()):
        means, stds = [], []
        for agent in agents:
            sub   = df[df["agent"] == agent][metric]
            means.append(sub.mean())
            stds.append(sub.std())

        colors = [AGENT_COLORS.get(a, "gray") for a in agents]
        bars   = ax.bar(x, means, width, yerr=stds, color=colors,
                        capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Highlight best bar
        best_idx = int(np.argmin(means) if not higher_better else np.argmax(means))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{'↓ Lower' if not higher_better else '↑ Higher'} is better",
                     fontsize=9, color="gray")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Final Evaluation: All Agents — 4×4 Grid (mean ± std, 5 seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved final comparison → {save_path}")


def plot_queue_evolution(
    queue_data: dict,   # {agent_name: list of avg_queue per step}
    save_path:  str = "results/queue_evolution.png",
):
    """
    Line plot of average queue length over one episode for each agent.
    Useful for seeing how agents respond to traffic buildup.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    for agent, queues in queue_data.items():
        color = AGENT_COLORS.get(agent, "gray")
        steps = np.arange(len(queues)) * 10    # multiply by delta_time
        ax.plot(steps, queues, label=agent, color=color, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Simulation time (seconds)", fontsize=11)
    ax.set_ylabel("Avg queue length (vehicles/lane)", fontsize=11)
    ax.set_title("Queue evolution during episode — lower is better",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved queue evolution → {save_path}")
