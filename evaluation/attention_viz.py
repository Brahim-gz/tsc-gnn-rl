# evaluation/attention_viz.py
# ============================================================
# Visualise Graph Attention Network weights on the road network.
# Person B's Week 2 Day 11 deliverable.
#
# Produces:
#   - Heatmap: edge thickness/colour = attention weight
#   - Bar chart: top-K most attended edges per intersection
#   - Animation: attention weights evolving across timesteps
#
# Run:
#   python evaluation/attention_viz.py \
#       --checkpoint checkpoints/dqn_multi_final.pt \
#       --roadnet   data/synthetic_4x4/roadnet.json
# ============================================================

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — works on Kaggle and Colab
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, DEFAULT_CONFIG
from env.graph_builder import build_grid_edge_index, build_graph_from_roadnet
from models.gnn_qnetwork import GNN_QNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_weights(
    model:      GNN_QNetwork,
    x:          torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one forward pass and extract GAT attention weights.

    Returns
    -------
    src_nodes  : np.ndarray[E]          — source node of each edge
    dst_nodes  : np.ndarray[E]          — destination node of each edge
    alpha1     : np.ndarray[E, heads]   — layer-1 attention weights
    alpha2     : np.ndarray[E, 1]       — layer-2 attention weights
    """
    model.eval()
    x = x.to(device)
    ei = edge_index.to(device)
    ea = edge_attr.to(device) if edge_attr is not None else None

    with torch.no_grad():
        _, (ei1, a1), (ei2, a2) = model.encoder.forward_with_attention(x, ei, ea)

    return (
        ei1[0].cpu().numpy(),   # src
        ei1[1].cpu().numpy(),   # dst
        a1.cpu().numpy(),       # [E, heads]
        a2.cpu().numpy(),       # [E, 1]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_layout(n: int, side: int) -> np.ndarray:
    """Returns (n, 2) array of (x, y) positions for a grid layout."""
    pos = np.zeros((n, 2))
    for i in range(n):
        r, c = divmod(i, side)
        pos[i] = [c, side - 1 - r]    # flip row so row 0 is at top
    return pos


def roadnet_layout(roadnet_path: str, tl_ids: list) -> np.ndarray:
    """
    Extract real (x, y) positions from CityFlow roadnet.json.
    Falls back to grid_layout if positions are unavailable.
    """
    import json
    with open(roadnet_path) as f:
        rn = json.load(f)

    pos_map = {}
    for inter in rn.get("intersections", []):
        pt = inter.get("point", {})
        pos_map[inter["id"]] = [pt.get("x", 0), pt.get("y", 0)]

    positions = np.array([pos_map.get(iid, [0, 0]) for iid in tl_ids], dtype=float)

    # Normalize to [0, 1] range
    if positions.max() > 1:
        positions -= positions.min(axis=0)
        scale = positions.max(axis=0)
        scale[scale == 0] = 1
        positions /= scale

    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Attention heatmap on road network
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    src_nodes:  np.ndarray,
    dst_nodes:  np.ndarray,
    alpha:      np.ndarray,     # [E, heads] or [E, 1]
    positions:  np.ndarray,     # [N, 2]
    tl_ids:     list,
    title:      str  = "GAT Attention Weights — Layer 1",
    save_path:  str  = "results/attention_heatmap.png",
    layer:      int  = 1,
):
    """
    Draws the road network graph with edges coloured and thickened
    by their attention weight (averaged across attention heads).

    Darker / thicker edge = the destination intersection attends
    more strongly to its neighbour at the source.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Average over heads
    alpha_mean = alpha.mean(axis=-1)     # [E]

    # Normalise to [0, 1] for colour mapping
    a_min, a_max = alpha_mean.min(), alpha_mean.max()
    if a_max > a_min:
        alpha_norm = (alpha_mean - a_min) / (a_max - a_min)
    else:
        alpha_norm = np.ones_like(alpha_mean) * 0.5

    cmap   = cm.YlOrRd
    N      = len(tl_ids)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Draw edges ────────────────────────────────────────────────────────────
    for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
        if s == d:
            continue   # skip self-loops

        xs, ys = positions[s]
        xd, yd = positions[d]
        w      = alpha_norm[i]
        colour = cmap(w)
        lw     = 0.5 + w * 4.5   # linewidth: 0.5 (low attn) → 5.0 (high attn)

        ax.annotate(
            "", xy=(xd, yd), xytext=(xs, ys),
            arrowprops=dict(
                arrowstyle="-|>",
                color=colour,
                lw=lw,
                mutation_scale=10 + w * 8,
            ),
        )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for i, (px, py) in enumerate(positions):
        ax.scatter(px, py, s=350, color="#4a9fd4", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(px, py, str(i), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=6)

    # ── Colour bar ────────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=a_min, vmax=a_max)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(f"Attention weight (avg across {alpha.shape[1]} head(s))",
                   color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=14, pad=16, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"✓ Saved attention heatmap → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Top-K attended neighbours per intersection
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_k_attention(
    src_nodes:  np.ndarray,
    dst_nodes:  np.ndarray,
    alpha:      np.ndarray,    # [E, heads]
    tl_ids:     list,
    k:          int  = 5,
    save_path:  str  = "results/attention_topk.png",
):
    """
    For each intersection, bar-chart the top-K source nodes it attends to most.
    One subplot per intersection (shows first 16 intersections).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N          = len(tl_ids)
    alpha_mean = alpha.mean(axis=-1)    # [E]
    n_show     = min(N, 16)
    ncols      = 4
    nrows      = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5))
    axes = axes.flatten()

    for node_idx in range(n_show):
        # Find all edges where dst == node_idx
        mask = dst_nodes == node_idx
        srcs = src_nodes[mask]
        atts = alpha_mean[mask]

        # Sort by attention (descending) and take top-k
        order = np.argsort(atts)[::-1][:k]
        srcs  = srcs[order]
        atts  = atts[order]

        ax = axes[node_idx]
        labels = [f"node {s}" for s in srcs]
        colors = plt.cm.YlOrRd(atts / (atts.max() + 1e-8))

        bars = ax.barh(labels, atts, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(f"Intersection {node_idx}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Attention", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, alpha_mean.max() * 1.1)
        ax.invert_yaxis()

    # Hide unused subplots
    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Top-K attended neighbours per intersection (Layer 1)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved top-K attention → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Attention per layer comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_comparison(
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    alpha1:    np.ndarray,   # [E, heads]
    alpha2:    np.ndarray,   # [E, 1]
    positions: np.ndarray,
    tl_ids:    list,
    save_path: str = "results/attention_layers.png",
):
    """
    Side-by-side comparison of layer 1 and layer 2 attention patterns.
    Reveals how deeper layers focus attention differently.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, alpha, lbl in zip(axes, [alpha1, alpha2], ["Layer 1", "Layer 2"]):
        alpha_m = alpha.mean(axis=-1)
        a_norm  = (alpha_m - alpha_m.min()) / (alpha_m.max() - alpha_m.min() + 1e-8)
        cmap    = cm.plasma

        ax.set_facecolor("#f5f5f5")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Attention Weights — {lbl}", fontsize=13, fontweight="bold", pad=12)

        for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
            if s == d:
                continue
            xs, ys = positions[s]
            xd, yd = positions[d]
            w      = a_norm[i]
            ax.plot([xs, xd], [ys, yd],
                    color=cmap(w), linewidth=0.5 + w * 4, alpha=0.7, zorder=1)

        for i, (px, py) in enumerate(positions):
            ax.scatter(px, py, s=250, color="#2563EB", zorder=3,
                       edgecolors="white", linewidths=1)
            ax.text(px, py, str(i), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved layer comparison → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────────
    try:
        graph, tl_ids = build_graph_from_roadnet(args.roadnet)
        positions = roadnet_layout(args.roadnet, tl_ids)
        print(f"✓ Loaded roadnet: {graph.num_nodes} intersections")
    except Exception as e:
        print(f"⚠ Roadnet parse failed ({e}). Using synthetic 4×4 grid.")
        side  = args.side
        n     = side * side
        graph = build_grid_edge_index(n=n, side=side)
        tl_ids    = [f"I{i}" for i in range(n)]
        positions = grid_layout(n, side)

    # ── Load model ────────────────────────────────────────────────────────────
    model = GNN_QNetwork(
        state_dim  = STATE_DIM,
        embed_dim  = cfg["embed_dim"],
        num_phases = NUM_PHASES,
        gat_heads  = cfg["gat_heads"],
        hidden_dim = cfg["hidden_dim"],
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        print(f"✓ Loaded checkpoint: {args.checkpoint}")
    else:
        print("⚠ No checkpoint found — using random initialisation (for testing)")

    # ── Extract attention ─────────────────────────────────────────────────────
    N  = graph.num_nodes
    x  = torch.randn(N, STATE_DIM)   # random state for visualisation
    ei = graph.edge_index
    ea = graph.edge_attr

    src, dst, alpha1, alpha2 = extract_attention_weights(model, x, ei, ea, device)
    print(f"✓ Attention extracted: {len(src)} edges, "
          f"{alpha1.shape[1]} heads (layer 1), {alpha2.shape[1]} head (layer 2)")

    # ── Generate plots ────────────────────────────────────────────────────────
    plot_attention_heatmap(
        src, dst, alpha1, positions, tl_ids,
        title     = "GAT Attention Weights — Layer 1",
        save_path = "results/attention_layer1.png",
        layer     = 1,
    )
    plot_attention_heatmap(
        src, dst, alpha2, positions, tl_ids,
        title     = "GAT Attention Weights — Layer 2",
        save_path = "results/attention_layer2.png",
        layer     = 2,
    )
    plot_top_k_attention(
        src, dst, alpha1, tl_ids,
        k         = 5,
        save_path = "results/attention_topk.png",
    )
    plot_layer_comparison(
        src, dst, alpha1, alpha2, positions, tl_ids,
        save_path = "results/attention_layers.png",
    )

    print("\n✓ All visualisations saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--roadnet",    default="data/synthetic_4x4/roadnet.json")
    parser.add_argument("--side",       type=int, default=4)
    args = parser.parse_args()
    main(args)
