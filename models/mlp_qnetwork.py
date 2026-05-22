# models/mlp_qnetwork.py
# ============================================================
# MLP Q-Network — ablation baseline with NO graph structure.
# Same forward() interface as GNN_QNetwork so the
# training loop can swap it in with zero code changes.
#
# Purpose: isolate the GNN's contribution.
# If GNN_QNetwork >> MLP_QNetwork → the graph structure helps.
# If they are equal → GNN adds no value on this topology.
#
# Parameter budget is matched to GNN_QNetwork so the
# comparison is fair (same model capacity).
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, EDGE_DIM, DEFAULT_CONFIG


class MLP_QNetwork(nn.Module):
    """
    Flat MLP Q-Network — no message passing, no graph structure.

    Each intersection agent acts based only on its own local state
    and a hand-crafted aggregation of its neighbours (mean pooling).
    No attention, no learnable neighbourhood weighting.

    The GNN replaces this hand-crafted aggregation with a learned one.

    Interface (identical to GNN_QNetwork):
        q_values = model.forward(x, edge_index, edge_attr)
        returns  → Tensor[N, NUM_PHASES]

    edge_index is accepted (but not used for learning) so Person A's
    training loop doesn't need any changes to run this ablation.

    Parameters
    ----------
    state_dim       : input feature dim per node (= STATE_DIM = 8)
    n_neighbors     : max neighbours to aggregate (hand-coded, not learned)
    hidden_dim      : MLP hidden width
    num_phases      : output action dimension
    aggregate       : "mean" | "max" | "none"  — how to use neighbour info
    """

    def __init__(
        self,
        state_dim:   int  = STATE_DIM,
        n_neighbors: int  = 4,          # 4-connected grid
        hidden_dim:  int  = DEFAULT_CONFIG["hidden_dim"],
        num_phases:  int  = NUM_PHASES,
        aggregate:   str  = "mean",     # "mean" | "max" | "none"
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.aggregate   = aggregate
        self.n_neighbors = n_neighbors

        # Input dim: own state + (aggregated neighbour state if aggregate != "none")
        if aggregate == "none":
            in_dim = state_dim
        else:
            in_dim = state_dim * 2      # own state ++ mean-neighbour state

        # MLP body — parameter count tuned to match GATEncoder
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_phases),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : Tensor[N, state_dim]
        edge_index : Tensor[2, E]           — used ONLY for manual aggregation
        edge_attr  : Tensor[E, edge_dim]    — ignored

        Returns
        -------
        q_values : Tensor[N, num_phases]
        """
        if self.aggregate == "none":
            # Fully local: each agent sees only its own state
            features = x

        else:
            # Manual neighbourhood aggregation (no learnable weights)
            N            = x.size(0)
            neighbour_agg = self._aggregate_neighbours(x, edge_index, N)  # [N, state_dim]

            # Concatenate own state with aggregated neighbour state
            features = torch.cat([x, neighbour_agg], dim=-1)              # [N, state_dim*2]

        return self.net(features)                                          # [N, num_phases]

    def _aggregate_neighbours(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        N:          int,
    ) -> torch.Tensor:
        """
        Simple non-learnable neighbour aggregation.
        This is what a GNN does — but here the weights are hand-coded (uniform).
        """
        src, dst = edge_index[0], edge_index[1]

        # Accumulate
        agg = torch.zeros_like(x)        # [N, state_dim]
        cnt = torch.zeros(N, 1, device=x.device, dtype=x.dtype)

        # Scatter add: for each edge (src → dst), add src features to dst
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, device=x.device))

        if self.aggregate == "mean":
            return agg / (cnt + 1e-8)
        elif self.aggregate == "max":
            # Max aggregation (scatter_reduce — requires PyTorch >= 1.12)
            max_agg = torch.full_like(x, float('-inf'))
            max_agg.scatter_reduce_(
                0, dst.unsqueeze(1).expand(-1, x.size(1)),
                x[src], reduce="amax", include_self=True
            )
            return max_agg
        else:
            return agg

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"MLP_QNetwork("
            f"aggregate={self.aggregate}, "
            f"params={self.count_parameters():,})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit test + parameter budget comparison
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.graph_builder import build_grid_edge_index
    from models.gnn_qnetwork import GNN_QNetwork

    print("── MLP_QNetwork unit test ──────────────────────────────")

    graph = build_grid_edge_index(n=16, side=4)
    x     = torch.randn(16, STATE_DIM)
    ei    = graph.edge_index
    ea    = graph.edge_attr

    for agg in ["none", "mean", "max"]:
        mlp  = MLP_QNetwork(aggregate=agg)
        out  = mlp(x, ei, ea)
        assert out.shape == (16, NUM_PHASES), f"aggregate={agg}: {out.shape}"
        print(f"✓ aggregate='{agg}':  output={out.shape}  params={mlp.count_parameters():,}")

    # ── Gradient flow ─────────────────────────────────────────────────────────
    mlp  = MLP_QNetwork(aggregate="mean")
    out  = mlp(x, ei, ea)
    out.sum().backward()
    print(f"✓ Gradient flow:       {all(p.grad is not None for p in mlp.parameters())}")

    # ── Parameter count comparison ────────────────────────────────────────────
    gnn  = GNN_QNetwork(dueling=True)
    mlp2 = MLP_QNetwork(aggregate="mean")
    print(f"\nParameter count comparison:")
    print(f"  GNN_QNetwork (dueling): {gnn.count_parameters():>8,}")
    print(f"  MLP_QNetwork (mean):    {mlp2.count_parameters():>8,}")
    print(f"  Ratio:                  {gnn.count_parameters()/mlp2.count_parameters():.2f}×")

    print("\n── All tests passed ────────────────────────────────────")
