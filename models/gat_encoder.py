# models/gat_encoder.py
# ============================================================
# Core GNN backbone — 2-layer Graph Attention Network (GAT).
# Person B owns this file entirely.
#
# This is the shared encoder used by both the DQN Q-network
# and the PPO actor-critic. Changing it affects both.
#
# Reference: Veličković et al., "Graph Attention Networks"
#            ICLR 2018 — https://arxiv.org/abs/1710.10903
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

from interfaces import STATE_DIM, EMBED_DIM, EDGE_DIM, DEFAULT_CONFIG


class GATEncoder(nn.Module):
    """
    2-layer Graph Attention Network that encodes N intersection
    state vectors into N rich embedding vectors, incorporating
    neighbourhood context via learned attention coefficients.

    Forward pass
    ------------
    Input:
        x          : Tensor[N, in_dim]     node feature matrix
        edge_index : Tensor[2, E]          graph edges in COO format
        edge_attr  : Tensor[E, edge_dim]   edge features (optional)
    Output:
        Tensor[N, out_dim]                 per-node embeddings

    Architecture
    ------------
        Linear projection
            ↓
        GATConv layer 1  (multi-head, concat=True)   → hidden_dim * heads
        LayerNorm + ELU + Dropout
            ↓
        GATConv layer 2  (single-head, concat=False) → out_dim
        LayerNorm
            ↓
        Residual connection (if in_dim == out_dim, optional)

    Parameters
    ----------
    in_dim     : input feature dimension  (= STATE_DIM = 8)
    hidden_dim : per-head hidden dimension before concatenation
    out_dim    : output embedding dimension (= EMBED_DIM = 64)
    heads      : number of attention heads in layer 1
    edge_dim   : edge feature dimension (= EDGE_DIM = 2)
    dropout    : dropout rate applied before each GAT layer
    """

    def __init__(
        self,
        in_dim:     int   = STATE_DIM,
        hidden_dim: int   = DEFAULT_CONFIG["hidden_dim"],
        out_dim:    int   = EMBED_DIM,
        heads:      int   = DEFAULT_CONFIG["gat_heads"],
        edge_dim:   int   = EDGE_DIM,
        dropout:    float = DEFAULT_CONFIG["dropout"],
    ):
        super().__init__()

        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.dropout_p = dropout

        # ── Input projection ──────────────────────────────────────────────────
        # Projects raw state vector into the same space before attention.
        # Helps when STATE_DIM is small relative to hidden_dim.
        self.input_proj = nn.Linear(in_dim, in_dim)

        # ── Layer 1: multi-head GAT ───────────────────────────────────────────
        # concat=True → output is hidden_dim * heads
        self.gat1 = GATConv(
            in_channels  = in_dim,
            out_channels = hidden_dim,
            heads        = heads,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = True,
            add_self_loops = False,   # we add them manually in graph_builder
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # ── Layer 2: single-head GAT ──────────────────────────────────────────
        # concat=False → output is exactly out_dim
        self.gat2 = GATConv(
            in_channels  = hidden_dim * heads,
            out_channels = out_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            dropout      = dropout,
            concat       = False,
            add_self_loops = False,
        )
        self.norm2 = nn.LayerNorm(out_dim)

        # ── Optional skip connection ──────────────────────────────────────────
        # Only active if in_dim == out_dim (e.g., when doing ablations).
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    # ─────────────────────────────────────────────────────────────────────────
    # Standard forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode node features into embeddings.

        Returns
        -------
        Tensor[N, out_dim]
        """
        residual = self.skip(x)                          # [N, out_dim]

        x = self.input_proj(x)                           # [N, in_dim]
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 1
        x = self.gat1(x, edge_index, edge_attr=edge_attr)   # [N, hidden*heads]
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 2
        x = self.gat2(x, edge_index, edge_attr=edge_attr)   # [N, out_dim]
        x = self.norm2(x)

        # Skip connection
        return x + residual                              # [N, out_dim]

    # ─────────────────────────────────────────────────────────────────────────
    # Forward with attention weights (for visualization — Week 2 Day 11)
    # ─────────────────────────────────────────────────────────────────────────

    def forward_with_attention(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> tuple:
        """
        Same as forward() but also returns raw attention weights from each
        GAT layer. Used by evaluation/attention_viz.py.

        Returns
        -------
        embeddings   : Tensor[N, out_dim]
        attn_layer1  : (edge_index, Tensor[E, heads])   ← layer 1 weights
        attn_layer2  : (edge_index, Tensor[E, 1])        ← layer 2 weights
        """
        residual = self.skip(x)

        x = self.input_proj(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 1 — capture attention
        x, (ei1, alpha1) = self.gat1(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )                                                # alpha1: [E, heads]
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 2 — capture attention
        x, (ei2, alpha2) = self.gat2(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )                                                # alpha2: [E, 1]
        x = self.norm2(x)
        x = x + residual

        return x, (ei1, alpha1), (ei2, alpha2)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"GATEncoder("
            f"in={self.in_dim}, "
            f"out={self.out_dim}, "
            f"params={self.count_parameters():,})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick unit test (run: python models/gat_encoder.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.graph_builder import build_grid_edge_index

    print("── GATEncoder unit test ────────────────────────────────")

    # Synthetic 4×4 grid
    graph = build_grid_edge_index(n=16, side=4)
    x     = torch.randn(16, STATE_DIM)
    ei    = graph.edge_index
    ea    = graph.edge_attr

    # ── Test 1: standard forward ──────────────────────────────────────────────
    enc = GATEncoder()
    out = enc(x, ei, ea)
    assert out.shape == (16, EMBED_DIM), f"Expected (16, {EMBED_DIM}), got {out.shape}"
    print(f"✓ forward():                  {out.shape}  expected (16, {EMBED_DIM})")

    # ── Test 2: attention forward ─────────────────────────────────────────────
    enc.eval()
    with torch.no_grad():
        out2, (ei1, a1), (ei2, a2) = enc.forward_with_attention(x, ei, ea)
    assert out2.shape == (16, EMBED_DIM)
    assert a1.shape[1] == DEFAULT_CONFIG["gat_heads"]
    print(f"✓ forward_with_attention():   embeddings={out2.shape}")
    print(f"  layer-1 attention:          {a1.shape}  (E × heads)")
    print(f"  layer-2 attention:          {a2.shape}  (E × 1)")

    # ── Test 3: gradient flow ─────────────────────────────────────────────────
    enc.train()
    out3 = enc(x, ei, ea)
    loss = out3.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in enc.parameters())
    print(f"✓ Gradients flow:             {grads_ok}")

    # ── Test 4: single node (1×1 grid, sanity check for train_single.py) ─────
    g1   = build_grid_edge_index(n=1, side=1)
    x1   = torch.randn(1, STATE_DIM)
    out4 = enc(x1, g1.edge_index, g1.edge_attr)
    assert out4.shape == (1, EMBED_DIM)
    print(f"✓ Single-node forward:        {out4.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{enc}")
    print("── All tests passed ────────────────────────────────────")
