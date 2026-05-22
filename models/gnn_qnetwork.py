# models/gnn_qnetwork.py
# ============================================================
# GNN Q-Network for DQN training.
# Wraps GATEncoder with an MLP Q-value head.
#
# Interface contract (from interfaces.py):
#     q_values = model.forward(x, edge_index, edge_attr)
#     returns  → Tensor[N, NUM_PHASES]
#
# Person A calls this from training/train_single.py
# and training/train_multi.py without knowing the internals.
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, EDGE_DIM, DEFAULT_CONFIG
from models.gat_encoder import GATEncoder


class GNN_QNetwork(nn.Module):
    """
    GATEncoder → MLP Q-head.

    Produces one Q-value per phase per intersection.
    Used by DQN: action = argmax over Q-values.

    Parameters
    ----------
    state_dim  : dimension of the input node feature vector (STATE_DIM = 8)
    embed_dim  : GATEncoder output dimension (EMBED_DIM = 64)
    num_phases : number of discrete action choices (NUM_PHASES = 4)
    gat_heads  : number of GAT attention heads
    hidden_dim : MLP hidden layer width
    edge_dim   : edge feature dimension (EDGE_DIM = 2)
    dropout    : dropout rate (applied in encoder and Q-head)
    dueling    : if True, uses Dueling DQN architecture (V + A streams)
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        embed_dim:  int   = EMBED_DIM,
        num_phases: int   = NUM_PHASES,
        gat_heads:  int   = DEFAULT_CONFIG["gat_heads"],
        hidden_dim: int   = DEFAULT_CONFIG["hidden_dim"],
        edge_dim:   int   = EDGE_DIM,
        dropout:    float = DEFAULT_CONFIG["dropout"],
        dueling:    bool  = True,
    ):
        super().__init__()

        self.num_phases = num_phases
        self.dueling    = dueling

        # ── Shared GNN backbone ───────────────────────────────────────────────
        self.encoder = GATEncoder(
            in_dim     = state_dim,
            hidden_dim = hidden_dim,
            out_dim    = embed_dim,
            heads      = gat_heads,
            edge_dim   = edge_dim,
            dropout    = dropout,
        )

        if dueling:
            # ── Dueling DQN: separate Value and Advantage streams ─────────────
            # V(s): scalar state value
            self.value_stream = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            # A(s, a): advantage for each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_phases),
            )
        else:
            # ── Standard DQN: single Q-head ───────────────────────────────────
            self.q_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_phases),
            )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for the MLP heads (improves DQN stability)."""
        def _ortho(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        if self.dueling:
            self.value_stream.apply(_ortho)
            self.advantage_stream.apply(_ortho)
        else:
            self.q_head.apply(_ortho)

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
        edge_index : Tensor[2, E]
        edge_attr  : Tensor[E, edge_dim]   (optional)

        Returns
        -------
        q_values : Tensor[N, num_phases]
        """
        emb = self.encoder(x, edge_index, edge_attr)   # [N, embed_dim]

        if self.dueling:
            value     = self.value_stream(emb)          # [N, 1]
            advantage = self.advantage_stream(emb)      # [N, num_phases]
            # Q = V + (A - mean(A))  — subtract mean to keep identifiability
            q_values  = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_head(emb)                 # [N, num_phases]

        return q_values                                 # [N, num_phases]

    def get_embeddings(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> torch.Tensor:
        """Return raw GNN embeddings (for analysis / visualization)."""
        with torch.no_grad():
            return self.encoder(x, edge_index, edge_attr)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.graph_builder import build_grid_edge_index

    print("── GNN_QNetwork unit test ──────────────────────────────")

    graph = build_grid_edge_index(n=16, side=4)
    x     = torch.randn(16, STATE_DIM)
    ei    = graph.edge_index
    ea    = graph.edge_attr

    # Standard DQN
    q_net  = GNN_QNetwork(dueling=False)
    q_out  = q_net(x, ei, ea)
    assert q_out.shape == (16, NUM_PHASES), f"Got {q_out.shape}"
    print(f"✓ Standard DQN Q-values:  {q_out.shape}  expected (16, {NUM_PHASES})")

    # Dueling DQN
    q_dual = GNN_QNetwork(dueling=True)
    q_out2 = q_dual(x, ei, ea)
    assert q_out2.shape == (16, NUM_PHASES)
    print(f"✓ Dueling DQN Q-values:   {q_out2.shape}")

    # Gradient flow
    loss = q_out2.sum()
    loss.backward()
    print(f"✓ Gradients flow:         {all(p.grad is not None for p in q_dual.parameters())}")

    # Embeddings
    emb = q_dual.get_embeddings(x, ei, ea)
    assert emb.shape == (16, EMBED_DIM)
    print(f"✓ Embeddings:             {emb.shape}  expected (16, {EMBED_DIM})")

    print(f"\nTotal parameters (dueling): {q_dual.count_parameters():,}")
    print("── All tests passed ────────────────────────────────────")
