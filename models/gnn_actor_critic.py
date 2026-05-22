# models/gnn_actor_critic.py
# ============================================================
# GNN Actor-Critic for PPO training .
# Shares the GATEncoder backbone between the actor and critic.
#
# Interface contract (from interfaces.py):
#     action_logits, state_value = model.forward(x, edge_index, edge_attr)
#     action_logits → Tensor[N, NUM_PHASES]
#     state_value   → Tensor[N, 1]
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import STATE_DIM, EMBED_DIM, NUM_PHASES, EDGE_DIM, DEFAULT_CONFIG
from models.gat_encoder import GATEncoder


class GNN_ActorCritic(nn.Module):
    """
    Shared GATEncoder + separate Actor head + Critic head.

    The shared backbone learns a single spatial representation
    used by both policy and value estimation. This is standard
    practice in PPO and reduces the total parameter count.

    Architecture
    ------------
        x, edge_index, edge_attr
                ↓
          GATEncoder (shared)
                ↓
          ┌─────┴─────┐
        Actor        Critic
        (MLP)        (MLP)
          ↓             ↓
        logits       value
      [N, phases]    [N, 1]

    Parameters
    ----------
    state_dim      : input node feature dimension (STATE_DIM = 8)
    embed_dim      : GATEncoder output dimension  (EMBED_DIM = 64)
    num_phases     : action space size            (NUM_PHASES = 4)
    gat_heads      : number of GAT attention heads
    hidden_dim     : MLP hidden width for both heads
    edge_dim       : edge feature dimension       (EDGE_DIM = 2)
    dropout        : dropout in encoder only (heads use no dropout for stability)
    shared_encoder : if False, actor and critic each get their own GATEncoder
                     (costs more memory, sometimes more stable)
    """

    def __init__(
        self,
        state_dim:      int   = STATE_DIM,
        embed_dim:      int   = EMBED_DIM,
        num_phases:     int   = NUM_PHASES,
        gat_heads:      int   = DEFAULT_CONFIG["gat_heads"],
        hidden_dim:     int   = DEFAULT_CONFIG["hidden_dim"],
        edge_dim:       int   = EDGE_DIM,
        dropout:        float = DEFAULT_CONFIG["dropout"],
        shared_encoder: bool  = True,
    ):
        super().__init__()

        self.num_phases     = num_phases
        self.shared_encoder = shared_encoder

        encoder_kwargs = dict(
            in_dim     = state_dim,
            hidden_dim = hidden_dim,
            out_dim    = embed_dim,
            heads      = gat_heads,
            edge_dim   = edge_dim,
            dropout    = dropout,
        )

        if shared_encoder:
            # One encoder for both actor and critic
            self.encoder = GATEncoder(**encoder_kwargs)
        else:
            # Separate encoders (uncomment for ablation)
            self.actor_encoder  = GATEncoder(**encoder_kwargs)
            self.critic_encoder = GATEncoder(**encoder_kwargs)

        # ── Actor head ────────────────────────────────────────────────────────
        # Outputs un-normalised logits — softmax is applied during action sampling.
        # Using Tanh activations (more stable for policy gradients than ReLU).
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_phases),
        )

        # ── Critic head ───────────────────────────────────────────────────────
        # Outputs a scalar state-value estimate per node (agent).
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal init with small gain for the output layers.
        Standard practice from the PPO implementation (Schulman et al.)
        """
        def _ortho(module, gain=1.0):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

        # Encoder layers use default PyG init
        # Actor head: small gain on final layer (exploration)
        for i, layer in enumerate(self.actor_head):
            if isinstance(layer, nn.Linear):
                gain = 0.01 if i == len(self.actor_head) - 1 else 1.0
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

        # Critic head: standard gain
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    # ─────────────────────────────────────────────────────────────────────────
    # Core forward pass (called by training loop)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x          : Tensor[N, state_dim]
        edge_index : Tensor[2, E]
        edge_attr  : Tensor[E, edge_dim]   (optional)

        Returns
        -------
        action_logits : Tensor[N, num_phases]   — raw logits (before softmax)
        state_value   : Tensor[N, 1]            — critic value estimate
        """
        if self.shared_encoder:
            emb = self.encoder(x, edge_index, edge_attr)         # [N, embed_dim]
            return self.actor_head(emb), self.critic_head(emb)
        else:
            actor_emb  = self.actor_encoder(x, edge_index, edge_attr)
            critic_emb = self.critic_encoder(x, edge_index, edge_attr)
            return self.actor_head(actor_emb), self.critic_head(critic_emb)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods (called by training loop)
    # ─────────────────────────────────────────────────────────────────────────

    def get_action(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution.

        Returns
        -------
        actions   : Tensor[N]        — sampled (or greedy) phase indices
        log_probs : Tensor[N]        — log probability of chosen actions
        entropy   : Tensor[N]        — per-agent policy entropy
        """
        logits, _ = self.forward(x, edge_index, edge_attr)
        dist      = Categorical(logits=logits)              # [N] distributions

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()                        # [N]

        log_probs = dist.log_prob(actions)                 # [N]
        entropy   = dist.entropy()                         # [N]
        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        actions:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate previously taken actions (used in PPO update epochs).

        Parameters
        ----------
        actions : Tensor[N]   — actions taken during rollout collection

        Returns
        -------
        log_probs : Tensor[N]    — log π(a|s) for the given actions
        values    : Tensor[N, 1] — critic value estimates
        entropy   : Tensor[N]    — policy entropy (for entropy bonus)
        """
        logits, values = self.forward(x, edge_index, edge_attr)
        dist           = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)       # [N]
        entropy   = dist.entropy()               # [N]
        return log_probs, values, entropy

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        mode = "shared" if self.shared_encoder else "separate"
        return (
            f"GNN_ActorCritic("
            f"encoder={mode}, "
            f"phases={self.num_phases}, "
            f"params={self.count_parameters():,})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.graph_builder import build_grid_edge_index

    print("── GNN_ActorCritic unit test ───────────────────────────")

    graph = build_grid_edge_index(n=16, side=4)
    x     = torch.randn(16, STATE_DIM)
    ei    = graph.edge_index
    ea    = graph.edge_attr

    # ── Shared encoder ────────────────────────────────────────────────────────
    ac = GNN_ActorCritic(shared_encoder=True)
    logits, value = ac(x, ei, ea)
    assert logits.shape == (16, NUM_PHASES), f"logits: {logits.shape}"
    assert value.shape  == (16, 1),          f"value:  {value.shape}"
    print(f"✓ forward() shared encoder:  logits={logits.shape}, value={value.shape}")

    # ── Separate encoders (ablation) ──────────────────────────────────────────
    ac2 = GNN_ActorCritic(shared_encoder=False)
    logits2, value2 = ac2(x, ei, ea)
    assert logits2.shape == (16, NUM_PHASES)
    print(f"✓ forward() separate encoders: logits={logits2.shape}, value={value2.shape}")

    # ── get_action ────────────────────────────────────────────────────────────
    actions, log_probs, entropy = ac.get_action(x, ei, ea, deterministic=False)
    assert actions.shape    == (16,), f"actions:   {actions.shape}"
    assert log_probs.shape  == (16,), f"log_probs: {log_probs.shape}"
    assert entropy.shape    == (16,), f"entropy:   {entropy.shape}"
    print(f"✓ get_action():              actions={actions.shape}, log_probs={log_probs.shape}")

    # ── evaluate_actions ──────────────────────────────────────────────────────
    lp2, v2, ent2 = ac.evaluate_actions(x, ei, ea, actions)
    assert lp2.shape == (16,) and v2.shape == (16, 1)
    print(f"✓ evaluate_actions():        log_probs={lp2.shape}, values={v2.shape}")

    # ── Gradient flow ─────────────────────────────────────────────────────────
    (logits.sum() + value.sum()).backward()
    ok = all(p.grad is not None for p in ac.parameters())
    print(f"✓ Gradient flow:             {ok}")

    print(f"\n{ac}")
    print("── All tests passed ────────────────────────────────────")
