# training/replay_buffer.py
# ============================================================
# Replay buffer that stores graph-structured transitions.
# Compatible with PyTorch Geometric's Batch API.
# ============================================================

import random
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import Optional

from torch_geometric.data import Data, Batch


@dataclass
class Transition:
    """One transition: (s, a, r, s', done)."""
    obs:      np.ndarray   # [N, STATE_DIM]  — raw numpy for memory efficiency
    actions:  np.ndarray   # [N]
    reward:   float
    next_obs: np.ndarray   # [N, STATE_DIM]
    done:     bool


class GraphReplayBuffer:
    """
    Experience replay buffer that stores (obs, actions, reward, next_obs, done)
    tuples as raw numpy arrays and converts them to PyG Batch objects on demand.

    The graph topology (edge_index, edge_attr) is shared across all transitions
    and stored once as a template to save memory.

    Parameters
    ----------
    capacity       : maximum number of transitions to store
    edge_index     : Tensor[2, E] — fixed graph topology (from graph_builder)
    edge_attr      : Tensor[E, EDGE_DIM] — fixed edge features
    num_nodes      : N (number of intersections)
    device         : torch device for model tensors
    """

    def __init__(
        self,
        capacity:   int,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        num_nodes:  int,
        device:     torch.device,
    ):
        self.capacity   = capacity
        self.edge_index = edge_index.to(device)
        self.edge_attr  = edge_attr.to(device)
        self.num_nodes  = num_nodes
        self.device     = device
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        actions:  np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add one transition to the buffer."""
        self._buffer.append(Transition(
            obs      = obs.copy().astype(np.float32),
            actions  = actions.copy().astype(np.int64),
            reward   = float(reward),
            next_obs = next_obs.copy().astype(np.float32),
            done     = bool(done),
        ))

    def sample(self, batch_size: int) -> dict:
        """
        Sample batch_size transitions and return them as batched tensors.

        Returns a dict with keys:
            obs_batch      Tensor[B*N, STATE_DIM]
            actions_batch  Tensor[B, N]
            rewards_batch  Tensor[B]
            next_obs_batch Tensor[B*N, STATE_DIM]
            dones_batch    Tensor[B]
            edge_index     Tensor[2, B*E]   — repeated for each item in batch
            edge_attr      Tensor[B*E, EDGE_DIM]
            batch_mask     Tensor[B*N]      — maps node → batch index
        """
        assert len(self) >= batch_size, \
            f"Not enough transitions: have {len(self)}, need {batch_size}"

        transitions = random.sample(self._buffer, batch_size)
        B = batch_size
        N = self.num_nodes
        E = self.edge_index.size(1)

        obs_batch      = torch.tensor(
            np.stack([t.obs      for t in transitions]), dtype=torch.float32
        ).to(self.device)                              # [B, N, STATE_DIM]
        next_obs_batch = torch.tensor(
            np.stack([t.next_obs for t in transitions]), dtype=torch.float32
        ).to(self.device)                              # [B, N, STATE_DIM]
        actions_batch  = torch.tensor(
            np.stack([t.actions  for t in transitions]), dtype=torch.long
        ).to(self.device)                              # [B, N]
        rewards_batch  = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32
        ).to(self.device)                              # [B]
        dones_batch    = torch.tensor(
            [t.done   for t in transitions], dtype=torch.float32
        ).to(self.device)                              # [B]

        # Flatten obs for GNN: [B, N, D] → [B*N, D]
        obs_flat      = obs_batch.view(B * N, -1)
        next_obs_flat = next_obs_batch.view(B * N, -1)

        # Repeat edge_index for batch: shift node indices by N*i for each item i
        batch_ei   = torch.cat([
            self.edge_index + i * N for i in range(B)
        ], dim=1)                                      # [2, B*E]
        batch_ea   = self.edge_attr.repeat(B, 1)       # [B*E, EDGE_DIM]

        # Batch mask: which batch item does each node belong to?
        batch_mask = torch.arange(B, device=self.device).repeat_interleave(N)

        return {
            "obs":        obs_flat,
            "actions":    actions_batch,
            "rewards":    rewards_batch,
            "next_obs":   next_obs_flat,
            "dones":      dones_batch,
            "edge_index": batch_ei,
            "edge_attr":  batch_ea,
            "batch_mask": batch_mask,
            "B": B, "N": N,
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size