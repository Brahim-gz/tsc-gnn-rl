# env/graph_builder.py
# ============================================================
# Builds PyTorch Geometric Data objects from:
#   (a) CityFlow roadnet.json files  ← main source
#   (b) Synthetic grid topology      ← fallback / testing
#
# ============================================================

import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from interfaces import STATE_DIM, EDGE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Primary: parse CityFlow roadnet.json
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_roadnet(roadnet_path: str) -> tuple[Data, list]:
    """
    Parse a CityFlow roadnet.json and return a static graph template.

    The graph topology is fixed for the entire training run.
    Only graph.x (node features) is updated at each timestep via
    update_graph_features().

    Returns
    -------
    graph : torch_geometric.data.Data
        x          → zeros placeholder [N, STATE_DIM]
        edge_index → COO edge list     [2, E]
        edge_attr  → edge features     [E, EDGE_DIM]  (length_norm, lanes_norm)
    tl_ids : list[str]
        Intersection IDs in the same order as graph node rows.
        graph.x[i] corresponds to tl_ids[i].
    """
    with open(roadnet_path, "r") as f:
        roadnet = json.load(f)

    # 1. Collect traffic-light intersections
    tl_ids = [
        inter["id"] for inter in roadnet.get("intersections", [])
        if not inter.get("virtual", False)
    ]
    idx_map = {iid: i for i, iid in enumerate(tl_ids)}
    n_nodes = len(tl_ids)

    if n_nodes == 0:
        raise ValueError(f"No traffic-light intersections found in {roadnet_path}")

    # 2. Build edges from road segments
    edge_src, edge_dst, edge_feats = [], [], []

    for road in roadnet.get("roads", []):
        start_id = road.get("startIntersection", "")
        end_id   = road.get("endIntersection",   "")

        if start_id not in idx_map or end_id not in idx_map:
            continue  # skip roads that connect to virtual/boundary intersections

        src = idx_map[start_id]
        dst = idx_map[end_id]

        # Road features
        lanes    = road.get("lanes", [{}])
        n_lanes  = len(lanes)
        # Estimate road length from points if available
        points   = road.get("points", [])
        length   = _polyline_length(points) if len(points) >= 2 else 200.0

        # Normalize: length by 500m, lanes by 4
        feat = [min(length / 500.0, 2.0), min(n_lanes / 4.0, 1.0)]

        # Directed graph: add both directions
        for s, d in [(src, dst), (dst, src)]:
            edge_src.append(s)
            edge_dst.append(d)
            edge_feats.append(feat)

    if len(edge_src) == 0:
        print(f"Warning: no edges found in {roadnet_path}. Using fully-connected fallback.")
        return build_grid_edge_index(n=n_nodes, side=int(n_nodes ** 0.5)), tl_ids

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float)

    # 3. Add self-loops so isolated nodes still get an update
    edge_index, edge_attr = add_self_loops(
        edge_index,
        edge_attr=edge_attr,
        fill_value=torch.tensor([1.0, 1.0]),
        num_nodes=n_nodes,
    )

    # 4. Placeholder node features (filled at runtime)
    x = torch.zeros(n_nodes, STATE_DIM)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph, tl_ids


def _polyline_length(points: list) -> float:
    """Euclidean length of a polyline defined as [{'x':..,'y':..}, ...]."""
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1]["x"] - points[i]["x"]
        dy = points[i + 1]["y"] - points[i]["y"]
        total += (dx ** 2 + dy ** 2) ** 0.5
    return max(total, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: synthetic grid topology
# ─────────────────────────────────────────────────────────────────────────────

def build_grid_edge_index(n: int = 16, side: int = 4) -> Data:
    """
    Build a regular 2D grid graph.
    Used for:
        - Unit testing without a roadnet file
        - Kaggle sanity checks before real data is available
        - The 4x4 synthetic benchmark

    Parameters
    ----------
    n    : total number of nodes (should equal side * side)
    side : grid dimension (4 for 4x4, 6 for 6x6)
    """
    edges, feats = [], []

    for i in range(n):
        r, c = divmod(i, side)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side:
                j = nr * side + nc
                edges.append([i, j])
                feats.append([0.4, 0.75])  # ~200m road, 3 lanes (normalized)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(feats, dtype=torch.float)

    # Add self-loops
    edge_index, edge_attr = add_self_loops(
        edge_index,
        edge_attr=edge_attr,
        fill_value=torch.tensor([1.0, 1.0]),
        num_nodes=n,
    )

    x = torch.zeros(n, STATE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime: update node features every timestep
# ─────────────────────────────────────────────────────────────────────────────

def update_graph_features(
    graph:   Data,
    obs_dict: dict,   # {agent_id: np.ndarray[STATE_DIM]}
    tl_ids:  list,
) -> Data:
    """
    Fill graph.x with live observations from the environment.
    Called at every environment step — graph topology stays fixed.

    Parameters
    ----------
    graph    : the static graph template (from build_graph_from_roadnet)
    obs_dict : mapping from agent/intersection ID → state vector
    tl_ids   : list of intersection IDs in graph node order

    Returns
    -------
    The same graph object with graph.x updated in-place.
    """
    for i, agent_id in enumerate(tl_ids):
        if agent_id in obs_dict:
            vec = obs_dict[agent_id]
            if isinstance(vec, np.ndarray):
                graph.x[i] = torch.from_numpy(vec.astype(np.float32))
            else:
                graph.x[i] = vec.float()
    return graph


def obs_matrix_to_dict(obs_matrix: np.ndarray, tl_ids: list) -> dict:
    """
    Convert a [N, STATE_DIM] observation matrix to the dict format
    expected by update_graph_features().

    obs_matrix : output of CityFlowEnv._get_observation()
    tl_ids     : list of intersection IDs (same order as matrix rows)
    """
    return {tl_ids[i]: obs_matrix[i] for i in range(len(tl_ids))}