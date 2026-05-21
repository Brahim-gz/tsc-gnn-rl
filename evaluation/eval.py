# evaluation/eval.py
# ============================================================
# Runs trained model + all baselines across N seeds.
# Exports results/metrics.csv and prints a comparison table.
#
# Run:
#     python evaluation/eval.py --checkpoint checkpoints/dqn_multi_final.pt
#     python evaluation/eval.py --baseline-only   # run baselines, no model
# ============================================================

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import DEFAULT_CONFIG, STATE_DIM, NUM_PHASES
from env.cityflow_wrapper import CityFlowEnv, make_cityflow_config
from env.graph_builder import build_graph_from_roadnet, build_grid_edge_index, update_graph_features, obs_matrix_to_dict
from agents.baselines import RandomBaseline, FixedTimeBaseline, MaxPressureBaseline


# ─────────────────────────────────────────────────────────────────────────────
# Single-episode evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, policy_fn, graph=None, tl_ids=None, device=None) -> dict:
    """
    Run one full episode (3600 sim-seconds) and return metrics.

    policy_fn : callable(obs, graph, tl_ids, device) → np.ndarray[N]
    """
    obs, _ = env.reset()
    total_reward = 0.0
    step_queues  = []
    done         = False

    while not done:
        actions = policy_fn(obs, graph, tl_ids, device)
        obs, reward, done, _, info = env.step(actions)
        total_reward += reward
        step_queues.append(info.get("avg_queue", 0.0))

    return {
        "total_reward":   total_reward,
        "avg_travel_time":env.avg_travel_time,       # CityFlow built-in (seconds/vehicle)
        "avg_queue":      float(np.mean(step_queues)),
        "max_queue":      float(np.max(step_queues)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrappers (all share the same interface)
# ─────────────────────────────────────────────────────────────────────────────

def make_gnn_policy(model, greedy=True):
    """Returns a policy_fn that uses the trained GNN model."""
    def policy_fn(obs, graph, tl_ids, device):
        graph = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
        x     = graph.x.to(device)
        with torch.no_grad():
            try:
                # DQN: returns [N, phases]
                q_vals = model(x, graph.edge_index, graph.edge_attr)
                return q_vals.argmax(dim=1).cpu().numpy()
            except (ValueError, TypeError):
                # PPO: returns (logits, value)
                logits, _ = model(x, graph.edge_index, graph.edge_attr)
                return logits.argmax(dim=1).cpu().numpy()
    return policy_fn


def make_baseline_policy(baseline, env):
    """Wraps a Baseline object into a policy_fn."""
    if hasattr(baseline, "reset"):
        baseline.reset()
    def policy_fn(obs, graph=None, tl_ids=None, device=None):
        return baseline.act(obs=obs, info=None, env=env)
    return policy_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N      = args.num_intersections

    os.makedirs("results", exist_ok=True)

    # Build environment
    config_path = make_cityflow_config(
        roadnet_path = args.roadnet,
        flow_path    = args.flow,
        save_dir     = "data/eval",
    )
    env = CityFlowEnv(
        config_path       = config_path,
        num_intersections = N,
        episode_seconds   = cfg["episode_seconds"],
        delta_time        = cfg["delta_time"],
        reward_fn         = "queue",  # always evaluate with queue reward
    )

    # Build graph template
    try:
        graph, tl_ids = build_graph_from_roadnet(args.roadnet)
        graph = graph.to(device)
    except Exception:
        side  = args.side
        graph = build_grid_edge_index(n=side * side, side=side).to(device)
        tl_ids = env.intersection_ids

    # ── Build list of agents to evaluate ─────────────────────────────────────
    agents = {}

    # Baselines (always included)
    agents["Random"]       = make_baseline_policy(RandomBaseline(N),       env)
    agents["Fixed-time"]   = make_baseline_policy(FixedTimeBaseline(N),    env)
    agents["Max-Pressure"] = make_baseline_policy(MaxPressureBaseline(N),  env)

    # Trained model (optional)
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            from models.gnn_qnetwork import GNN_QNetwork
            model = GNN_QNetwork(
                state_dim=STATE_DIM, embed_dim=cfg["embed_dim"],
                num_phases=NUM_PHASES, gat_heads=cfg["gat_heads"],
                hidden_dim=cfg["hidden_dim"],
            ).to(device)
            ckpt  = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt.get("model", ckpt))
            model.eval()
            agents["GNN-DQN"] = make_gnn_policy(model)
            print(f"✓ Loaded GNN-DQN from {args.checkpoint}")
        except Exception as e:
            print(f"⚠ Could not load DQN model: {e}")

        try:
            from models.gnn_actor_critic import GNN_ActorCritic
            model = GNN_ActorCritic(
                state_dim=STATE_DIM, embed_dim=cfg["embed_dim"],
                num_phases=NUM_PHASES, gat_heads=cfg["gat_heads"],
                hidden_dim=cfg["hidden_dim"],
            ).to(device)
            ckpt  = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt.get("model", ckpt))
            model.eval()
            agents["GNN-PPO"] = make_gnn_policy(model)
            print(f"✓ Loaded GNN-PPO from {args.checkpoint}")
        except Exception as e:
            pass  # DQN checkpoint was loaded above — skip silently

    # ── Run all agents across seeds ───────────────────────────────────────────
    all_results = []
    seeds       = list(range(args.num_seeds))

    print(f"\nEvaluating {len(agents)} agents × {args.num_seeds} seeds "
          f"on {N}-intersection grid...")
    print(f"{'Agent':<16}  {'Seed':>4}  {'Travel Time':>12}  "
          f"{'Avg Queue':>10}  {'Reward':>10}")
    print("-" * 60)

    for agent_name, policy_fn in agents.items():
        for seed in seeds:
            env_seed = seed * 100
            config_s = make_cityflow_config(
                roadnet_path = args.roadnet,
                flow_path    = args.flow,
                save_dir     = f"data/eval/seed{seed}",
                seed         = env_seed,
            )
            env_s = CityFlowEnv(
                config_path       = config_s,
                num_intersections = N,
                episode_seconds   = cfg["episode_seconds"],
                delta_time        = cfg["delta_time"],
                reward_fn         = "queue",
            )

            t0     = time.time()
            result = run_episode(env_s, policy_fn, graph, tl_ids, device)
            result.update({
                "agent": agent_name,
                "seed":  seed,
                "wall_time_s": time.time() - t0,
            })
            all_results.append(result)

            print(f"{agent_name:<16}  {seed:>4}  "
                  f"{result['avg_travel_time']:>12.1f}  "
                  f"{result['avg_queue']:>10.3f}  "
                  f"{result['total_reward']:>10.1f}")
            env_s.close()

    env.close()

    # ── Aggregate and save ────────────────────────────────────────────────────
    df     = pd.DataFrame(all_results)
    csv_path = "results/metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Full results saved to {csv_path}")

    # Summary table
    summary = df.groupby("agent").agg(
        avg_travel_time = ("avg_travel_time", ["mean", "std"]),
        avg_queue       = ("avg_queue",       ["mean", "std"]),
        total_reward    = ("total_reward",    ["mean", "std"]),
    ).round(2)

    print("\n── Summary (mean ± std across seeds) ──────────────────────────")
    print(summary.to_string())
    summary.to_csv("results/summary.csv")
    print(f"✓ Summary saved to results/summary.csv")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet",           default="data/synthetic_4x4/roadnet.json")
    parser.add_argument("--flow",              default="data/synthetic_4x4/flow.json")
    parser.add_argument("--checkpoint",        default=None, help="Path to .pt model file")
    parser.add_argument("--num-intersections", type=int, default=16)
    parser.add_argument("--side",              type=int, default=4)
    parser.add_argument("--num-seeds",         type=int, default=5)
    parser.add_argument("--baseline-only",     action="store_true")
    args = parser.parse_args()

    if args.baseline_only:
        args.checkpoint = None

    evaluate(args)