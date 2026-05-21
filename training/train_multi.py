# training/train_multi.py
# ============================================================
# Run:
#     python training/train_multi.py --algo dqn
#     python training/train_multi.py --algo ppo
# ============================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import DEFAULT_CONFIG, STATE_DIM, NUM_PHASES, EMBED_DIM
from env.cityflow_wrapper import CityFlowEnv, make_cityflow_config
from env.graph_builder import build_graph_from_roadnet, build_grid_edge_index, update_graph_features, obs_matrix_to_dict
from training.replay_buffer import GraphReplayBuffer
from agents.baselines import FixedTimeBaseline, MaxPressureBaseline

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (waits for Person B's file, falls back to stub)
# ─────────────────────────────────────────────────────────────────────────────

def load_dqn_model(cfg, device):
    try:
        from models.gnn_qnetwork import GNN_QNetwork
        model = GNN_QNetwork(
            state_dim  = STATE_DIM,
            embed_dim  = cfg["embed_dim"],
            num_phases = NUM_PHASES,
            gat_heads  = cfg["gat_heads"],
            hidden_dim = cfg["hidden_dim"],
        ).to(device)
        print("✓ Loaded GNN_QNetwork")
        return model
    except ImportError:
        print("⚠ GNN_QNetwork not found — using MLP stub")
        import torch.nn as nn
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, NUM_PHASES)
                )
            def forward(self, x, edge_index, edge_attr=None):
                return self.fc(x)
        return _Stub().to(device)


def load_ppo_model(cfg, device):
    try:
        from models.gnn_actor_critic import GNN_ActorCritic
        model = GNN_ActorCritic(
            state_dim  = STATE_DIM,
            embed_dim  = cfg["embed_dim"],
            num_phases = NUM_PHASES,
            gat_heads  = cfg["gat_heads"],
            hidden_dim = cfg["hidden_dim"],
        ).to(device)
        print("✓ Loaded GNN_ActorCritic")
        return model
    except ImportError:
        print("⚠ GNN_ActorCritic not found — using MLP stub")
        import torch.nn as nn
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor  = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, NUM_PHASES))
                self.critic = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, 1))
            def forward(self, x, edge_index, edge_attr=None):
                return self.actor(x), self.critic(x)
        return _Stub().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# DQN training
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(env, graph, tl_ids, cfg, args, device):
    N = env.num_intersections

    q_net   = load_dqn_model(cfg, device)
    target  = load_dqn_model(cfg, device)
    target.load_state_dict(q_net.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])

    buffer = GraphReplayBuffer(
        capacity   = cfg["replay_capacity"],
        edge_index = graph.edge_index,
        edge_attr  = graph.edge_attr,
        num_nodes  = N,
        device     = device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path  = f"checkpoints/dqn_multi_latest.pt"
    start_step = 0

    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt       = torch.load(ckpt_path, map_location=device)
        q_net.load_state_dict(ckpt["model"])
        target.load_state_dict(ckpt["target"])
        start_step = ckpt["step"]
        print(f"✓ Resumed DQN from step {start_step:,}")

    obs, _ = env.reset()
    ep_reward, ep_steps, ep_num = 0.0, 0, 0
    t0 = time.time()

    print(f"\nDQN training — {cfg['total_steps']:,} steps on {N}-intersection grid")

    for step in range(start_step, cfg["total_steps"]):
        epsilon = _get_epsilon(step, cfg)

        # ε-greedy action via GNN
        if np.random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            graph   = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
            x       = graph.x.to(device)
            with torch.no_grad():
                q_vals  = q_net(x, graph.edge_index, graph.edge_attr)  # [N, phases]
                actions = q_vals.argmax(dim=1).cpu().numpy()

        next_obs, reward, done, _, info = env.step(actions)
        buffer.push(obs, actions, reward, next_obs, done)
        obs        = next_obs
        ep_reward += reward
        ep_steps  += 1

        # Learn
        loss_val = 0.0
        if buffer.is_ready(cfg["batch_size"]) and step % cfg["train_freq"] == 0:
            batch    = buffer.sample(cfg["batch_size"])
            loss_val = _dqn_update(q_net, target, optimizer, batch, cfg, N)

        if step % cfg["target_update_freq"] == 0:
            target.load_state_dict(q_net.state_dict())

        if done:
            ep_num += 1
            _log({
                "algo":           "dqn",
                "step":           step,
                "episode":        ep_num,
                "episode_reward": ep_reward,
                "avg_travel_time":env.avg_travel_time,
                "avg_queue":      info.get("avg_queue", 0),
                "epsilon":        epsilon,
                "steps_per_sec":  ep_steps / max(time.time() - t0, 1e-6),
                "loss":           loss_val,
            }, args)

            if ep_num % 10 == 0:
                print(f"  step={step:>8,}  ep={ep_num:>4}  reward={ep_reward:>8.1f}"
                      f"  tt={env.avg_travel_time:>6.1f}s  ε={epsilon:.3f}")

            ep_reward, ep_steps = 0.0, 0
            t0     = time.time()
            obs, _ = env.reset()

        # Checkpoint
        if step % cfg["save_freq"] == 0 and step > start_step:
            _save({"model": q_net.state_dict(), "target": target.state_dict(),
                   "step": step}, ckpt_path, step, "dqn")

    return q_net


def _dqn_update(q_net, target, optimizer, batch, cfg, N) -> float:
    B = batch["B"]

    q_curr = q_net(
        batch["obs"], batch["edge_index"], batch["edge_attr"]
    ).view(B, N, NUM_PHASES)

    with torch.no_grad():
        q_next = target(
            batch["next_obs"], batch["edge_index"], batch["edge_attr"]
        ).view(B, N, NUM_PHASES).max(dim=2).values

    target_q = (
        batch["rewards"].unsqueeze(1) +
        cfg["gamma"] * q_next * (1.0 - batch["dones"].unsqueeze(1))
    )

    q_chosen = q_curr.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)
    loss     = F.smooth_l1_loss(q_chosen, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# PPO training 
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one PPO rollout."""
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def add(self, obs, actions, reward, log_prob, value, done):
        self.obs.append(obs.copy())
        self.actions.append(actions.copy())
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(float(done))

    def compute_returns(self, last_value, gamma, gae_lambda):
        """GAE advantage estimation."""
        T  = len(self.rewards)
        N  = len(self.actions[0])
        adv = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N,      dtype=np.float32)

        values_np = np.array([v.cpu().numpy().flatten() for v in self.values])  # [T, N]
        last_v    = last_value.cpu().numpy().flatten()  # [N]

        for t in reversed(range(T)):
            next_val  = last_v if t == T - 1 else values_np[t + 1]
            delta     = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - values_np[t]
            gae       = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            adv[t]    = gae

        returns = adv + values_np
        return adv, returns

    def clear(self):
        self.__init__()


def train_ppo(env, graph, tl_ids, cfg, args, device):
    N     = env.num_intersections
    model = load_ppo_model(cfg, device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path  = "checkpoints/ppo_multi_latest.pt"
    start_step = 0

    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt       = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        print(f"✓ Resumed PPO from step {start_step:,}")

    rollout  = RolloutBuffer()
    obs, _   = env.reset()
    ep_reward = 0.0
    ep_num    = 0
    global_step = start_step

    print(f"\nPPO training — {cfg['total_steps']:,} steps on {N}-intersection grid")

    while global_step < cfg["total_steps"]:
        # ── Collect rollout ───────────────────────────────────────────────────
        for _ in range(cfg["rollout_steps"]):
            graph = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
            x     = graph.x.to(device)
            ei    = graph.edge_index
            ea    = graph.edge_attr

            with torch.no_grad():
                logits, value = model(x, ei, ea)   # [N, phases], [N, 1]
                dist          = Categorical(logits=logits)
                actions_t     = dist.sample()       # [N]
                log_prob      = dist.log_prob(actions_t).sum()  # scalar

            actions = actions_t.cpu().numpy()
            next_obs, reward, done, _, info = env.step(actions)
            rollout.add(obs, actions, reward, log_prob.item(), value, done)

            obs        = next_obs
            ep_reward += reward
            global_step += 1

            if done:
                ep_num += 1
                _log({
                    "algo":           "ppo",
                    "step":           global_step,
                    "episode":        ep_num,
                    "episode_reward": ep_reward,
                    "avg_travel_time":env.avg_travel_time,
                    "avg_queue":      info.get("avg_queue", 0),
                }, args)
                if ep_num % 5 == 0:
                    print(f"  step={global_step:>8,}  ep={ep_num:>4}"
                          f"  reward={ep_reward:>8.1f}  tt={env.avg_travel_time:>6.1f}s")
                ep_reward = 0.0
                obs, _    = env.reset()

        # ── Compute returns ───────────────────────────────────────────────────
        with torch.no_grad():
            graph   = update_graph_features(graph, obs_matrix_to_dict(obs, tl_ids), tl_ids)
            _, last_v = model(graph.x.to(device), graph.edge_index, graph.edge_attr)
        adv, returns = rollout.compute_returns(last_v, cfg["gamma"], cfg["gae_lambda"])

        # ── PPO update (multiple epochs) ──────────────────────────────────────
        T         = len(rollout.obs)
        obs_arr   = np.array(rollout.obs, dtype=np.float32)     # [T, N, D]
        act_arr   = np.array(rollout.actions, dtype=np.int64)   # [T, N]
        adv_t     = torch.tensor(adv,     dtype=torch.float32).to(device)   # [T, N]
        ret_t     = torch.tensor(returns, dtype=torch.float32).to(device)   # [T, N]
        old_lp    = torch.tensor(rollout.log_probs, dtype=torch.float32).to(device)  # [T]

        # Normalize advantages
        adv_flat  = adv_t.view(-1)
        adv_t     = ((adv_t - adv_flat.mean()) / (adv_flat.std() + 1e-8))

        total_loss = 0.0
        for _ in range(cfg["ppo_epochs"]):
            # Mini-batch over timesteps
            idxs = np.random.permutation(T)
            for start in range(0, T, 32):
                mb_idx = idxs[start:start + 32]
                mb_obs = torch.tensor(obs_arr[mb_idx]).view(-1, STATE_DIM).to(device)
                mb_act = torch.tensor(act_arr[mb_idx]).to(device)   # [mb*N]
                mb_adv = adv_t[mb_idx].view(-1)                     # [mb*N]
                mb_ret = ret_t[mb_idx].view(-1, 1)
                mb_olp = old_lp[mb_idx]                              # [mb]

                # Repeat edges for mini-batch
                mb      = len(mb_idx)
                mb_ei   = torch.cat([graph.edge_index + i * N for i in range(mb)], dim=1)
                mb_ea   = graph.edge_attr.repeat(mb, 1)

                logits, value = model(mb_obs, mb_ei, mb_ea)
                dist          = Categorical(logits=logits.view(mb, N, NUM_PHASES)
                                            .view(mb * N, NUM_PHASES))
                new_lp        = dist.log_prob(mb_act.view(-1)).view(mb, N).sum(dim=1)

                ratio         = torch.exp(new_lp - mb_olp)
                pg_loss1      = -mb_adv.view(mb, N).mean(1) * ratio
                pg_loss2      = -mb_adv.view(mb, N).mean(1) * torch.clamp(
                    ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]
                )
                pg_loss       = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss       = F.mse_loss(value.view(-1, 1), mb_ret)
                entropy       = dist.entropy().mean()

                loss  = pg_loss + cfg["vf_coef"] * vf_loss - cfg["ent_coef"] * entropy
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                opt.step()
                total_loss += loss.item()

        rollout.clear()

        if global_step % cfg["save_freq"] < cfg["rollout_steps"]:
            _save({"model": model.state_dict(), "step": global_step},
                  ckpt_path, global_step, "ppo")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_epsilon(step, cfg):
    progress = min(step / cfg["epsilon_decay"], 1.0)
    return cfg["epsilon_start"] + progress * (cfg["epsilon_end"] - cfg["epsilon_start"])

def _log(data, args):
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.log(data)

def _save(state, path, step, tag):
    torch.save(state, path)
    versioned = path.replace("latest", f"step_{step:07d}")
    torch.save(state, versioned)
    print(f"  → [{tag}] Checkpoint saved at step {step:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Algo: {args.algo}  |  Reward: {args.reward}")

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project = "project04-traffic-gnn",
            name    = f"{args.algo}-multi-{time.strftime('%m%d-%H%M')}",
            config  = {**cfg, "algo": args.algo, "reward": args.reward},
            resume  = "allow",
        )

    # Build environment
    config_path = make_cityflow_config(
        roadnet_path = args.roadnet,
        flow_path    = args.flow,
        save_dir     = "data/multi",
    )
    env = CityFlowEnv(
        config_path       = config_path,
        num_intersections = args.num_intersections,
        episode_seconds   = cfg["episode_seconds"],
        delta_time        = cfg["delta_time"],
        reward_fn         = args.reward,
    )

    # Build graph (try roadnet.json first, fall back to synthetic grid)
    try:
        graph, tl_ids = build_graph_from_roadnet(args.roadnet)
        graph = graph.to(device)
        print(f"✓ Graph from roadnet: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"⚠ Could not parse roadnet ({e}). Using synthetic {args.side}×{args.side} grid.")
        side  = args.side
        graph = build_grid_edge_index(n=side * side, side=side).to(device)
        tl_ids = env.intersection_ids

    # Train
    if args.algo == "dqn":
        model = train_dqn(env, graph, tl_ids, cfg, args, device)
    else:
        model = train_ppo(env, graph, tl_ids, cfg, args, device)

    # Save final model
    final_path = f"checkpoints/{args.algo}_multi_final.pt"
    torch.save({"model": model.state_dict(), "step": cfg["total_steps"]}, final_path)
    print(f"\n✓ Final model saved to {final_path}")

    env.close()
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet",           default="data/synthetic_4x4/roadnet.json")
    parser.add_argument("--flow",              default="data/synthetic_4x4/flow.json")
    parser.add_argument("--algo",              default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--reward",            default="queue", choices=["queue","pressure","combined"])
    parser.add_argument("--num-intersections", type=int, default=16)
    parser.add_argument("--side",              type=int, default=4, help="Grid side length")
    parser.add_argument("--no-wandb",          action="store_true")
    parser.add_argument("--fresh",             action="store_true")
    args = parser.parse_args()
    main(args)