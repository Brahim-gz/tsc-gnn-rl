# env/cityflow_wrapper.py
# ============================================================
# CityFlow → Gymnasium wrapper.
# Replaces sumo-rl entirely. No OS-level dependencies.
# pip install: cityflow (needs cmake)
# ============================================================

import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from interfaces import STATE_DIM, NUM_PHASES, DEFAULT_CONFIG


class CityFlowEnv(gym.Env):
    """
    Multi-intersection traffic signal control environment.

    Wraps CityFlow as a single Gymnasium environment where:
        - Each intersection = one RL agent
        - All agents share a single policy (parameter sharing)
        - Observation : Tensor[N, STATE_DIM]  — one row per intersection
        - Action      : np.ndarray[N]          — phase index per intersection

    Usage
    -----
        env = CityFlowEnv(config_path="config.json", num_intersections=16)
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(actions)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path:       str,
        num_intersections: int  = DEFAULT_CONFIG["num_intersections"],
        episode_seconds:   int  = DEFAULT_CONFIG["episode_seconds"],
        delta_time:        int  = DEFAULT_CONFIG["delta_time"],
        reward_fn:         str  = "queue",       # "queue" | "pressure" | "combined"
        thread_num:        int  = DEFAULT_CONFIG["cityflow_threads"],
        seed:              int  = 0,
    ):
        super().__init__()

        import cityflow  # imported here so the file can be read without cityflow installed
        self._cityflow = cityflow

        self.config_path       = config_path
        self.num_intersections = num_intersections
        self.episode_seconds   = episode_seconds
        self.delta_time        = delta_time
        self.reward_fn_name    = reward_fn
        self.thread_num        = thread_num
        self._seed             = seed
        self._step_count       = 0

        # Build the engine
        self.eng = self._cityflow.Engine(config_path, thread_num=thread_num)

        # Cache intersection IDs (only traffic-light controlled ones)
        all_ids = list(self.eng.get_intersection_ids())
        try:
            self.intersection_ids = [i for i in all_ids
                                      if not self.eng.get_intersection_id(i).is_virtual][:num_intersections]
        except AttributeError:
            self.intersection_ids = self._load_non_virtual_intersections_from_roadnet(
                config_path
            )[:num_intersections]

        # If the engine doesn't expose metadata, fall back to all IDs
        # (safe fallback — CityFlow API varies slightly by version)
        if len(self.intersection_ids) == 0:
            self.intersection_ids = all_ids[:num_intersections]

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(num_intersections, STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [NUM_PHASES] * num_intersections
        )

        # Cache lane lists per intersection (set on first reset)
        self._lane_cache: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eng.reset()
        self._step_count   = 0
        self._phase_timers = {iid: 0 for iid in self.intersection_ids}
        self._last_phases  = {iid: 0 for iid in self.intersection_ids}
        obs = self._get_observation()
        return obs, {}

    def step(self, actions: np.ndarray):
        """
        Apply phase actions for all intersections, advance the simulator
        by delta_time seconds, return (obs, reward, done, truncated, info).
        """
        assert len(actions) == self.num_intersections, \
            f"Expected {self.num_intersections} actions, got {len(actions)}"

        # Set phase for each intersection
        for i, iid in enumerate(self.intersection_ids):
            phase = int(actions[i]) % self._get_num_phases(iid)
            self.eng.set_tl_phase(iid, phase)

            # Update phase timer
            if phase != self._last_phases[iid]:
                self._phase_timers[iid] = 0
                self._last_phases[iid]  = phase
            else:
                self._phase_timers[iid] += self.delta_time

        # Advance simulator
        for _ in range(self.delta_time):
            self.eng.next_step()
        self._step_count += self.delta_time

        obs    = self._get_observation()
        reward = self._compute_reward(actions)
        done   = self._step_count >= self.episode_seconds

        info = {
            "step":          self._step_count,
            "avg_queue":     float(np.mean([self._get_queue(iid)
                                            for iid in self.intersection_ids])),
            "total_vehicles": self.eng.get_vehicle_count(),
        }
        return obs, reward, done, False, info

    def render(self):
        pass  # CityFlow has a web-based replay — no inline rendering needed

    def close(self):
        try:
            self.eng.reset()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Returns np.ndarray[N, STATE_DIM] where STATE_DIM = 8:
            [queue_0, queue_1, queue_2, queue_3,   ← 4 normalized queue lengths
             phase_oh_0, phase_oh_1, phase_oh_2,   ← 3-bit one-hot current phase
             elapsed_norm]                          ← time since last switch / 60
        """
        lane_counts = self.eng.get_lane_vehicle_count()
        states = []

        for iid in self.intersection_ids:
            lanes    = self._get_approach_lanes(iid)
            capacity = max(self._get_lane_capacity(iid), 1)

            # Queue per approach lane (normalized, padded/trimmed to 4)
            queues = [lane_counts.get(l, 0) / capacity for l in lanes[:4]]
            queues += [0.0] * (4 - len(queues))

            # Phase one-hot (3 bits — groups 4 phases into 3 for compactness)
            phase    = self._last_phases.get(iid, 0) % 3
            phase_oh = [float(phase == k) for k in range(3)]

            # Elapsed time since last phase change (normalized to 60 s)
            elapsed = min(self._phase_timers.get(iid, 0), 60) / 60.0

            states.append(queues + phase_oh + [elapsed])

        return np.array(states, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward functions
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, actions: np.ndarray) -> float:
        """
        Dispatch to the selected reward function.
        All return a scalar float (negative = penalty).
        """
        if self.reward_fn_name == "queue":
            return self._reward_queue()
        elif self.reward_fn_name == "pressure":
            return self._reward_pressure()
        elif self.reward_fn_name == "combined":
            return self._reward_combined(actions)
        else:
            raise ValueError(f"Unknown reward function: {self.reward_fn_name}")

    def _reward_queue(self) -> float:
        """Negative sum of all waiting vehicles across all intersections."""
        lane_counts = self.eng.get_lane_vehicle_count()
        total = sum(lane_counts.values())
        return -total / max(self.num_intersections, 1)

    def _reward_pressure(self) -> float:
        """
        PressLight-style pressure reward.
        Pressure = |incoming_vehicles - outgoing_vehicles| per intersection.
        Lower pressure = better flow balance.
        """
        lane_counts = self.eng.get_lane_vehicle_count()
        total_pressure = 0.0
        for iid in self.intersection_ids:
            in_lanes  = self._get_approach_lanes(iid)
            out_lanes = self._get_exit_lanes(iid)
            incoming  = sum(lane_counts.get(l, 0) for l in in_lanes)
            outgoing  = sum(lane_counts.get(l, 0) for l in out_lanes)
            total_pressure += abs(incoming - outgoing)
        return -total_pressure / max(self.num_intersections, 1)

    def _reward_combined(self, actions: np.ndarray) -> float:
        """
        Weighted combination:
            0.5 * queue  +  0.3 * waiting_time  +  0.2 * phase_switch_penalty
        """
        queue_r   = self._reward_queue()
        waiting_r = -self.eng.get_average_travel_time() / 300.0  # normalize to ~300s
        switch_r  = -sum(
            1.0 for i, iid in enumerate(self.intersection_ids)
            if int(actions[i]) != self._last_phases.get(iid, 0)
        ) / max(self.num_intersections, 1)

        return 0.5 * queue_r + 0.3 * waiting_r + 0.2 * switch_r

    # ─────────────────────────────────────────────────────────────────────────
    # CityFlow helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_non_virtual_intersections_from_roadnet(self, config_path: str) -> list:
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            roadnet_file = cfg.get("roadnetFile")
            if not roadnet_file:
                return []
            base_dir = cfg.get("dir", "")
            if base_dir == "":
                base_dir = os.path.dirname(os.path.abspath(config_path))
            rn_path = os.path.join(base_dir, roadnet_file)
            with open(rn_path, "r") as f:
                rn_data = json.load(f)
            return [
                i["id"] for i in rn_data.get("intersections", [])
                if not i.get("virtual", False)
            ]
        except Exception:
            return []

    def _get_approach_lanes(self, iid: str) -> list:
        """Return incoming lane IDs for an intersection (cached)."""
        if iid not in self._lane_cache:
            self._lane_cache[iid] = self._build_lane_cache(iid)
        return self._lane_cache[iid]["in"]

    def _get_exit_lanes(self, iid: str) -> list:
        if iid not in self._lane_cache:
            self._lane_cache[iid] = self._build_lane_cache(iid)
        return self._lane_cache[iid]["out"]

    def _build_lane_cache(self, iid: str) -> dict:
        """
        Parse lane IDs from CityFlow's lane naming convention.
        Incoming lanes: road ends at this intersection → contain iid in name.
        This is a heuristic; adjust if your roadnet uses different naming.
        """
        all_lanes = list(self.eng.get_lane_vehicle_count().keys())
        in_lanes  = [l for l in all_lanes if f"_{iid}_" in l or l.endswith(f"_{iid}")]
        out_lanes = [l for l in all_lanes if l.startswith(f"{iid}_")]
        return {"in": in_lanes[:8], "out": out_lanes[:8]}

    def _get_queue(self, iid: str) -> float:
        """Total waiting vehicles at an intersection (for logging)."""
        lane_counts = self.eng.get_lane_vehicle_count()
        lanes = self._get_approach_lanes(iid)
        return sum(lane_counts.get(l, 0) for l in lanes)

    def _get_num_phases(self, iid: str) -> int:
        """Number of valid phases at an intersection (default 4)."""
        return NUM_PHASES

    def _get_lane_capacity(self, iid: str) -> int:
        """Approximate lane capacity (vehicles). Heuristic: 20 vehicles."""
        return 20

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def avg_travel_time(self) -> float:
        """CityFlow's built-in average travel time metric."""
        return self.eng.get_average_travel_time()

    def get_obs_dict(self) -> dict:
        """Return {agent_id: state_vector} dict (used by graph_builder)."""
        obs_matrix = self._get_observation()
        return {iid: obs_matrix[i] for i, iid in enumerate(self.intersection_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# Config file builder
# ─────────────────────────────────────────────────────────────────────────────

def make_cityflow_config(
    roadnet_path: str,
    flow_path:    str,
    save_dir:     str,
    seed:         int  = 0,
    save_replay:  bool = False,
) -> str:
    """
    Build a CityFlow engine config JSON and write it to save_dir/config.json.
    Returns the path to the config file.

    Parameters
    ----------
    roadnet_path : path to roadnet.json (absolute or relative to save_dir)
    flow_path    : path to flow.json
    save_dir     : directory where config.json will be written
    seed         : random seed for reproducibility
    save_replay  : whether to save a replay log (for visualization)
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")

    replay_path = os.path.join(save_dir, "replay.txt") if save_replay else ""

    config = {
        "interval":          1.0,
        "seed":              seed,
        "dir":               "",
        "roadnetFile":       os.path.abspath(roadnet_path),
        "flowFile":          os.path.abspath(flow_path),
        "rlTrafficLight":    True,
        "laneChange":        False,
        "saveReplay":        save_replay,
        "roadnetLogFile":    replay_path,
        "replayLogFile":     replay_path,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path