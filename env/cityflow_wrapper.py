# env/cityflow_wrapper.py

import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from interfaces import STATE_DIM, NUM_PHASES


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path:       str,
        num_intersections: int  = 16,
        episode_seconds:   int  = 3600,
        delta_time:        int  = 10,
        reward_fn:         str  = "queue",
        thread_num:        int  = 4,
        seed:              int  = 0,
    ):
        super().__init__()

        import cityflow
        self.eng = cityflow.Engine(config_path, thread_num=thread_num)

        self.config_path       = config_path
        self.num_intersections = num_intersections
        self.episode_seconds   = episode_seconds
        self.delta_time        = delta_time
        self.reward_fn_name    = reward_fn
        self._step_count       = 0

        # ── Parse intersection IDs from roadnet.json ──────────────────────────
        # CityFlow Engine has no get_intersection_ids() method.
        # We read the roadnet file that was passed in the config instead.
        roadnet_path = self._get_roadnet_path(config_path)
        self.intersection_ids = self._parse_tl_ids(roadnet_path, num_intersections)
        print(f"  Found {len(self.intersection_ids)} traffic-light intersections")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(num_intersections, STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [NUM_PHASES] * num_intersections
        )

        self._phase_timers = {iid: 0 for iid in self.intersection_ids}
        self._last_phases  = {iid: 0 for iid in self.intersection_ids}
        self._lane_cache   = {}

    # ── Helpers to read roadnet path from config ───────────────────────────────

    def _get_roadnet_path(self, config_path: str) -> str:
        """Read the roadnetFile path out of the CityFlow config JSON."""
        with open(config_path) as f:
            cfg = json.load(f)
        roadnet = cfg.get("roadnetFile", "")
        # Path may be absolute or relative to the config file's directory
        if not os.path.isabs(roadnet):
            roadnet = os.path.join(os.path.dirname(config_path), roadnet)
        return roadnet

    def _parse_tl_ids(self, roadnet_path: str, limit: int) -> list:
        """
        Parse traffic-light intersection IDs from roadnet.json.
        Returns at most `limit` IDs.
        """
        with open(roadnet_path) as f:
            rn = json.load(f)
        ids = [
            inter["id"]
            for inter in rn.get("intersections", [])
            if not inter.get("virtual", False)
        ]
        if len(ids) == 0:
            raise ValueError(f"No non-virtual intersections found in {roadnet_path}")
        return ids[:limit]

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eng.reset()
        self._step_count   = 0
        self._phase_timers = {iid: 0 for iid in self.intersection_ids}
        self._last_phases  = {iid: 0 for iid in self.intersection_ids}
        self._lane_cache   = {}
        return self._get_observation(), {}

    def step(self, actions: np.ndarray):
        assert len(actions) == self.num_intersections

        for i, iid in enumerate(self.intersection_ids):
            phase = int(actions[i]) % NUM_PHASES
            self.eng.set_tl_phase(iid, phase)

            if phase != self._last_phases[iid]:
                self._phase_timers[iid] = 0
                self._last_phases[iid]  = phase
            else:
                self._phase_timers[iid] += self.delta_time

        for _ in range(self.delta_time):
            self.eng.next_step()
        self._step_count += self.delta_time

        obs    = self._get_observation()
        reward = self._compute_reward(actions)
        done   = self._step_count >= self.episode_seconds

        info = {
            "step":           self._step_count,
            "avg_queue":      self._avg_queue(),
            "total_vehicles": self.eng.get_vehicle_count(),
        }
        return obs, reward, done, False, info

    def render(self): pass

    def close(self):
        try:
            self.eng.reset()
        except Exception:
            pass

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Returns np.ndarray[N, STATE_DIM=8]:
            [queue_0..3 (normalized), phase_oh_0..2, elapsed_norm]
        """
        # get_lane_waiting_vehicle_count() returns waiting vehicles per lane
        # (more meaningful than total count for queue estimation)
        try:
            lane_counts = self.eng.get_lane_waiting_vehicle_count()
        except AttributeError:
            lane_counts = self.eng.get_lane_vehicle_count()

        states = []
        for iid in self.intersection_ids:
            lanes    = self._get_approach_lanes(iid, lane_counts)
            capacity = 20.0

            queues   = [lane_counts.get(l, 0) / capacity for l in lanes[:4]]
            queues  += [0.0] * (4 - len(queues))

            phase    = self._last_phases.get(iid, 0) % 3
            phase_oh = [float(phase == k) for k in range(3)]

            elapsed  = min(self._phase_timers.get(iid, 0), 60) / 60.0

            states.append(queues + phase_oh + [elapsed])

        return np.array(states, dtype=np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self, actions) -> float:
        if self.reward_fn_name == "queue":
            return self._reward_queue()
        elif self.reward_fn_name == "pressure":
            return self._reward_pressure()
        elif self.reward_fn_name == "combined":
            return self._reward_combined(actions)
        else:
            raise ValueError(f"Unknown reward: {self.reward_fn_name}")

    def _reward_queue(self) -> float:
        try:
            counts = self.eng.get_lane_waiting_vehicle_count()
        except AttributeError:
            counts = self.eng.get_lane_vehicle_count()
        return -sum(counts.values()) / max(self.num_intersections, 1)

    def _reward_pressure(self) -> float:
        try:
            counts = self.eng.get_lane_waiting_vehicle_count()
        except AttributeError:
            counts = self.eng.get_lane_vehicle_count()
        total = 0.0
        for iid in self.intersection_ids:
            in_lanes  = self._get_approach_lanes(iid, counts)
            out_lanes = self._get_exit_lanes(iid, counts)
            total    += abs(
                sum(counts.get(l, 0) for l in in_lanes) -
                sum(counts.get(l, 0) for l in out_lanes)
            )
        return -total / max(self.num_intersections, 1)

    def _reward_combined(self, actions) -> float:
        q = self._reward_queue()
        w = -self.eng.get_average_travel_time() / 300.0
        s = -sum(
            1.0 for i, iid in enumerate(self.intersection_ids)
            if int(actions[i]) != self._last_phases.get(iid, 0)
        ) / max(self.num_intersections, 1)
        return 0.5 * q + 0.3 * w + 0.2 * s

    # ── Lane helpers ───────────────────────────────────────────────────────────

    def _get_approach_lanes(self, iid: str, lane_counts: dict) -> list:
        if iid not in self._lane_cache:
            self._build_lane_cache(iid, lane_counts)
        return self._lane_cache[iid]["in"]

    def _get_exit_lanes(self, iid: str, lane_counts: dict) -> list:
        if iid not in self._lane_cache:
            self._build_lane_cache(iid, lane_counts)
        return self._lane_cache[iid]["out"]

    def _build_lane_cache(self, iid: str, lane_counts: dict):
        """
        CityFlow lane naming convention from generate_grid_scenario.py:
            road_{startIntersection}_{endIntersection}_{lane_index}
        Incoming lanes end at iid → road_*_{iid}_*
        Outgoing lanes start at iid → road_{iid}_*_*
        """
        all_lanes = list(lane_counts.keys())
        in_lanes  = [l for l in all_lanes if f"_{iid}_" in l]
        out_lanes = [l for l in all_lanes if l.startswith(f"road_{iid}_")]

        # Fallback: if naming convention doesn't match, grab any lanes mentioning iid
        if not in_lanes and not out_lanes:
            in_lanes  = [l for l in all_lanes if iid in l][:8]
            out_lanes = []

        self._lane_cache[iid] = {"in": in_lanes[:8], "out": out_lanes[:8]}

    def _avg_queue(self) -> float:
        try:
            counts = self.eng.get_lane_waiting_vehicle_count()
        except AttributeError:
            counts = self.eng.get_lane_vehicle_count()
        return sum(counts.values()) / max(len(counts), 1)

    # ── Utility ────────────────────────────────────────────────────────────────

    @property
    def avg_travel_time(self) -> float:
        return self.eng.get_average_travel_time()

    def get_obs_dict(self) -> dict:
        obs = self._get_observation()
        return {iid: obs[i] for i, iid in enumerate(self.intersection_ids)}


# ── Config builder ─────────────────────────────────────────────────────────────

def make_cityflow_config(
    roadnet_path: str,
    flow_path:    str,
    save_dir:     str,
    seed:         int  = 0,
    save_replay:  bool = False,
) -> str:
    """
    Build a CityFlow engine config JSON.

    Key fix: CityFlow resolves ALL file paths relative to the `dir` field.
    We set `dir` to save_dir and use only filenames (not absolute paths)
    for roadnetFile, flowFile, and the replay log files.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")

    # CityFlow needs dir to end with a slash
    abs_save_dir = os.path.abspath(save_dir).rstrip("/") + "/"

    # Replay log filenames — simple names, resolved relative to dir
    roadnet_log = "replay_roadnet.json" if save_replay else ""
    replay_log  = "replay.txt"          if save_replay else ""

    config = {
        "interval":       1.0,
        "seed":           seed,
        "dir":            abs_save_dir,          # ← base for ALL paths
        "roadnetFile":    os.path.abspath(roadnet_path),  # absolute → safe
        "flowFile":       os.path.abspath(flow_path),     # absolute → safe
        "rlTrafficLight": True,
        "laneChange":     False,
        "saveReplay":     save_replay,
        "roadnetLogFile": roadnet_log,           # relative to dir
        "replayLogFile":  replay_log,            # relative to dir
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path