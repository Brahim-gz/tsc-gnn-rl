# agents/baselines.py
# ============================================================
# Non-learning baselines for comparison.
#
# All baselines implement the same interface:
#     actions = baseline.act(obs, info, env)  → np.ndarray[N]
# ============================================================

import numpy as np
from interfaces import NUM_PHASES


class RandomBaseline:
    """
    Selects a random phase at every step.
    Absolute floor — every real method should beat this.
    """
    def __init__(self, num_intersections: int, num_phases: int = NUM_PHASES, seed: int = 0):
        self.num_intersections = num_intersections
        self.num_phases        = num_phases
        self.rng               = np.random.default_rng(seed)

    def act(self, obs=None, info=None, env=None) -> np.ndarray:
        return self.rng.integers(0, self.num_phases, size=self.num_intersections)


class FixedTimeBaseline:
    """
    Cycles through phases with a fixed green duration.
    Classic engineering baseline — no data required.

    Parameters
    ----------
    phase_duration : seconds each phase stays green (default 30s)
    num_phases     : number of phases per intersection
    delta_time     : seconds per action step (must match env)
    """
    def __init__(
        self,
        num_intersections: int,
        phase_duration:    int = 30,
        num_phases:        int = NUM_PHASES,
        delta_time:        int = 10,
    ):
        self.num_intersections = num_intersections
        self.phase_duration    = phase_duration
        self.num_phases        = num_phases
        self.delta_time        = delta_time
        self._elapsed          = 0

    def act(self, obs=None, info=None, env=None) -> np.ndarray:
        # All intersections follow the same fixed cycle
        phase = int(self._elapsed // self.phase_duration) % self.num_phases
        self._elapsed += self.delta_time
        return np.full(self.num_intersections, phase, dtype=np.int64)

    def reset(self):
        self._elapsed = 0


class MaxPressureBaseline:
    """
    Greedy heuristic: at each intersection, select the phase that
    maximizes |incoming - outgoing| vehicle counts.

    This is a strong non-learning baseline — it can be difficult
    to beat with RL on small grids.

    Requires access to the CityFlow engine through the env object.
    """
    def __init__(self, num_intersections: int, num_phases: int = NUM_PHASES):
        self.num_intersections = num_intersections
        self.num_phases        = num_phases

    def act(self, obs=None, info=None, env=None) -> np.ndarray:
        if env is None:
            # Fallback: if no env provided, use phase=0
            return np.zeros(self.num_intersections, dtype=np.int64)

        lane_counts = env.eng.get_lane_vehicle_count()
        actions     = np.zeros(self.num_intersections, dtype=np.int64)

        for i, iid in enumerate(env.intersection_ids):
            best_phase    = 0
            best_pressure = -1.0

            for phase_idx in range(self.num_phases):
                # Approximate: count vehicles in lanes associated with this phase
                # CityFlow lane naming: incoming lanes named *_iid_*
                in_count  = sum(
                    v for k, v in lane_counts.items()
                    if f"_{iid}_" in k or k.endswith(f"_{iid}")
                )
                out_count = sum(
                    v for k, v in lane_counts.items()
                    if k.startswith(f"{iid}_")
                )
                pressure = abs(in_count - out_count)

                if pressure > best_pressure:
                    best_pressure = pressure
                    best_phase    = phase_idx

            actions[i] = best_phase

        return actions


class CyclicBaseline:
    """
    Simple round-robin: each intersection advances to the next phase
    every N steps. Similar to fixed-time but per-intersection offset
    to reduce simultaneous red phases across the grid.
    """
    def __init__(
        self,
        num_intersections: int,
        steps_per_phase:   int = 3,   # action steps (= steps_per_phase * delta_time seconds)
        num_phases:        int = NUM_PHASES,
    ):
        self.num_intersections = num_intersections
        self.steps_per_phase   = steps_per_phase
        self.num_phases        = num_phases
        self._step             = 0

    def act(self, obs=None, info=None, env=None) -> np.ndarray:
        # Offset each intersection by its index to stagger the green waves
        phases = np.array([
            int((self._step + i) // self.steps_per_phase) % self.num_phases
            for i in range(self.num_intersections)
        ], dtype=np.int64)
        self._step += 1
        return phases

    def reset(self):
        self._step = 0