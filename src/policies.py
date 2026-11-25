"""Dispatch policies. A policy decides WHEN a bus leaves the depot and, for
the hybrid policy, in WHICH ORDER it visits the terminals."""
import numpy as np


class Policy:
    name = "base"
    adaptive_order = False
    express = False

    def reset(self, cfg):
        self.cfg = cfg

    def should_dispatch(self, t, sys) -> bool:
        raise NotImplementedError

    def on_dispatch(self, t, sys):
        pass


# P0 Baseline: static timetable, fixed headway all day --------------------
class StaticFixed(Policy):
    name = "P0 Static (baseline)"

    def __init__(self, headway=5.0):
        self.headway = headway

    def reset(self, cfg):
        super().reset(cfg)
        self.next_slot = 0.0

    def should_dispatch(self, t, sys):
        return t >= self.next_slot

    def on_dispatch(self, t, sys):
        self.next_slot += self.headway      # catches up if a bus was late


# P1 Peak-aware static timetable (historical mean demand only) -------------
class StaticPeakAware(Policy):
    name = "P1 Peak-aware timetable"

    def __init__(self, target_load=0.9, h_min=4.2, h_max=5.0):
        self.target_load, self.h_min, self.h_max = target_load, h_min, h_max

    def reset(self, cfg):
        super().reset(cfg)
        self.next_slot = 0.0

    def headway_at(self, t):
        lam = self.cfg.total_rate(min(t, self.cfg.window))
        h = self.target_load * self.cfg.capacity / max(lam, 1e-6)
        return float(np.clip(h, self.h_min, self.h_max))

    def should_dispatch(self, t, sys):
        return t >= self.next_slot

    def on_dispatch(self, t, sys):
        self.next_slot += self.headway_at(self.next_slot)


# P2 Dynamic (queue-triggered) dispatch using live, in-transit-aware queues -
class DynamicQueue(Policy):
    name = "P2 Dynamic dispatch"

    def __init__(self, q_frac=0.8, w_max=5.0, h_min=1.0, min_load=0.4, w_hard=12.0):
        self.q_frac, self.w_max, self.h_min = q_frac, w_max, h_min
        self.min_load, self.w_hard = min_load, w_hard

    def reset(self, cfg):
        super().reset(cfg)
        self.last = -1e9

    def expected_queue(self, t, sys):
        return sys.net_queue()

    def trigger(self, t, sys):
        c = self.cfg
        q = self.expected_queue(t, sys)
        if q <= 0:
            return False
        w = sys.oldest_wait(t)
        return (q >= self.q_frac * c.capacity) or \
               (q >= self.min_load * c.capacity and w >= self.w_max) or \
               (w >= self.w_hard)               # flush rule: nobody waits forever

    def should_dispatch(self, t, sys):
        return (t - self.last >= self.h_min) and self.trigger(t, sys)

    def on_dispatch(self, t, sys):
        self.last = t


# P3 Hybrid: demand-regulated headway (blend of live EWMA and historical rate)
#            + oldest-first visiting order + fair-share seat quota (see model.py)
class HybridAdaptive(Policy):
    name = "P3 Hybrid adaptive"
    adaptive_order = True       # visit the terminal with the oldest waiting student first
    express = False             # single-stop "express" trips: implemented, but did not help (ablation)

    def __init__(self, target_load=0.9, h_min=4.2, h_max=5.0, blend=0.5):
        self.target_load, self.h_min, self.h_max, self.blend = target_load, h_min, h_max, blend

    def reset(self, cfg):
        super().reset(cfg)
        self.next_slot = 0.0

    def target_headway(self, t, sys):
        c = self.cfg
        lam = self.blend * sys.rate_estimate() + (1 - self.blend) * c.total_rate(min(t, c.window))
        return float(np.clip(self.target_load * c.capacity / max(lam, 1e-6), self.h_min, self.h_max))

    def should_dispatch(self, t, sys):
        # never send a bus when nobody is waiting; otherwise follow the regulated headway
        return t >= self.next_slot and sys.total_queue() > 0

    def on_dispatch(self, t, sys):
        self.next_slot = t + self.target_headway(t, sys)


ALL_POLICIES = [StaticFixed, StaticPeakAware, DynamicQueue, HybridAdaptive]
