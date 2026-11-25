"""Verification checks for the simulator (run: python tests/verify.py)."""
import os, sys, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import Config
from policies import StaticFixed, StaticPeakAware, DynamicQueue, HybridAdaptive, ALL_POLICIES
from model import TransitSystem

cfg = Config()

# 1. Conservation: every arrival is eventually served, no one is lost
for P in ALL_POLICIES:
    r = TransitSystem(cfg, P(), seed=7).run()
    assert r["arrivals"] == r["served"], P.name
print("[ok] 1. conservation of passengers (arrivals == served) for all policies")

# 2. Arrival process: simulated mean count vs integral of lambda(t)
expected = sum(cfg.total_rate(t + 0.5) for t in range(int(cfg.window)))
sim = np.mean([TransitSystem(cfg, StaticFixed(), seed=s).run()["arrivals"] for s in range(60)])
print(f"[ok] 2. arrivals: analytic E[N]={expected:.1f}, simulated mean={sim:.1f}  (err {100*abs(sim-expected)/expected:.1f}%)")
assert abs(sim - expected) / expected < 0.03

# 3. Deterministic limit: no randomness in travel, one passenger -> wait = headway slot
c = copy.deepcopy(cfg); c.traffic_sigma = 0; c.incident_prob = 0; c.base_rate = 0; c.peak_rate = 0
s = TransitSystem(c, StaticFixed(5.0), seed=0)
s.queues[0].append([0.0, False]); s.arrivals = 1
s.env.process(s.dispatcher_proc()); s.env.process(s.monitor_proc()); s.env.run(until=30)
assert len(s.waits) == 1 and 4.9 < s.waits[0] < 20, s.waits
print(f"[ok] 3. deterministic single-passenger run: wait={s.waits[0]:.2f} min (bus reaches STC after depot leg)")

# 4. Capacity is never exceeded
for P in ALL_POLICIES:
    s = TransitSystem(cfg, P(), seed=3); s.run()
    assert max(l for _, l in s.trips) <= cfg.capacity
print("[ok] 4. bus load never exceeds capacity")

# 5. Little's law: time-average queue L == lambda_eff * W (total wait area equals integral of queue)
s = TransitSystem(cfg, StaticFixed(), seed=11); s.run()
q = np.array(s.q_trace); area = np.trapezoid(q[:, 1] + q[:, 2], q[:, 0])
total_wait = sum(s.waits)
print(f"[ok] 5. Little's law: integral of queue = {area:.0f} pax-min, sum of waits = {total_wait:.0f} pax-min "
      f"(diff {100*abs(area-total_wait)/total_wait:.1f}%, trace is sampled once/min)")
assert abs(area - total_wait) / total_wait < 0.05

# 6. Monotonicity: more demand never lowers the baseline wait
w = []
for sc in (0.8, 1.0, 1.2):
    c2 = copy.deepcopy(cfg); c2.demand_scale = sc
    w.append(np.mean([TransitSystem(c2, StaticFixed(), seed=s).run()["mean_wait"] for s in range(15)]))
assert w[0] < w[1] < w[2]
print(f"[ok] 6. monotonic response to demand: {w[0]:.1f} < {w[1]:.1f} < {w[2]:.1f} min")
print("ALL CHECKS PASSED")
