"""Discrete-event simulation of the MUET Point (bus) system using SimPy.

Entities : passengers (students/staff), buses (Points)
Resources: depot (FIFO store of idle buses), two terminal queues (STC, Main Gate)
Events   : passenger arrival, dispatch, bus arrival at terminal, boarding
           complete, arrival at destination, return to depot
"""
from collections import deque
import numpy as np
import simpy


class TransitSystem:
    def __init__(self, cfg, policy, seed=0):
        self.cfg, self.policy = cfg, policy
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.depot = simpy.Store(self.env)
        for i in range(cfg.n_buses):
            self.depot.put(i)
        self.queues = [deque() for _ in cfg.terminals]   # entries: [arrival_time, bypassed_flag]
        # statistics
        self.waits = []                  # wait of every boarded passenger
        self.waits_by_term = [[] for _ in cfg.terminals]
        self.arrivals = 0
        self.left_behind = 0             # unique passengers who watched a full bus leave
        self.trips = []                  # (t_dispatch, load)
        self.bus_busy_time = 0.0
        self.q_trace = []                # (t, q_STC, q_MG)
        self.pending_seats = 0           # seats of buses dispatched but not yet loaded
        self._ewma = cfg.total_rate(0.0)
        self._arr_in_tick = 0
        self.policy.reset(cfg)

    # ---------------- helper views used by policies --------------------
    def total_queue(self):
        return sum(len(q) for q in self.queues)

    def net_queue(self):
        """Queue minus seats already committed to buses on their way to load."""
        return max(0, self.total_queue() - self.pending_seats)

    def oldest_wait(self, t):
        heads = [q[0][0] for q in self.queues if q]
        return t - min(heads) if heads else 0.0

    def rate_estimate(self):
        return self._ewma                       # pax/min (EWMA of arrivals)

    # ---------------- stochastic building blocks -----------------------
    def travel(self, mean, t):
        c = self.cfg
        mult = self.rng.lognormal(-0.5 * c.traffic_sigma ** 2, c.traffic_sigma)
        if c.is_peak(t):
            mult *= c.peak_congestion
        d = mean * mult
        if self.rng.random() < c.incident_prob:
            d += c.incident_delay
        return d

    # ---------------- processes ----------------------------------------
    def arrivals_proc(self, k):
        """Non-homogeneous Poisson arrivals by Lewis-Shedler thinning."""
        c = self.cfg
        lam_max = c.demand_scale * (c.base_rate + c.peak_rate) * c.share_terminal[k]
        t = 0.0
        while True:
            t += self.rng.exponential(1.0 / lam_max)
            if t > c.window:
                return
            yield self.env.timeout(max(0.0, t - self.env.now))
            if self.rng.random() <= c.rate(t, k) / lam_max:
                self.queues[k].append([self.env.now, False])
                self.arrivals += 1
                self._arr_in_tick += 1

    def monitor_proc(self):
        c, alpha = self.cfg, 0.05
        while True:
            inst = self._arr_in_tick / c.dispatch_tick
            self._ewma = (1 - alpha) * self._ewma + alpha * inst
            self._arr_in_tick = 0
            if int(self.env.now / c.dispatch_tick) % 4 == 0:
                self.q_trace.append((self.env.now, *[len(q) for q in self.queues]))
            yield self.env.timeout(c.dispatch_tick)

    def dispatcher_proc(self):
        c = self.cfg
        while True:
            if self.policy.should_dispatch(self.env.now, self):
                bus = yield self.depot.get()      # waits if no idle bus
                t_disp = self.env.now
                self.policy.on_dispatch(t_disp, self)
                self.pending_seats += c.capacity
                self.env.process(self.bus_proc(bus, t_disp))
            else:
                yield self.env.timeout(c.dispatch_tick)

    def board(self, k, free):
        """Board up to `free` passengers (FIFO). Returns number boarded."""
        q, n = self.queues[k], 0
        while q and n < free:
            a, _ = q.popleft()
            w = self.env.now - a
            self.waits.append(w)
            self.waits_by_term[k].append(w)
            n += 1
        for e in q:                             # remaining passengers saw a bus leave
            if not e[1]:
                e[1] = True
                self.left_behind += 1
        return n

    def bus_proc(self, bus, t_disp):
        c, env = self.cfg, self.env
        load, t_start = 0, env.now
        order = list(range(len(c.terminals)))
        if self.policy.adaptive_order:          # terminal with the oldest waiting student first
            order.sort(key=lambda k: self.queues[k][0][0] if self.queues[k] else 1e9)
        yield env.timeout(self.travel(c.t_depot_to_first, env.now))
        express = False
        if self.policy.express:
            # saturation: one terminal alone can fill the bus -> skip the other (short route)
            k0 = order[0]
            express = len(self.queues[k0]) >= c.capacity
        if express:
            order = order[:1]
        for i, k in enumerate(order):
            if load < c.capacity:
                free = c.capacity - load
                if self.policy.adaptive_order and not express and i < len(order) - 1:
                    # fair-share quota: keep seats for the terminals still to be visited
                    others = sum(len(self.queues[j]) for j in order[i + 1:])
                    mine = len(self.queues[k])
                    prop = int(round(c.capacity * mine / max(mine + others, 1)))
                    free = min(free, max(prop, c.capacity - others))
                n = min(len(self.queues[k]), free)
                yield env.timeout(c.door_time + n * c.board_time)
                load += self.board(k, free)
            if i < len(order) - 1:
                yield env.timeout(self.travel(c.t_between_terminals, env.now))
        if express:
            # the skipped inter-terminal leg is partly avoided
            yield env.timeout(self.travel(c.t_between_terminals - c.express_saving, env.now))
        self.pending_seats -= c.capacity         # bus finished loading
        yield env.timeout(self.travel(c.t_last_to_dest, env.now))
        yield env.timeout(c.unload_time)
        yield env.timeout(self.travel(c.t_dest_to_depot, env.now))
        self.trips.append((t_disp, load))
        self.bus_busy_time += env.now - t_start
        yield self.depot.put(bus)

    # ---------------- run ----------------------------------------------
    def run(self):
        c = self.cfg
        for k in range(len(c.terminals)):
            self.env.process(self.arrivals_proc(k))
        self.env.process(self.monitor_proc())
        self.env.process(self.dispatcher_proc())
        # run until all arrivals served (or max_time)
        while self.env.now < c.max_time:
            self.env.run(until=self.env.now + 5.0)
            if self.env.now > c.window and self.total_queue() == 0:
                break
        return self.summary()

    def summary(self):
        c = self.cfg
        w = np.array(self.waits) if self.waits else np.array([0.0])
        loads = np.array([l for _, l in self.trips]) if self.trips else np.array([0])
        horizon = max(self.env.now, 1.0)
        return {
            "policy": self.policy.name,
            "arrivals": self.arrivals,
            "served": len(self.waits),
            "mean_wait": float(w.mean()),
            "median_wait": float(np.median(w)),
            "p95_wait": float(np.percentile(w, 95)),
            "max_wait": float(w.max()),
            "max_queue": int(max((a + b) for _, a, b in self.q_trace)) if self.q_trace else 0,
            "left_behind_pct": 100.0 * self.left_behind / max(self.arrivals, 1),
            "trips": len(self.trips),
            "avg_load_factor": float(loads.mean() / c.capacity) if len(loads) else 0.0,
            "fleet_util": float(self.bus_busy_time / (c.n_buses * horizon)),
            "mean_wait_STC": float(np.mean(self.waits_by_term[0])) if self.waits_by_term[0] else 0.0,
            "mean_wait_MG": float(np.mean(self.waits_by_term[1])) if self.waits_by_term[1] else 0.0,
        }
