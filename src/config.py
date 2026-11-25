"""Central configuration for the MUET Point-bus DES model.

All numbers are ASSUMED/CALIBRATED parameters (no official MUET data was
provided with the assignment). Replace them with measured values (GPS logs,
smart-card taps, manual counts) to calibrate the model to reality.
Time is measured in MINUTES from the start of a session window.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    # ---- Fleet ----------------------------------------------------------
    n_buses: int = 12            # existing fleet; the study never adds buses
    capacity: int = 50           # passengers per bus (seated + standing)

    # ---- Route (mean travel times, minutes, off-peak) ------------------
    t_depot_to_first: float = 5.0
    t_between_terminals: float = 6.0
    t_last_to_dest: float = 14.0
    t_dest_to_depot: float = 15.0
    express_saving: float = 4.0  # minutes saved when a bus serves ONE terminal only (assumed)
    unload_time: float = 2.0     # unloading at destination
    door_time: float = 0.5       # fixed dwell per terminal stop
    board_time: float = 0.035    # per passenger boarding time (~2 s)

    # ---- Traffic uncertainty -------------------------------------------
    traffic_sigma: float = 0.18      # log-normal sigma of travel multiplier
    peak_congestion: float = 1.25    # extra road congestion in peak hour
    incident_prob: float = 0.03      # prob. of a blockage on a trip leg
    incident_delay: float = 10.0     # minutes added by an incident

    # ---- Demand (non-homogeneous Poisson) -------------------------------
    session: str = "morning"         # label only: morning / afternoon
    window: float = 180.0            # arrivals generated for 180 minutes
    base_rate: float = 3.0           # pax/min off-peak (both terminals)
    peak_rate: float = 11.0          # additional pax/min at the peak
    peak_time: float = 90.0          # minute of the peak inside the window
    peak_sigma: float = 22.0         # width of the peak (minutes)
    share_terminal: tuple = (0.55, 0.45)   # STC, Main Gate
    demand_scale: float = 1.0        # for what-if surge/drop scenarios

    terminals: tuple = ("STC", "Main Gate")

    # ---- Simulation -----------------------------------------------------
    max_time: float = 420.0          # hard stop
    dispatch_tick: float = 0.25      # dispatcher polling interval (min)

    def rate(self, t: float, terminal: int) -> float:
        """Arrival intensity lambda_k(t) in pax/min for one terminal."""
        g = np.exp(-0.5 * ((t - self.peak_time) / self.peak_sigma) ** 2)
        total = self.base_rate + self.peak_rate * g
        return self.demand_scale * total * self.share_terminal[terminal]

    def total_rate(self, t: float) -> float:
        return sum(self.rate(t, k) for k in range(len(self.terminals)))

    def is_peak(self, t: float) -> bool:
        return abs(t - self.peak_time) <= 1.5 * self.peak_sigma

    def cycle_time(self, peak: bool = False) -> float:
        """Mean bus round-trip time (no queueing), minutes."""
        base = (self.t_depot_to_first + self.t_between_terminals +
                self.t_last_to_dest + self.t_dest_to_depot)
        base *= self.peak_congestion if peak else 1.0
        return base + 2 * self.door_time + self.unload_time + \
            self.capacity * self.board_time + 4 * self.incident_prob * self.incident_delay
