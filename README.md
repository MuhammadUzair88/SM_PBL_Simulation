# MUET Point (Bus) Scheduling – Queuing Model & Discrete Event Simulation

PBL assignment: *model, simulate and analyse the MUET Point bus scheduling and queue-management
system, identify bottlenecks, and evaluate algorithmic scheduling optimisations that minimise
student wait times and maximise fleet utilisation — without buying new buses.*

The final IEEE-style report is in [`report/MUET_Bus_Simulation_Final_Report.pdf`](report/MUET_Bus_Simulation_Final_Report.pdf)
(LaTeX source: `report/main.tex`).

> **Important:** no measured MUET data was supplied, so all demand/travel-time numbers in
> `src/config.py` are **assumptions**. The *relative* comparison of policies and the methodology are the
> result; replace the parameters with real counts/GPS data to calibrate.

## What is modelled
* **Entities:** students (non-homogeneous Poisson arrivals, 08:00–09:00 peak), 12 buses of capacity 50.
* **Resources/queues:** FCFS queues at the **STC** and **Main Gate** terminals, a depot (idle buses).
* **Route:** depot → terminal 1 → terminal 2 → campus → depot, with log-normal travel noise,
  peak-hour congestion and random incidents.
* **Policies (`src/policies.py`):**
  | ID | Policy |
  |----|--------|
  | P0 | Static timetable, 5-min headway (baseline) |
  | P1 | Peak-aware timetable (headway 4.2–5 min from historical demand) |
  | P2 | Dynamic, queue-triggered dispatch (reactive) |
  | P3 | Hybrid: demand-regulated headway + oldest-first stop order + fair-share seat quota |

## Project layout
```
src/config.py        all model parameters (edit to calibrate)
src/model.py         SimPy discrete-event model (arrivals, dispatcher, buses, boarding, KPIs)
src/policies.py      dispatch policies P0–P3
src/experiments.py   all experiments E1–E7: tables (CSV) + figures (PNG)
tests/verify.py      verification checks (conservation, Little's law, arrivals, capacity, ...)
results/             generated CSV tables and figures
report/main.tex      IEEE-format report (pdflatex)   report/*.pdf = compiled report
```

## Run instructions
Requires Python 3.9+.
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/verify.py                 # 6 verification checks, should print ALL CHECKS PASSED
python src/experiments.py --quick      # smoke test, ~1 min (8 replications)
python src/experiments.py              # full run used in the report (50 replications, a few minutes)
```
Outputs are written to `results/` (`E1_…csv` … `E7_…csv`, `results/figures/*.png`).

Run a single scenario from Python:
```python
import sys; sys.path.insert(0, "src")
from config import Config
from policies import StaticPeakAware
from model import TransitSystem
print(TransitSystem(Config(), StaticPeakAware(), seed=1).run())
```

Rebuild the report (needs `pdflatex` with `IEEEtran`):
```bash
cd report && pdflatex main.tex && pdflatex main.tex
```

## Experiments
| ID | Content |
|----|---------|
| E1 | Morning session, 4 policies, 50 replications, 95% CIs, Welch tests vs baseline |
| E2 | Analytic traffic intensity ρ(t)=λ(t)/μ(t) and bottleneck window |
| E3 | What-if: fleet size 8–16 |
| E4 | What-if: demand 80–130% |
| E5 | Afternoon session (14:00–15:30 peak) |
| E6 | Traffic-uncertainty sensitivity |
| E7 | Ablation of P3 components (adaptive order, express trips) |

## Headline results (morning, 12 buses, 50 replications)
| Policy | Mean wait (min) | 95th pct (min) | Peak queue | STC / Main Gate wait |
|--------|----------------|----------------|-----------|----------------------|
| P0 Static baseline | 7.75 | 27.7 | 180 | 3.2 / 13.4 |
| P1 Peak-aware timetable | **6.09** | 19.0 | **148** | 3.2 / 9.7 |
| P2 Dynamic dispatch | 12.60 | 32.2 | 231 | 5.4 / 21.6 |
| P3 Hybrid adaptive | 8.63 | **18.0** | 177 | 8.6 / 8.7 |

## Publishing on GitHub
```bash
git init && git add . && git commit -m "MUET bus scheduling DES"
git branch -M main
git remote add origin https://github.com/<your-username>/muet-bus-simulation.git
git push -u origin main
```
