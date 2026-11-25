"""Run every experiment of the report and write CSV tables + PNG figures.

    python src/experiments.py            # full run (~ a few minutes)
    python src/experiments.py --quick    # 8 replications, for a smoke test
"""
import argparse, copy, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from policies import StaticFixed, StaticPeakAware, DynamicQueue, HybridAdaptive, ALL_POLICIES
from model import TransitSystem

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
COL = {"P0": "#d62728", "P1": "#2ca02c", "P2": "#ff7f0e", "P3": "#1f77b4"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})


def key(name):
    return name.split()[0]


def replicate(cfg, policy_cls, n, seed0=0, **kw):
    runs, traces = [], []
    for s in range(n):
        sysm = TransitSystem(cfg, policy_cls(**kw), seed=seed0 + s)
        runs.append(sysm.run())
        traces.append(np.array(sysm.q_trace))
    return pd.DataFrame(runs), traces


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


METRICS = ["mean_wait", "median_wait", "p95_wait", "max_wait", "max_queue",
           "left_behind_pct", "avg_load_factor", "fleet_util", "mean_wait_STC", "mean_wait_MG"]


def table(results):
    rows = []
    for name, df in results.items():
        r = {"policy": name}
        for m in METRICS:
            r[m] = df[m].mean()
            r[m + "_ci"] = ci95(df[m])
        rows.append(r)
    return pd.DataFrame(rows)


def mean_trace(traces, horizon=240):
    grid = np.arange(0, horizon, 1.0)
    acc = []
    for tr in traces:
        tt, q = tr[:, 0], tr[:, 1] + tr[:, 2]
        acc.append(np.interp(grid, tt, q, right=0.0))
    return grid, np.mean(acc, axis=0)


def clock(m, start_h=7.0):
    h = start_h + m / 60.0
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


def main(n):
    cfg = Config()
    # ------------------------------------------------------------------ E1
    results, traces = {}, {}
    for P in ALL_POLICIES:
        df, tr = replicate(cfg, P, n)
        results[P.name], traces[P.name] = df, tr
    t1 = table(results)
    t1.to_csv(f"{OUT}/E1_morning_policy_comparison.csv", index=False)
    # paired-style significance vs baseline (Welch t-test)
    base = results[StaticFixed.name]
    sig = []
    for name, df in results.items():
        if name == StaticFixed.name:
            continue
        for m in ["mean_wait", "p95_wait"]:
            t, p = stats.ttest_ind(df[m], base[m], equal_var=False)
            sig.append({"policy": name, "metric": m, "improvement_pct": 100 * (1 - df[m].mean() / base[m].mean()),
                        "p_value": p})
    pd.DataFrame(sig).to_csv(f"{OUT}/E1_significance_vs_baseline.csv", index=False)

    # Fig: queue length over time
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for name, tr in traces.items():
        g, q = mean_trace(tr)
        ax.plot(g, q, label=name, color=COL[key(name)], lw=1.6)
    ax.axvspan(60, 120, color="grey", alpha=.12, label="08:00-09:00 peak")
    ax.set_xlabel("Time since 07:00 (min)"); ax.set_ylabel("Students waiting (both terminals)")
    ax.set_title("Mean total queue length (50 replications)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/queue_over_time.png"); plt.close(fig)

    # Fig: bar chart of main metrics
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    names = list(results)
    for ax, (m, lab) in zip(axs, [("mean_wait", "Mean wait (min)"), ("p95_wait", "95th pct wait (min)"),
                                  ("max_queue", "Peak queue (students)")]):
        vals = [results[n_][m].mean() for n_ in names]
        errs = [ci95(results[n_][m]) for n_ in names]
        ax.bar([key(n_) for n_ in names], vals, yerr=errs, color=[COL[key(n_)] for n_ in names], capsize=3)
        ax.set_ylabel(lab)
    fig.tight_layout(); fig.savefig(f"{FIG}/metric_bars.png"); plt.close(fig)

    # Fig: terminal inequity
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [results[n_].mean_wait_STC.mean() for n_ in names], w, label="STC", color="#4c78a8")
    ax.bar(x + w / 2, [results[n_].mean_wait_MG.mean() for n_ in names], w, label="Main Gate", color="#f58518")
    ax.set_xticks(x); ax.set_xticklabels([key(n_) for n_ in names]); ax.set_ylabel("Mean wait (min)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/terminal_fairness.png"); plt.close(fig)

    # Fig: wait-time CDF from a pooled set of single runs
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for P in ALL_POLICIES:
        w_all = []
        for s in range(min(n, 20)):
            sm = TransitSystem(cfg, P(), seed=s); sm.run(); w_all += sm.waits
        w_all = np.sort(w_all)
        ax.plot(w_all, np.linspace(0, 1, len(w_all)), color=COL[key(P.name)], label=key(P.name))
    ax.set_xlabel("Wait (min)"); ax.set_ylabel("CDF"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/wait_cdf.png"); plt.close(fig)

    # ------------------------------------------------------------------ E2 analytic utilisation
    tt = np.arange(0, 180, 1.0)
    lam = np.array([cfg.total_rate(t) for t in tt])
    Tc = np.array([cfg.cycle_time(cfg.is_peak(t)) for t in tt])
    mu = cfg.n_buses * cfg.capacity / Tc            # max fleet throughput (pax/min)
    rho = lam / mu
    pd.DataFrame({"t": tt, "lambda": lam, "mu_fleet": mu, "rho": rho}).to_csv(f"{OUT}/E2_utilisation_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(tt, lam, label=r"Arrival rate $\lambda(t)$", color="#1f77b4")
    ax.plot(tt, mu, label=r"Fleet capacity $\mu(t)=NC/T_c(t)$", color="#d62728")
    ax.fill_between(tt, lam, mu, where=lam > mu, color="red", alpha=.15, label=r"$\rho>1$ (bottleneck)")
    ax.set_xlabel("Time since 07:00 (min)"); ax.set_ylabel("Students / min"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/rho_curve.png"); plt.close(fig)
    over = tt[rho > 1]
    with open(f"{OUT}/E2_bottleneck_window.txt", "w") as f:
        f.write(f"rho>1 from minute {over.min():.0f} ({clock(over.min())}) to {over.max():.0f} ({clock(over.max())}); peak rho={rho.max():.2f}\n")

    # ------------------------------------------------------------------ E3 fleet size sweep
    rows = []
    for N in range(8, 17):
        c2 = copy.deepcopy(cfg); c2.n_buses = N
        for P in [StaticFixed, StaticPeakAware, HybridAdaptive]:
            df, _ = replicate(c2, P, max(n // 2, 8), seed0=500)
            rows.append({"fleet": N, "policy": P.name, "mean_wait": df.mean_wait.mean(), "p95_wait": df.p95_wait.mean(),
                         "mean_wait_ci": ci95(df.mean_wait)})
    t3 = pd.DataFrame(rows); t3.to_csv(f"{OUT}/E3_fleet_sweep.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.7))
    for P in [StaticFixed, StaticPeakAware, HybridAdaptive]:
        d = t3[t3.policy == P.name]
        axs[0].plot(d.fleet, d.mean_wait, "o-", ms=3, color=COL[key(P.name)], label=key(P.name))
        axs[1].plot(d.fleet, d.p95_wait, "o-", ms=3, color=COL[key(P.name)], label=key(P.name))
    axs[0].set_ylabel("Mean wait (min)"); axs[1].set_ylabel("95th pct wait (min)")
    for a in axs: a.set_xlabel("Fleet size (buses)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/fleet_sweep.png"); plt.close(fig)

    # ------------------------------------------------------------------ E4 demand scenarios
    rows = []
    for sc in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        c2 = copy.deepcopy(cfg); c2.demand_scale = sc
        for P in [StaticFixed, StaticPeakAware, DynamicQueue, HybridAdaptive]:
            df, _ = replicate(c2, P, max(n // 2, 8), seed0=700)
            rows.append({"demand_scale": sc, "policy": P.name, "mean_wait": df.mean_wait.mean(),
                         "p95_wait": df.p95_wait.mean(), "max_queue": df.max_queue.mean()})
    t4 = pd.DataFrame(rows); t4.to_csv(f"{OUT}/E4_demand_scenarios.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for P in [StaticFixed, StaticPeakAware, DynamicQueue, HybridAdaptive]:
        d = t4[t4.policy == P.name]
        ax.plot(d.demand_scale * 100, d.mean_wait, "o-", ms=3, color=COL[key(P.name)], label=key(P.name))
    ax.set_xlabel("Demand (% of baseline)"); ax.set_ylabel("Mean wait (min)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/demand_scenarios.png"); plt.close(fig)

    # ------------------------------------------------------------------ E5 afternoon session
    ca = copy.deepcopy(cfg); ca.session = "afternoon"; ca.peak_time = 75.0; ca.peak_sigma = 20.0
    ca.share_terminal = (0.45, 0.55)
    res_a = {P.name: replicate(ca, P, max(n // 2, 8), seed0=900)[0] for P in ALL_POLICIES}
    table(res_a).to_csv(f"{OUT}/E5_afternoon_policy_comparison.csv", index=False)

    # ------------------------------------------------------------------ E6 traffic sensitivity
    rows = []
    for sig_, inc in [(0.10, 0.01), (0.18, 0.03), (0.30, 0.06), (0.40, 0.10)]:
        c2 = copy.deepcopy(cfg); c2.traffic_sigma, c2.incident_prob = sig_, inc
        for P in [StaticFixed, StaticPeakAware, HybridAdaptive]:
            df, _ = replicate(c2, P, max(n // 2, 8), seed0=1100)
            rows.append({"traffic_sigma": sig_, "incident_prob": inc, "policy": P.name,
                         "mean_wait": df.mean_wait.mean(), "p95_wait": df.p95_wait.mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/E6_traffic_sensitivity.csv", index=False)

    # ------------------------------------------------------------------ E7 express ablation / quota
    class P3_noquota(HybridAdaptive):
        name = "P3 no adaptive order"; adaptive_order = False
    class P3_express(HybridAdaptive):
        name = "P3 + express"; express = True
    rows = []
    for P in [StaticPeakAware, P3_noquota, HybridAdaptive, P3_express]:
        df, _ = replicate(cfg, P, max(n // 2, 8), seed0=1300)
        rows.append({"variant": P.name, "mean_wait": df.mean_wait.mean(), "p95_wait": df.p95_wait.mean(),
                     "wait_STC": df.mean_wait_STC.mean(), "wait_MG": df.mean_wait_MG.mean(),
                     "max_queue": df.max_queue.mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/E7_ablation.csv", index=False)
    print(t1[["policy", "mean_wait", "p95_wait", "max_queue", "fleet_util", "avg_load_factor", "left_behind_pct",
              "mean_wait_STC", "mean_wait_MG"]].round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(8 if a.quick else 50)
