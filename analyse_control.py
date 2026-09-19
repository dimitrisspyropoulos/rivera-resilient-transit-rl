# -*- coding: utf-8 -*-
"""Control vs designed networks: the decomposition reported in the paper.

    python tools/analyse_control.py
"""
import json
from pathlib import Path

TOT, C0 = 10036.3608, 32101.844943896358
B_TOT = C0 * 1.20
RES = Path(__file__).resolve().parent.parent / "results"
NAME = "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_results.json"

mc = json.loads((RES / "v7_monte_carlo_disruption_results.json").read_text(encoding="utf-8"))
des = json.loads((RES / NAME).read_text(encoding="utf-8"))
r0 = json.loads((RES / "control_R0" / NAME).read_text(encoding="utf-8"))


def stats(data, scen):
    eps = data["scenarios"][scen]["episodes"]
    d0 = [e["metrics_after_disruption"]["dun"] for e in eps]
    d1 = [e["metrics_after_intervention"]["dun"] for e in eps]
    fl = [e["intervention_info"]["coverage_floor_percent"] for e in eps]
    c0 = [e["metrics_after_disruption"]["cost"] for e in eps]
    c1 = [e["metrics_after_intervention"]["cost"] for e in eps]
    work = [i for i in range(len(eps)) if d0[i] > fl[i] + 1e-6]
    return dict(
        base=data["scenarios"][scen]["baseline"]["cost"],
        headroom=B_TOT - data["scenarios"][scen]["baseline"]["cost"],
        dmg=sum(d0) / len(eps), fin=sum(d1) / len(eps),
        work=len(work), acted=sum(1 for i in work if eps[i]["action"] != "do_nothing"),
        rec=100 * sum(d0[i] - d1[i] for i in work) / sum(d0[i] - fl[i] for i in work),
        floor=sum(1 for i in range(len(eps)) if d1[i] <= fl[i] + 1e-6),
        spend=sum(max(0.0, c1[i] - c0[i]) for i in range(len(eps))) / len(eps),
        mx=max(c1), over=sum(1 for c in c1 if c > B_TOT + 1e-6),
        x1=sum(e["metrics_after_intervention"]["exposure"] for e in eps) / len(eps))


r0_t0 = sum(r["metrics"]["dun"] for r in mc["scenarios"]["R0_reference"]["runs"]) / 150
print(f"R0 unserved demand at failure (disruption stage): {r0_t0:.3f} %\n")
print(f"{'':32s}{'R0 + agent':>14s}{'design + agent':>16s}")
print("-" * 62)
for scen in ("S1_service", "S2_energy"):
    A, B = stats(r0, scen), stats(des, scen)
    print(f"\n=== {scen} ===")
    for lab, k, f in [("undisrupted cost (EUR)", "base", "{:,.0f}"),
                      ("headroom to B_total (EUR)", "headroom", "{:,.0f}"),
                      ("DUN at failure (%)", "dmg", "{:.3f}"),
                      ("DUN after recovery (%)", "fin", "{:.3f}"),
                      ("recovered / recoverable (%)", "rec", "{:.1f}"),
                      ("episodes with recoverable demand", "work", "{:.0f}"),
                      ("acted", "acted", "{:.0f}"),
                      ("reached the coverage floor", "floor", "{:.0f}"),
                      ("spend per event (EUR)", "spend", "{:,.0f}"),
                      ("max final cost (EUR)", "mx", "{:,.0f}"),
                      ("budget overruns", "over", "{:.0f}"),
                      ("exposure after recovery (%)", "x1", "{:.2f}")]:
        print(f"  {lab:32s}{f.format(A[k]):>14s}{f.format(B[k]):>16s}")
    d_des = r0_t0 - B["dmg"]
    print(f"\n  design absorbs before anyone acts: {d_des:.3f} points "
          f"({d_des / 100 * TOT:.0f} passengers per event)")
