# -*- coding: utf-8 -*-
"""Reserve sensitivity: lambda = 0.10 against the main lambda = 0.05 run.

    python tools/analyse_sensitivity.py
"""
import json
from pathlib import Path

C0 = 32101.844943896358
B_TOT = C0 * 1.20
RES = Path(__file__).resolve().parent.parent / "results"
NAME = "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_results.json"

d05 = json.loads((RES / "design/best_scenarios_baseline_v7.json").read_text(encoding="utf-8"))
d10 = json.loads((RES / "sensitivity_lambda10/best_scenarios_baseline_v7_lambda10.json").read_text(encoding="utf-8"))
r05 = json.loads((RES / "recovery" / NAME).read_text(encoding="utf-8"))
r10 = json.loads((RES / "sensitivity_lambda10/recovery" / NAME).read_text(encoding="utf-8"))
mc10 = json.loads((RES / "sensitivity_lambda10/disruption/v7_monte_carlo_disruption_results.json").read_text(encoding="utf-8"))


def stats(data, scen):
    eps = data["scenarios"][scen]["episodes"]
    d0 = [e["metrics_after_disruption"]["dun"] for e in eps]
    d1 = [e["metrics_after_intervention"]["dun"] for e in eps]
    fl = [e["intervention_info"]["coverage_floor_percent"] for e in eps]
    c1 = [e["metrics_after_intervention"]["cost"] for e in eps]
    work = [i for i in range(len(eps)) if d0[i] > fl[i] + 1e-6]
    return dict(
        base=data["scenarios"][scen]["baseline"]["cost"],
        headroom=B_TOT - data["scenarios"][scen]["baseline"]["cost"],
        dmg=sum(d0) / len(eps), fin=sum(d1) / len(eps),
        rec=100 * sum(d0[i] - d1[i] for i in work) / sum(d0[i] - fl[i] for i in work),
        floor=sum(1 for i in range(len(eps)) if d1[i] <= fl[i] + 1e-6),
        over=sum(1 for c in c1 if c > B_TOT + 1e-6),
        x1=sum(e["metrics_after_intervention"]["exposure"] for e in eps) / len(eps))


print("The two runs must see identical damage. R0 is unchanged, so it is the check:")
print(f"  R0 unserved demand at failure, lambda = 0.10 run : "
      f"{sum(r['metrics']['dun'] for r in mc10['scenarios']['R0_reference']['runs']) / 150:.3f} %  "
      f"(paper, lambda = 0.05 run: 12.666 %)")
print(f"  mean coverage floor                              : "
      f"{sum(r['disruption']['coverage_floor_percent'] for r in mc10['scenarios']['S1_service']['runs']) / 150:.3f} %  "
      f"(paper: 2.165 %)\n")

extra = B_TOT - C0
for scen in ("S1_service", "S2_energy"):
    a, b = d05["best_scenarios"][scen], d10["best_scenarios"][scen]
    A, B = stats(r05, scen), stats(r10, scen)
    print(f"=== {scen} ==={'lambda=0.05':>22s}{'lambda=0.10':>14s}")
    rows = [("design budget B (EUR)", f"{a['B']:,.0f}", f"{b['B']:,.0f}"),
            ("withheld (EUR)", f"{B_TOT - a['B']:,.0f}", f"{B_TOT - b['B']:,.0f}"),
            ("withheld, share of the room", f"{100 * (B_TOT - a['B']) / extra:.0f} %", f"{100 * (B_TOT - b['B']) / extra:.0f} %"),
            ("CEF2", f"{a['best_CEF2']:.4f}", f"{b['best_CEF2']:.4f}"),
            ("exposure (%)", f"{a['best_exposure']:.2f}", f"{b['best_exposure']:.2f}"),
            ("design cost (EUR)", f"{A['base']:,.0f}", f"{B['base']:,.0f}"),
            ("headroom to B_total (EUR)", f"{A['headroom']:,.0f}", f"{B['headroom']:,.0f}"),
            ("DUN at failure (%)", f"{A['dmg']:.3f}", f"{B['dmg']:.3f}"),
            ("recovered / recoverable (%)", f"{A['rec']:.1f}", f"{B['rec']:.1f}"),
            ("reached the floor (of 150)", f"{A['floor']}", f"{B['floor']}"),
            ("DUN after recovery (%)", f"{A['fin']:.3f}", f"{B['fin']:.3f}"),
            ("exposure after recovery (%)", f"{A['x1']:.2f}", f"{B['x1']:.2f}"),
            ("budget overruns", f"{A['over']}", f"{B['over']}")]
    for lab, x, y in rows:
        print(f"  {lab:30s}{x:>18s}{y:>14s}")
    print()
