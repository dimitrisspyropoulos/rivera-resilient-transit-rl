# -*- coding: utf-8 -*-
"""Builds the input file for the control experiment.

The control asks: what does the recovery agent achieve on the network the
operator already runs? To answer it, recovery.py is fed a "design" that is R0
itself. This script takes the real design file and replaces every scenario's
solution with R0's own route set and metrics, keeping p, C0 and B unchanged so
that the recovery stage faces exactly the same approved budget.

    python make_R0_control.py best_scenarios_baseline_v7.json best_scenarios_baseline_R0.json

Then point recovery.py at the output:

    BEST_JSON_FILE = "best_scenarios_baseline_R0.json"
    OUT_DIR        = "R0_recovery_results"
"""
import json
import sys

FIELDS = {"best_att": "att", "best_CEF": "CEF", "best_CEF2": "CEF2",
          "best_exposure": "exposure", "best_dun": "dun", "best_cost": "cost",
          "best_energy": "energy", "best_routes": "routes"}


def main(src: str, dst: str) -> None:
    data = json.loads(open(src, encoding="utf-8").read())
    r0 = data["initial_solution"]
    for name, scen in data["best_scenarios"].items():
        for target, source in FIELDS.items():
            scen[target] = r0[source]
        # the energy term exists only in the energy scenario
        if name != "S2_energy":
            scen["best_energy"] = None
    open(dst, "w", encoding="utf-8").write(
        json.dumps(data, ensure_ascii=False, indent=2))
    print(f"{dst}: every scenario now carries R0's {len(r0['routes'])} routes "
          f"at {r0['cost']:,.2f} EUR")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["best_scenarios_baseline_v7.json",
                             "best_scenarios_baseline_R0.json"]))
