# Results

Output of the pipeline. Every figure quoted in the main [README](../README.md) — and every
number in the ITS 2026 paper — comes from these files.

The paper writes the budget reserve as **λ**; in the code it is `BUDGET_RESERVE`. Greek
*rho* was dropped because in Times New Roman it is hard to tell apart from `p`, the budget
level, and from `r`, the reward.

```
results/
  design/                  best_scenarios_baseline_v7.json   structural, λ = 0.05
                           best_scenarios_baseline_v6.json   local,      λ = 0.05
  disruption/              150 events applied to R₀, S1 and S2
  recovery/                the agent on the two designed networks
  control_R0/              the same agent on R₀ itself — the control
  sensitivity_lambda10/    the whole pipeline again at λ = 0.10
```

**One folder per experiment, deliberately.** The three runs produce files with identical
names — `v7_recovery_agent_…_mc_results.json` is written by the main run, by the control and
by the λ = 0.10 run alike. Flattened into one folder they overwrite each other silently, and
the analysis scripts then read the wrong experiment.

## Stage 1 — Design

`design/best_scenarios_baseline_v7.json` (structural action set) and
`best_scenarios_baseline_v6.json` (local action set).

Each holds `initial_solution` — the existing network `R₀` — and, per scenario, the budget
level `p`, the design budget `B` after the reserve, the episode the best solution was found
at, and `best_att`, `best_CEF`, `best_CEF2`, `best_exposure`, `best_dun`, `best_cost`,
`best_energy` and `best_routes`, the route sets themselves.

**Use them to skip the design stage.** Point `BEST_JSON_FILE` in `disruption.py` at one of
these and go straight to the resilience analysis, saving the three-hour design run.

## Stage 2 — Disruption

`disruption/v7_monte_carlo_disruption_results.json` — all 150 events, applied identically to
`R₀` and both designed networks.

Each run records `metrics` at *t₀* (the moment of failure), `metrics_after_detour` at *t₁*,
and `coverage_floor_percent`, the demand no route set can serve after the damage. Because
the events are drawn with common random numbers, run *k* is the same event for every network
and the comparison is paired.

## Stage 3 — Recovery

`recovery/v7_recovery_agent_…_mc_results.json` — all 300 episodes, including every candidate
plan the agent evaluated and why each was rejected.

The `candidate_evaluations` field is where the hard constraints are visible: each rejected
plan carries an `infeasible_reason` — `over_budget`, `coverage_loss` or
`above_coverage_floor`. This is how the claim of no budget overrun and no coverage loss can
be checked episode by episode, not just in aggregate.

## The control — the same agent on the existing network

`control_R0/` holds the recovery agent run on a "design" that is `R₀` itself, under **the
same 150 events** from `disruption/`, the same approved budget and the same reward. Only the
starting route set differs, so the comparison separates what the redesign contributes from
what the agent does. It has no disruption run of its own — reusing the main one is the point.

```bash
python tools/make_R0_control.py \
       results/design/best_scenarios_baseline_v7.json \
       best_scenarios_baseline_R0.json

# in recovery.py
BEST_JSON_FILE = "best_scenarios_baseline_R0.json"
OUT_DIR        = "R0_recovery_results"

python recovery.py
python tools/analyse_control.py
```

| | R₀ + agent | service design + agent | energy design + agent |
|---|---|---|---|
| undisrupted cost | 32,102 € | 36,201 € | 35,245 € |
| headroom to `B_total` | 6,420 € | 2,321 € | 3,277 € |
| unserved demand at failure | 12.666 % | 7.797 % | 9.318 % |
| recovered / recoverable | 97.1 % | 67.6 % | 73.0 % |
| reached the coverage floor | 139 / 150 | 119 / 150 | 128 / 150 |
| unserved demand after recovery | 2.470 % | 3.991 % | 4.099 % |
| exposure after recovery | 55.87 % | 29.09 % | 21.37 % |
| budget overruns | 0 | 0 | 0 |

Repaired, the network the operator already runs ends an event with less unserved demand than
the redesigned one, because it kept 2.8 times more budget to spend on the day. It also ends
every stage of every event with more than half its demand on a single acceptable path.
Unserved demand converges once the vehicles move again; exposure never does.

## Reserve sensitivity — λ = 0.10

`sensitivity_lambda10/` holds all three stages run again with one line changed:

```python
# v7.py, line 103
BUDGET_RESERVE = 0.10        # main run: 0.05
```

It keeps its own `disruption/` because the designed networks are different and take the
damage themselves. **The two runs must see identical damage**, and `R₀` is unchanged, so it
is the check — it reproduces exactly: 12.666 % unserved demand at failure, a mean coverage
floor of 2.165 %, and the same 58 / 32 / 25 / 18 / 17 split of event types.
`tools/analyse_sensitivity.py` prints that check first.

`B_total = C₀(1 + p) = 38,522 €` in both runs. Since `C₀ = 32,102 €` is already committed,
the approval adds 6,420 € of room, and the reserve takes `λ(1 + p)/p` of it — at `p = 0.20`,
six times λ.

| | λ = 0.05 | λ = 0.10 |
|---|---|---|
| design budget `B` | 36,596 € | 34,670 € |
| withheld | 1,926 € (30 % of the room) | 3,852 € (60 %) |
| CEF₂, service / energy | 1.5981 / 1.6674 | 1.4890 / 1.5990 |
| exposure, energy design | 20.76 % | 31.43 % |
| headroom to `B_total` | 2,321 € / 3,277 € | 3,996 € / 4,179 € |
| recovered / recoverable | 67.6 % / 73.0 % | 91.1 % / 91.5 % |
| reached the coverage floor | 119 / 128 | 135 / 135 |
| unserved demand after recovery | 3.991 % / 4.099 % | 2.967 % / 2.687 % |
| exposure after recovery | 29.09 % / 21.37 % | 31.36 % / 29.14 % |
| budget overruns | 0 | 0 |

```bash
python tools/analyse_sensitivity.py
```

Money moved out of design buys a better end state on the bad day and a more exposed network
on every other one. The final unserved demand at λ = 0.10 lands between λ = 0.05 and the
control, which withheld everything — so the design–recovery trade-off is not a threshold but
a slope. The reserve is a dial, not a parameter.

Both runs use a single seed, so the recovery side — driven by headroom, which is arithmetic
— is robust, while the design-side gap between the two CEF₂ values is one draw of a
stochastic search.

## CSVs and plots

The per-event CSVs and the metric-vs-`p` plots of the **main** run are not committed: they
are regenerated by running the scripts, and `.gitignore` excludes the output folders. The
control and the λ = 0.10 folder keep their CSVs, because neither run is reproducible from the
committed configuration without editing it first.
