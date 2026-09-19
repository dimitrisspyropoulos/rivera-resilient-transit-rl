# One Budget, Two Stages: Resilient Urban Network Design and Disruption Recovery.

**Rivera, Uruguay — a reinforcement-learning framework**

Code, data and results for:

- *Multi-Objective Resilient Urban Network Design & Disruption Management under Budget
  Constraints: A Reinforcement Learning Framework for the Case of Rivera, Uruguay* —
  diploma thesis, University of Patras, 2026
- *One Budget, Two Stages: Resilient Transit Network Design and Disruption Recovery* —
  ITS 2026, Patras

Reinforcement-learning design of an urban public-transport network that withstands
extreme disruptions **without increasing the operating budget**.

Case study: **Rivera, Uruguay** — 84 nodes, 286 directed road links, 378 directed OD
pairs, 10 036.36 passengers over a 12-hour horizon.

---

## Overview

Transit networks are normally optimised for average-day performance. This project asks a
different question: *can the same budget buy a network that keeps running when part of the
infrastructure fails?*

The answer is built in three stages, each measured against the same objective:

| Stage | Question | Script |
|---|---|---|
| **Design** | What network does the budget buy? | `v7.py` (`v6.py` for comparison) |
| **Disruption** | What happens the moment infrastructure fails? | `disruption.py` |
| **Recovery** | How much can be restored, and at what cost? | `recovery.py` |

The design deliberately **leaves part of the budget unspent**. That reserve is the capacity
the recovery agent spends when damage occurs — so resilience is paid for from within the
approved budget, not on top of it.

---

## Method

### Stage 1 — Design

A tabular Q-learning agent restructures the existing network `R₀` under two hard constraints:

- **coverage** — all demand is served (`DUN = 0`)
- **budget** — operating cost `C(R) = 0.8 · VKM + 18 · VH` stays within `B`

It runs across four budget levels `p ∈ {0.05, 0.10, 0.15, 0.20}` with warm start: each level
begins from the previous level's route set *and* its learned Q-table and exploration rate.

Two scenarios, differing only in the sustainability term:

```
S1_service   r = ΔCEF + ΔATT + ΔCEF₂
S2_energy    r = ΔCEF + ΔATT + ΔCEF₂ + ΔE
```

All terms are normalised to `R₀`, so no manual weights are involved. The contrast between
S1 and S2 is exactly the functionality ↔ sustainability trade-off.

**Budget reserve.** The design may not commit the full budget:

```
B_design = max(C₀, C₀ · (1 + p) · (1 − λ))        λ = 0.05
```

The `max(C₀, …)` floor keeps the status-quo network feasible at low `p` — without it,
`1.05 × 0.95 = 0.9975 < 1` would make `p = 0.05` infeasible by construction.

In the code the reserve is `BUDGET_RESERVE`. The paper writes it as **λ**; Greek *rho* was
dropped because in Times New Roman it is hard to tell apart from `p`, the budget level, and
from `r`, the reward. A second run at `λ = 0.10` measures how much the split itself matters
— see [`results/sensitivity_lambda10/`](results/).

**Acceptance rule.** Lexicographic — capped redundancy, then travel time, then raw
redundancy. A clear `CEF₂` gain is always accepted; inside a 0.5 % tolerance band the two
solutions count as equivalent in redundancy and travel time decides; a `CEF₂` loss is never
accepted.

Travel time is separately bounded by a **fixed service standard**, identical at every budget
level:

```
ATT ≤ ATT(R₀) · (1 + 0.10) = 19.564 min
```

This is a promise to the passenger, not a monotonicity device. An earlier version tied the
ceiling to the previous budget level instead; that made the ATT–`p` curve monotone by
construction but cost 6–12 points of exposure, because it rejected redundancy gains that
came with a small time penalty. The fixed standard was measured to be the better trade.

### Stage 2 — Disruption

150 Monte Carlo events. Each event is applied **identically** to every network — the original
`R₀` and both designed solutions — using common random numbers. Nodes and links are always
drawn **from the road network**, never from the route set of one particular solution;
otherwise networks with different route counts would receive different damage and the
comparison would be void.

| Type | Share | Breaks infrastructure |
|---|---|---|
| `link_failure` | 37 % | yes — 1 / 2 / 4 roads cut |
| `demand_surge` | 22 % | no — negative control |
| `combined_disruption` | 17 % | two of the others |
| `node_removal` | 12 % | yes |
| `area_closure` | 12 % | yes |

Severity is drawn independently: mild 50 %, moderate 35 %, severe 15 %.

The shares above are the configured probabilities, not the counts in any one sample. With 150
draws this run produced 58 `link_failure`, 32 `combined_disruption`, 25 `demand_surge`, 18
`area_closure` and 17 `node_removal`, and 72 / 53 / 25 mild / moderate / severe.

`demand_surge` is retained as the contrast case: it raises demand without breaking anything,
so coverage is never lost and the coverage floor stays at zero. It separates what the design
withstands *structurally* from what it absorbs through spare capacity.

Two time points are recorded per run:

- **`metrics`** — *t₀*, the moment of failure. Lines stop where they meet a closed node or a
  cut link; nobody has planned a detour yet. This is the primary measure: what the design
  withstands on its own.
- **`metrics_after_detour`** — *t₁*, what an instant optimal detour would give. Reference only.

Each run also records **`coverage_floor_percent`**: the demand that *no* route set can serve
after the damage, either because it sits at a closed node or because the two endpoints are no
longer connected by road. Recovery is judged against this floor, not against zero.

When a node closes, its demand walks to the nearest open stop within 400 m at 4.8 km/h and the
walking time is charged.

### Stage 3 — Recovery

The agent replays each recorded disruption from the *t₀* state and greedily composes a
sequence of interventions (up to 5 steps, 8 packages built from 9 structural and operational
actions). Building the detour is **its** job and is paid for from its budget.

Two hard constraints, mirroring the design:

- **coverage never worsens**, and the target is the coverage floor
- spend ≤ `max(B_total, cost after damage)` — **no margin above the approved budget**

The coverage constraint is enforced at four independent points: inside the feasibility test,
in the candidate filter of the greedy loop, in the fallback pool of the single-plan selector,
and again at the moment a plan is applied. It is a constraint, never a term in the reward —
otherwise a cost-saving action can always buy reward by selling served demand.

The reward is the design objective, so all three stages measure the same thing and their
numbers are comparable.

---

## Metrics

| Symbol | Meaning |
|---|---|
| `ATT` | demand-weighted average travel time, over served demand |
| `DUN` | unserved demand (%) |
| `CEF` | demand-weighted path redundancy, discounted for overlap; a path counts only if its time ≤ `(1 + θ) · t_best`, with `θ = 0.5` |
| `CEF₂` | the same index capped at 2. A second independent path protects against a single link failure; a sixth adds nothing against the same event |
| `exposure` | share of demand whose OD pair has **one** acceptable path — the uninsured part of the network |

`CEF₂` is the optimisation surrogate: smooth enough to give the search a gradient. `exposure`
is the reporting metric: a strict count, directly interpretable.

Both `ATT` and `exposure` are computed over **served** demand. When recovery restores
previously disconnected pairs, both can rise — because more people are travelling, not because
service degraded.

---

## Results

### Baseline — the existing network `R₀`

15 routes, evaluated on the input data:

| ATT | CEF | CEF₂ | exposure | DUN | cost |
|---|---|---|---|---|---|
| 17.786 min | 1.1977 | 1.1827 | 68.08 % | 0 % | 32 101.84 € |

Two thirds of all demand travels on a single acceptable path. That is the problem the design
stage is given.

### Stage 1 — Design (`p = 0.20`, `B_design = 36 596.10 €`)

| Model | Scenario | ATT | CEF | CEF₂ | exposure | cost | % of `B_design` | routes |
|---|---|---|---|---|---|---|---|---|
| **v7** | S1_service | 18.852 | 1.9021 | 1.5981 | 31.84 % | 36 201 | 98.9 % | 13 |
| **v7** | S2_energy | 19.014 | 1.9344 | **1.6674** | **20.76 %** | 35 245 | 96.3 % | 13 |
| v6 | S1_service | 17.768 | 1.7763 | 1.5255 | 28.04 % | 36 557 | 99.9 % | 15 |
| v6 | S2_energy | 17.794 | 1.7679 | 1.5290 | 30.25 % | 36 490 | 99.7 % | 15 |

`DUN = 0` in all four. The structural action set (**v7**) buys more redundancy per euro in
both scenarios — higher `CEF` and `CEF₂` at a lower cost — and pays for it in travel time,
about 1.1 min. The local action set (**v6**) holds travel time essentially at the baseline.

Where the two disagree is *where* that redundancy lands. In S2, v7's extra redundancy reaches
the pairs that had none: exposure falls 9.5 points to 20.76 %, the lowest of any solution. In
S1 it does not — v7 has the higher `CEF₂` (1.5981 against 1.5255) and yet the **worse**
exposure (31.84 % against 28.04 %). Average redundancy is up while more demand is left on a
single path, because the structural actions concentrate alternatives on pairs that already
had some.

That is why both are reported. `CEF₂` measures how much redundancy exists; `exposure`
measures how evenly it is spread. A design can improve the first and worsen the second, and
only the pair of them says whether the network is actually less fragile.

### Stage 2 — Disruption (150 runs, common random numbers)

Mean unserved demand at *t₀*, by network:

| | `R₀` | v7 / S1 | v7 / S2 |
|---|---|---|---|
| **mean DUN at t₀** | 12.666 % | **7.797 %** | 9.318 % |

Mean coverage floor — demand that no route set can serve — is 2.165 %.

By disruption type (`R₀` / S1 / S2):

| Type | `R₀` | S1 | S2 |
|---|---|---|---|
| `area_closure` | 28.36 | 20.09 | 23.59 |
| `combined_disruption` | 21.47 | 14.81 | 17.57 |
| `node_removal` | 15.03 | 7.84 | 11.40 |
| `link_failure` | 7.70 | 3.46 | 3.74 |
| `demand_surge` | 0.00 | 0.00 | 0.00 |

By severity (`R₀` / S1 / S2): mild 9.42 / 5.16 / 6.54 · moderate 11.98 / 7.80 / 9.10 ·
severe 23.47 / 15.40 / 17.79.

### Stage 3 — Recovery

Measured over the episodes where damage left **recoverable** demand — DUN above the coverage
floor. S1: 103 of 150 · S2: 94 of 150.

| | S1_service | S2_energy |
|---|---|---|
| episodes where an action was taken | 96 (93.2 %) | 88 (93.6 %) |
| **recovered / recoverable** | **67.6 %** | **73.0 %** |
| severe episodes only | 54.9 % | 64.1 % |
| mean final cost | 37 296 € (96.8 % of `B_total`) | 36 440 € (94.6 %) |
| max final cost | 38 507 € | 38 480 € |
| **budget violations** | **0 / 150** | **0 / 150** |
| **episodes that worsened coverage** | **0 / 150** | **0 / 150** |
| exposure, before → after | 31.92 → 29.09 % | 22.48 → 21.37 % |

By disruption type (n / acted / recovered):

| Type | S1 | S2 |
|---|---|---|
| `node_removal` | 15 / 14 / 93.8 % | 14 / 14 / **100.0 %** |
| `link_failure` | 44 / 44 / 81.8 % | 38 / 37 / 78.9 % |
| `combined_disruption` | 26 / 26 / 67.1 % | 24 / 23 / 60.1 % |
| `area_closure` | 18 / 12 / 48.1 % | 18 / 14 / 76.0 % |
| `demand_surge` | — | — |

`demand_surge` does not appear in the table above: it destroys no infrastructure, so unserved
demand stays at zero and there is nothing to *recover*.

The agent is not idle there, though. In all 25 `demand_surge` episodes of each scenario it
applies the same single action, `corridor_relief`, and reaches the same final cost —
38 346 € in S1, 36 596 € in S2, both within `B_total`. Coverage stays at zero throughout;
what improves is travel time, by 0.51 min in S1 and 0.54 min in S2, at an average
2 144 € and 1 351 € per episode.

Two things follow, and both are worth stating plainly. First, the response is **identical
regardless of how large the surge is** — so the agent is not calibrating to the event; it is
taking the best structural improvement its budget allows. Second, that improvement was
available all along and the *design* could not afford it, because the design is capped at
`B_design` while recovery may spend up to `B_total`. The 5 % reserve is therefore not held
back for emergencies: it is spent on the first episode that offers any gain at all.

Whether that is the desired policy is a design decision, not a defect. Spending it buys real
service on a congested day; withholding it keeps capacity for the next structural failure.
Setting `GREEDY_MIN_GAIN` above zero, or gating recovery on `coverage_gap_before > 0`, would
restrict the agent to episodes where coverage was actually lost.

---

## Repository

```
v7.py             design — structural (SNA-based) action set, MAX_ROUTE_LEN = 18
v6.py             design — local action set, MAX_ROUTE_LEN = 14 (action-set comparison)
disruption.py     Monte Carlo resilience analysis
recovery.py       recovery agent
seekallpaths.py   path enumeration between OD pairs
RiveraTravel.txt  edge list: from  to  time(min)     — 286 directed links
RiveraDemand.txt  84 × 84 OD matrix, passengers/min  — scaled to 12 h in code
Rivera_coords.txt node  lat  lon  flag               — 84 nodes
results/          every number in the thesis and the paper — see results/README.md
tools/            make_R0_control.py · analyse_control.py · analyse_sensitivity.py
```

The structural action set of `v7.py` is four moves, each applied to a whole route:
`bridge_two_routes`, `shorten_route_via_shortest_path`,
`swap_terminal_to_high_demand_neighbor` and `insert_transfer_hub`. Headways are fixed, so
the design changes structure rather than frequency.

`v6.py` and the local-versus-structural comparison belong to the thesis; the ITS 2026 paper
reports the structural designs only.

Inline documentation is in Greek.

---

## Requirements

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

Optional, for the geographic route maps only — everything runs without them, and the map
falls back to a plain latitude/longitude plot:

```bash
pip install geopandas contextily shapely pyproj
```

`disruption.py` and `recovery.py` use the standard library alone.

---

## Usage

Run in order, from the repository folder. Each script writes to its own folder and the next
one reads it.

```bash
python v7.py           # → v7_2026-01-01_12-00-00/best_scenarios_baseline_v7.json
python v6.py           # → v6_2026-01-01_12-00-00/…   (optional, action-set comparison)
python disruption.py   # → monte_carlo_disruption_results_v7/
python recovery.py     # → v7_recovery_agent_..._results/
```

`v7.py` takes roughly 2–3 hours (800 episodes × 4 budget levels × 2 scenarios);
`disruption.py` about 20 minutes; `recovery.py` about 15 minutes.

**Selecting the input.** `disruption.py` and `recovery.py` ship with an absolute path in
`BEST_JSON_FILE` that will not exist on your machine. When a configured path is missing, both
scripts fall back to the most recently modified matching file and print which one they chose:

```
Το … δεν βρεθηκε.
Χρηση του: v7_2026-01-01_12-00-00/best_scenarios_baseline_v7.json
```

On a fresh clone with a single design run that fallback picks the right file. **As soon as
several runs exist, set the path explicitly** — otherwise the script silently analyses
whichever network was written last:

```python
# disruption.py
BEST_JSON_FILE = "v7_2026-01-01_12-00-00/best_scenarios_baseline_v7.json"

# recovery.py
BEST_JSON_FILE           = "v7_2026-01-01_12-00-00/best_scenarios_baseline_v7.json"
MONTE_CARLO_RESULTS_FILE = "monte_carlo_disruption_results_v7/v7_monte_carlo_disruption_results.json"
```

To analyse the **v6** network instead, point `BEST_JSON_FILE` at the `v6_…` folder and change
`TRAINING_MODULE_FILE` to `"v6.py"`.

---

## Outputs

**Design** — `best_scenarios_baseline_<tag>.json`, where `<tag>` is `v6` or `v7`, containing
`initial_solution` (the `R₀` reference) and, per scenario, `p`, `C0`, `B` (the design budget
after the reserve), `best_ep`, `best_att`, `best_CEF`, `best_CEF2`, `best_exposure`,
`best_dun`, `best_cost`, `best_energy` and `best_routes`. Alongside it, five metric-vs-`p`
plots (`att_vs_p`, `cef_vs_p` — which draws `CEF` and `CEF₂` together with the cap line —
`exposure_vs_p`, `energy_vs_p`, `cost_vs_p`, the last with the `B_total` and `B_design`
reference lines) and a route-set map, all tagged `_v6` or `_v7`.

**Disruption** — per-run JSON with `metrics` (t₀), `metrics_after_detour` (t₁),
`coverage_floor_percent`, and comparisons against each network's undisrupted baseline; four
summary CSVs grouped by type, location and severity.

**Recovery** — per-episode JSON and CSV with the chosen plan, the metrics before and after
intervention (including `CEF2` and `exposure`), the reward, and the coverage floor; five
summary CSVs including a breakdown by action.

> The recovery output folder is named `…_5actions_…` for continuity with earlier runs. The
> current action set has **nine** actions grouped into **eight** packages; the name is
> historical and does not describe the code.

---

## Key parameters

| Parameter | Value | File |
|---|---|---|
| `CEF_THETA` | 0.5 | `v6/v7.py` |
| `CEF_CAP` | 2.0 | `v6/v7.py` |
| `BUDGET_RESERVE` | 0.05 | `v6/v7.py` |
| `CEF2_TOL` | 0.005 | `v6/v7.py` |
| `ATT_TOLERANCE` | 0.10 | `v6/v7.py` |
| `STAGNATION_LIMIT` | 15 | `v6/v7.py` |
| `DEFAULT_MAX_ROUTE_LEN` | 18 (v7) / 14 (v6) | `v6/v7.py` |
| episodes per budget level | 800 | `v6/v7.py` |
| `alpha` / `gamma` / `epsilon` | 0.1 / 0.9 / 0.3 → 0.05 (decay 0.997) | `v6/v7.py` |
| `MONTE_CARLO_RUNS` | 150 | `disruption.py` |
| `RANDOM_SEED` | 42 | `disruption.py`, `recovery.py` |
| `WALK_ACCESS_RADIUS_M` / `WALK_SPEED_KMH` | 400 / 4.8 | `disruption.py` |
| `RECOVERY_START` | `"t0"` | `recovery.py` |
| `RECOVERY_BUDGET_MARGIN` | 0.00 | `recovery.py` |
| `GREEDY_MAX_STEPS` | 5 | `recovery.py` |
| `ENFORCE_COVERAGE_FLOOR` / `ENFORCE_NO_COVERAGE_LOSS` | `True` / `True` | `recovery.py` |
| `C_KM` / `C_H` / `E_KM` | 0.8 / 18.0 / 1.2 | all |
| `HEADWAY_MIN` / `OPERATING_HOURS` | 10.0 / 12.0 | all |

If recovery turns out too constrained, the lever is `BUDGET_RESERVE` in the design — more
reserve, weaker design. `RECOVERY_BUDGET_MARGIN` stays at zero so that the claim "without
increasing the budget" remains literally true.

---

## Reproducibility

Every stage is seeded and deterministic. The disruption stage uses common random numbers, so
run *k* applies the identical event to every network and comparisons are paired. Re-running
with the same seed and the same input files reproduces every figure in the results.

---

## Data

`RiveraTravel.txt`, `RiveraDemand.txt` and `Rivera_coords.txt` describe the road network,
the origin–destination demand and the stop coordinates of Rivera, Uruguay. They are included
so that the results above can be reproduced exactly.

---

## License

MIT — see [LICENSE](LICENSE). Applies to both the code and the included data files.

## Citation

```
Spyropoulos, D. (2026). Multi-Objective Resilient Urban Network Design & Disruption
Management under Budget Constraints: A Reinforcement Learning Framework for the Case of
Rivera, Uruguay. Diploma thesis, Department of Civil Engineering, University of Patras.
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).
