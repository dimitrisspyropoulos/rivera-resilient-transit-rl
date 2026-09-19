# -*- coding: utf-8 -*-
"""Σχεδιασμος δικτυου αστικων συγκοινωνιων με ενισχυτικη μαθηση - Rivera, Uruguay.

Συνολο ενεργειων: τοπικες παρεμβασεις, μεγιστο μηκος γραμμης 14 κομβοι.

Ο πρακτορας (tabular Q-learning) αναδιατασσει το υπαρχον δικτυο R0 υπο δυο
σκληρους περιορισμους: ολη η ζητηση εξυπηρετειται (DUN = 0) και το λειτουργικο
κοστος δεν ξεπερνα τον προυπολογισμο B. Εκτελειται για τεσσερα επιπεδα
προυπολογισμου p με warm start: η λυση καθε επιπεδου ξεκινα απο τη λυση του
προηγουμενου.

Δυο σεναρια:
  S1_service   r = dCEF + dATT + dCEF2
  S2_energy    r = dCEF + dATT + dCEF2 + dE
Οι οροι κανονικοποιουνται ως προς το αρχικο δικτυο R0. Η ενεργεια μετραει μονο
στο S2· η διαφορα των δυο σεναριων ειναι ακριβως το trade-off λειτουργικοτητας
και βιωσιμοτητας.

Δεικτες:
  CEF       σταθμισμενος με τη ζητηση πλεονασμος διαδρομων, με εκπτωση για
            επικαλυψη· μονοπατι μετραει μονο αν ο χρονος του <= (1+theta)*t_best
  CEF2      ο ιδιος δεικτης με ανω οριο CEF_CAP = 2. Η δευτερη ανεξαρτητη
            διαδρομη προστατευει απο αστοχια μιας συνδεσης· η εκτη δεν
            προσθετει τιποτα απεναντι στο ιδιο συμβαν
  exposure  % ζητησης της οποιας το ζευγος OD διαθετει ΜΙΑ μονο αποδεκτη
            διαδρομη - το ακαλυπτο μερος του δικτυου

Αρχεια στον ιδιο φακελο: RiveraTravel.txt, RiveraDemand.txt, Rivera_coords.txt,
seekallpaths.py
"""
import math
import numpy as np
import random
import copy
import collections
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter

try:
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point, LineString
except Exception:
    gpd = None
    ctx = None
    Point = None
    LineString = None
from typing import Dict, List, Tuple, Optional

from seekallpaths import seekpathfr

# ΕΛΕΓΧΟΣ: επιβεβαιωνει οτι φορτωθηκε το ΔΙΟΡΘΩΜΕΝΟ seekallpaths.
# Η εντολη `from seekallpaths import ...` ψαχνει ΠΑΝΤΑ αρχειο με ονομα
# ακριβως `seekallpaths.py`. Αν εχεις το διορθωμενο με αλλο ονομα
# (π.χ. seekallpaths.r2.py) ΔΕΝ φορτωνεται - φορτωνεται το παλιο.
def _check_seekallpaths():
    import inspect, seekallpaths as _sk
    try:
        _src = inspect.getsource(_sk.seekpathfr)
    except Exception:
        return
    if "sum(TT[" not in _src:
        print("=" * 78)
        print("*** ΠΡΟΣΟΧΗ: φορτωθηκε το ΠΑΛΙΟ, ΜΗ ΔΙΟΡΘΩΜΕΝΟ seekallpaths!")
        print(f"***       αρχειο: {getattr(_sk, '__file__', '?')}")
        print("***       Το διορθωμενο πρεπει να ονομαζεται ΑΚΡΙΒΩΣ  seekallpaths.py")
        print("***       (οχι seekallpaths.r2.py - το import δεν το βρισκει).")
        print("***       Τα αποτελεσματα θα εχουν τον ΜΗ διορθωμενο υπολογισμο χρονου.")
        print("=" * 78)
    else:
        print("seekallpaths: διορθωμενη εκδοση, ΟΚ")


_check_seekallpaths()

# ============================================================================
# [ΝΕΕΣ ΠΑΡΑΜΕΤΡΟΙ] Ολες οι υπολοιπες παραμετροι της μεθοδολογιας μενουν ιδιες.
# ============================================================================
# Κατωφλι αποδεκτου εναλλακτικου μονοπατιου για τον CEF:
#      δεκτο αν  χρονος <= (1 + CEF_THETA) * συντομοτερος χρονος.
#      Το §3.1.2 της διπλωματικης το δηλωνει ηδη· ο κωδικας δεν το ειχε.
CEF_THETA = 0.5

# Ποσα διαδοχικα βηματα χωρις βελτιωση πριν επιστρεψουμε στην καλυτερη λυση.
# Ανω οριο του περιορισμενου δεικτη CEF2. Με k = 2 ο στοχος γινεται
# «καθε ζευγος να εχει τουλαχιστον ΔΥΟ διαδρομες», που ειναι ακριβως η
# προστασια απεναντι στην αστοχια ΜΙΑΣ συνδεσης.
# Ετικετα του συνολου ενεργειων. Μπαινει στον φακελο και σε καθε αρχειο
# εξοδου, ωστε τα αποτελεσματα των δυο παραλλαγων να μη συγχεονται ποτε.
MODEL_TAG = "v6"

CEF_CAP = 2.0

# Μερος του προυπολογισμου που ΔΕΝ επιτρεπεται να δεσμευσει ο σχεδιασμος.
# Ειναι η εφεδρεια που ξοδευει ο πρακτορας αποκαταστασης οταν συμβει η ζημια:
# το δικτυο αντεχει την ακραια διαταραχη ΧΩΡΙΣ αυξηση του προυπολογισμου.
BUDGET_RESERVE = 0.05

# Σχετικη ανοχη στον CEF2. Μεσα σε αυτη τη ζωνη δυο λυσεις θεωρουνται ισοδυναμες
# ως προς τον πλεονασμο και αποφασιζει ο χρονος διαδρομης.
CEF2_TOL = 0.005

# Ανωτατη επιτρεπομενη επιδεινωση του μεσου χρονου διαδρομης εναντι του
# υπαρχοντος δικτυου R0. Ειναι προτυπο εξυπηρετησης, οχι στοχος: μεσα σε αυτο
# το περιθωριο ο πρακτορας ειναι ελευθερος να ανταλλαξει χρονο με πλεονασμο,
# που ειναι ακριβως το ζητουμενο trade-off. Δεσμευοντας τον χρονο πιο σφιχτα -
# π.χ. απαιτωντας μονοτονη πτωση ως προς p - η ανταλλαγη απαγορευεται και η
# εκτεθειμενη ζητηση μενει σημαντικα υψηλοτερη.
ATT_TOLERANCE = 0.10

STAGNATION_LIMIT = 15

# Μικρη αρνητικη ανταμοιβη για κινησεις που παραβιαζουν DUN=0 η το budget.
REJECT_PENALTY = -0.10

# Πληθος bins για τη διακριτοποιηση της κατασταστης ΣΧΕΤΙΚΑ με την R0.
STATE_BINS_ATT = 50
STATE_BINS_CEF = 20
STATE_BINS_COST = 20

# ===================== Global caches for speed =====================
ROUTE_METRIC_CACHE: Dict[Tuple[int, ...], Tuple[float, float]] = {}
ATTACK_EVAL_CACHE: Dict[Tuple[Tuple[int, ...], ...], Tuple[float, float, float]] = {}
SOLUTION_EVAL_CACHE: Dict[Tuple, Tuple[float, float, float, float, float, Optional[float], Tuple]] = {}
INITIAL_FEASIBLE_ROUTESET_CACHE: Dict[str, List[List[int]]] = {}


def canonical_route(route: List[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in route if x != -1)


def canonical_routeset(routeset: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    uniq = {canonical_route(r) for r in routeset if len(r) > 0}
    return tuple(sorted(uniq))


# ===================== 0. Rivera "bibliography-inspired" initial routeset =====
# NOTE:
# Rivera benchmark does not ship with an official published "best" routeset the way
# Mandl does. To operationalize paper [7] (resilience via redundancy), we seed a
# skeleton routeset: (i) a backbone through the dense core, (ii) coverage routes for
# peripheral clusters, and (iii) at least one redundancy/bridge corridor.

# Node IDs in the Rivera files are 1..84 and are kept 1-based throughout the file.
RIVERA_ROUTES_ART7_SEED_1BASED = [
    # Backbone through core (dense grid)
    [9, 12, 13, 15, 16, 14, 10, 8, 5, 4, 3],
    # East / NE coverage
    [32, 33, 34, 67, 68, 69, 64, 63, 62],
    # South axis
    [70, 73, 75, 76, 77, 78, 79, 84],
    # West connector into core
    [1, 2, 3, 7, 9],
    # Redundancy bridge corridor
    [23, 24, 27, 28, 29, 30, 31],
]

RIVERA_ROUTES_ART7_SEED = copy.deepcopy(RIVERA_ROUTES_ART7_SEED_1BASED)


# ===================== 1. Load network Rivera (TT, TD) ========================

def _infer_n_from_coords(coords_file: str) -> int:
    mx = 0
    with open(coords_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                node_id = int(float(parts[0]))
                mx = max(mx, node_id)
            except Exception:
                continue
    return mx


def load_rivera(
    travel_file: str = "RiveraTravel.txt",
    demand_file: str = "RiveraDemand.txt",
    coords_file: Optional[str] = "Rivera_coords.txt",
    n_nodes: Optional[int] = None,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], np.ndarray, np.ndarray]:
    """
    RiveraTravel.txt is an edge list: from  to  time (1-based node ids).
    RiveraDemand.txt is an NxN matrix (space-separated floats).

    All returned structures follow the same 1-based logic:
      - TT, TD are (N+1)x(N+1) arrays with row/column 0 unused
      - TT_dict, TD_dict have keys 1..N
    """
    TD_raw = np.loadtxt(demand_file)
    # Rivera demand values are given as passengers per minute; convert
    # to total passengers over the 12-hour operating horizon.
    TD_raw = TD_raw * 12.0 * 60.0
    if n_nodes is not None:
        N = int(n_nodes)
        if TD_raw.shape[0] != N:
            N = TD_raw.shape[0]
    elif coords_file is not None:
        N = _infer_n_from_coords(coords_file)
        if TD_raw.shape[0] != N:
            N = TD_raw.shape[0]
    else:
        N = TD_raw.shape[0]

    TT = np.full((N + 1, N + 1), float("inf"), dtype=float)
    TD = np.zeros((N + 1, N + 1), dtype=float)

    for i in range(1, N + 1):
        TT[i, i] = 0.0

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            TD[i, j] = float(TD_raw[i - 1, j - 1])

    with open(travel_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, t = line.split()
            i = int(float(a))
            j = int(float(b))
            if 1 <= i <= N and 1 <= j <= N:
                TT[i, j] = float(t)

    TT_dict: Dict[int, Dict[int, float]] = {}
    TD_dict: Dict[int, Dict[int, float]] = {}

    for i in range(1, N + 1):
        TT_dict[i] = {}
        TD_dict[i] = {}
        for j in range(1, N + 1):
            if i == j:
                continue
            if TT[i, j] != float("inf") and TT[i, j] != 0:
                TT_dict[i][j] = TT[i, j]
            if TD[i, j] > 0:
                TD_dict[i][j] = TD[i, j]

    return TT_dict, TD_dict, TT, TD


# ===================== 2. Helper: Shortest Path (Dijkstra-like) ===============

def shortestPath(graph: Dict[int, Dict[int, float]], start: int, end: int) -> Tuple[float, List[int]]:
    import heapq
    queue = [(0.0, start, [])]
    seen = set()
    while queue:
        cost, v, path = heapq.heappop(queue)
        if v in seen:
            continue
        path = path + [v]
        seen.add(v)
        if v == end:
            return cost, path
        for nxt, c in graph.get(v, {}).items():
            if nxt not in seen:
                heapq.heappush(queue, (cost + c, nxt, path))
    return float("inf"), []


def make_route_feasible(route, TT_dict, max_len=14):
    """Repair a route so every consecutive pair is a valid directed edge.
    If an edge is missing, we splice in the shortest path nodes (excluding the start node).
    If no path exists, we truncate at the last feasible node.
    """
    if not route or len(route) < 2:
        return route

    repaired = [route[0]]
    for k in range(1, len(route)):
        a = repaired[-1]
        b = route[k]
        if b in TT_dict.get(a, {}):
            repaired.append(b)
        else:
            cost, path = shortestPath(TT_dict, a, b)
            if not path or cost == float("inf"):
                break
            # splice path excluding the first node (a)
            for node in path[1:]:
                if node != repaired[-1]:
                    repaired.append(node)
        if len(repaired) >= max_len:
            repaired = repaired[:max_len]
            break

    # remove immediate duplicates
    cleaned = [repaired[0]]
    for n in repaired[1:]:
        if n != cleaned[-1]:
            cleaned.append(n)
    return cleaned


def make_routeset_feasible(routeset, TT_dict, max_len=14):
    return [make_route_feasible(r, TT_dict, max_len=max_len) for r in routeset]


# ===================== 3. Evaluate Attacks/Redundancy =========================


def evaluateattacks(routeset, TT_dict, TD_dict, tt_matrix):
    """ΣΥΜΒΑΤΟΤΗΤΑ: επιστρεφει ΑΚΡΙΒΩΣ τις 6 τιμες που επεστρεφε παντα.

    Τα disruption.py και recovery.py καλουν
        _, _, _, dun, att, cef = evaluateattacks(...)
    Αν αλλαζε ο αριθμος των τιμων, θα εσκαγαν ολα. Ο νεος δεικτης δινεται απο
    την evaluateattacks_ext."""
    d0, d1, d2, dun, att, cef, _cef2, _exp = evaluateattacks_ext(
        routeset, TT_dict, TD_dict, tt_matrix)
    return d0, d1, d2, dun, att, cef


def evaluateattacks_ext(routeset, TT_dict, TD_dict, tt_matrix):
    """Compute dun, att, CEF for a routeset using 1-based Rivera node ids."""
    routeset_key = canonical_routeset(routeset)
    cached = ATTACK_EVAL_CACHE.get(routeset_key)
    if cached is not None:
        dun, att, CEF_ave, CEF2_ave, exposure_pct = cached
        return 0.0, 0.0, 0.0, dun, att, CEF_ave, CEF2_ave, exposure_pct

    # Τα BC / WBC / EF / UEF υπολογιζονταν σε καθε κληση αλλα ΔΕΝ
    # επιστρεφονταν και δεν χρησιμοποιουνταν πουθενα. Αφαιρεθηκαν.
    # Τα αποτελεσματα ειναι ταυτοσημα· η αξιολογηση ~4% ταχυτερη.
    CEF = {}
    CEF2 = {}                 # περιορισμενος πλεονασμος
    exposure = 0.0            # ζητηση που εξαρταται απο ΜΙΑ διαδρομη
    totalpas = sum(sum(v.values()) for v in TD_dict.values())

    n_nodes = len(tt_matrix) - 1
    for node in range(1, n_nodes + 1):
        CEF[node] = {}
        CEF2[node] = {}

    routeset = copy.deepcopy([list(t) for t in set(tuple(x) for x in routeset)])
    routeset = copy.deepcopy([[i for i in r if i != -1] for r in routeset])

    d0 = d1 = d2 = dun = att = 0.0

    trstops = {}
    for id1, r1 in enumerate(routeset):
        trstops[id1] = {}
        for id2, r2 in enumerate(routeset):
            trstops[id1][id2] = list(set(r1).intersection(r2)) if r2 != r1 else []

    seenpairs = set()

    for ori in TD_dict:
        for (dst, _) in TD_dict[ori].items():
            pair = (ori, dst)
            if pair in seenpairs:
                continue

            sp = shortestPath(TT_dict, ori, dst)
            doit = 0
            paths = seekpathfr(0, ori, dst, routeset, tt_matrix, trstops, sp, doit)

            # FIX: RiveraDemand.txt is NOT symmetric (TD[ori][dst] != TD[dst][ori]
            # for ~67 of 311 OD pairs). The previous version processed each
            # unordered pair once (skipping the reverse direction entirely via
            # seenpairs.add((dst, ori))) and multiplied by 2 to approximate
            # "both directions", which silently discarded ~20% of total demand
            # mass and mis-weighted the rest. Each directed (ori, dst) pair with
            # TD_dict[ori][dst] > 0 is now evaluated independently, with no x2
            # factor, so both directions of every OD pair contribute their own
            # real demand and their own real shortest path / route-finding.
            if (not paths) or (paths[0][2] > 2) or (paths[0][0] >= 1e7):
                dun += 100 * TD_dict[ori][dst]
            elif paths[0][2] == 0:
                d0 += 100 * TD_dict[ori][dst]
            elif paths[0][2] == 1:
                d1 += 100 * TD_dict[ori][dst]
            elif paths[0][2] == 2:
                d2 += 100 * TD_dict[ori][dst]

            if paths and paths[0][0] < 1e7:
                att += paths[0][0] * TD_dict[ori][dst]

            if paths and paths[0][0] < 1e7:
                paths.sort(key=itemgetter(0, 2))
                # Ο ελεγχος 1e7 γινοταν ΜΟΝΟ στο paths[0]· ολα τα υπολοιπα
                #      μπαιναν στον CEF, ακομη και μονοπατια με ΑΠΕΙΡΟ χρονο
                #      (945 απο 25859 στο S2 -> 152 απο τα 378 ζευγη OD).
                # Εφαρμογη του κατωφλιου (1+θ) που δηλωνει το §3.1.2.
                t_best = paths[0][0]
                acceptable = [a for a in paths
                              if a[0] < 1e7 and a[0] <= (1.0 + CEF_THETA) * t_best]
                allp = set(tuple(a[1]) for a in acceptable)
                path_edges = [[(l[n], l[n + 1]) for n in range(len(l) - 1)] for l in allp]
                unique_pathedges = [x for L in path_edges for x in L]
                edgecounts = Counter(unique_pathedges)

                overlap_in = 0.0
                for path in path_edges:
                    overlap_in += sum((1 / edgecounts[x]) for x in path) / max(1, len(path))

                # Only write the (ori, dst) entry now - the reverse direction
                # (dst, ori), if it also has demand, gets its own independent
                # entry when it is processed as its own pair.
                CEF[ori][dst] = TD_dict[ori][dst] * overlap_in

                # Με ανω οριο, ο στοχος σταματα να ανταμειβει το πυκνωμα
                # εκει που ειναι ηδη πυκνα και ανταμειβει τη διασωση των
                # ζευγων που εξαρτωνται απο μια μονο διαδρομη.
                CEF2[ori][dst] = TD_dict[ori][dst] * min(overlap_in, CEF_CAP)
                # Ζητηση που εξαρταται απο ΜΙΑ και μονο διαδρομη.
                if len(allp) <= 1:
                    exposure += TD_dict[ori][dst]

                # Οι ενημερωσεις WBC / EF / UEF αφαιρεθηκαν (νεκρος κωδικας).

            seenpairs.add(pair)

    if totalpas > 0:
        d0 /= totalpas
        d1 /= totalpas
        d2 /= totalpas
        dun /= totalpas
    else:
        d0 = d1 = d2 = dun = 0.0

    sat = ((100 - dun) / 100.0) * totalpas
    att = att / sat if sat > 0 else float("inf")

    CEF_sum = sum(x for counter in CEF.values() for x in counter.values())
    CEF_ave = CEF_sum / totalpas if totalpas > 0 else 0.0
    # Ο περιορισμενος δεικτης και η εκτεθειμενη ζητηση (%).
    CEF2_sum = sum(x for counter in CEF2.values() for x in counter.values())
    CEF2_ave = CEF2_sum / totalpas if totalpas > 0 else 0.0
    exposure_pct = 100.0 * exposure / totalpas if totalpas > 0 else 0.0

    ATTACK_EVAL_CACHE[routeset_key] = (dun, att, CEF_ave, CEF2_ave, exposure_pct)
    return d0, d1, d2, dun, att, CEF_ave, CEF2_ave, exposure_pct


# ========================== 4. Solution & Actions ============================

class Solution:
    """Solution container with tabular Q-learning (state-action table).

    We store Q as a mapping:  Q[state_key][action_index] -> value
    where state_key is a discretized tuple derived from (CEF, ATT, DUN).

    This is a *state x action* table (tabular RL), but implemented as a dict to
    handle a large/combinatorial state space.
    """

    def __init__(self, routes, actions):
        self.Routes = copy.deepcopy(routes)
        self.actions = list(actions)
        self.nA = len(self.actions)

        # Q[state_key] = np.array(nA)
        self.Q = collections.defaultdict(lambda: np.zeros(self.nA, dtype=float))

        self.gamma = 0.90
        self.epsilon = 0.30  # exploration

        # Latest evaluated metrics
        self.att = None
        self.CEF = None
        self.CEF2 = None        # περιορισμενος πλεονασμος (min(m, CEF_CAP))
        self.exposure = None    # % ζητησης που εξαρταται απο ΜΙΑ διαδρομη
        self.dun = None

        # Current discretized state key
        self.state_key = None

    @staticmethod
    def make_state_key(att, cef, dun):
        # Discretize to keep the table compact.
        # - ATT: 0.1 resolution
        # - CEF: 0.01 resolution
        # - DUN: integer (it is already scaled by demand in this code)
        return (round(float(att), 1), round(float(cef), 2), int(round(float(dun))))


def neighbors(TT_dict, node):
    return list(TT_dict.get(node, {}).keys())


# ---------------- Action constraints ----------------
DEFAULT_MAX_ROUTE_LEN = 14   # prevents unrealistic long routes
DEFAULT_MIN_ROUTE_LEN = 3
DEFAULT_MAX_INSERT_TRIES = 40

def _route_is_edge_feasible(route: List[int], TT_dict: Dict[int, Dict[int, float]]) -> bool:
    """Every consecutive pair must be a directed edge in TT_dict."""
    if len(route) < 2:
        return True
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        if b not in TT_dict.get(a, {}):
            return False
    return True

def _route_sanitize(route: List[int]) -> List[int]:
    """Remove consecutive duplicates; keep order."""
    if not route:
        return route
    out = [route[0]]
    for x in route[1:]:
        if x != out[-1]:
            out.append(x)
    return out

def _routeset_sanitize(routeset: List[List[int]]) -> List[List[int]]:
    rs = []
    seen = set()
    for r in routeset:
        r2 = tuple(_route_sanitize([int(x) for x in r if x != -1]))
        if len(r2) == 0:
            continue
        if r2 in seen:
            continue
        seen.add(r2)
        rs.append(list(r2))
    return rs


def add_redundant_segment(sol, TT_dict, max_len=DEFAULT_MAX_ROUTE_LEN):
    """
    Insert an intermediate node w between u->v where u->w and w->v exist,
    and w not already in the route. Keeps directed feasibility.
    """
    new_sol = copy.deepcopy(sol)
    if not new_sol.Routes:
        return sol, 0
    ri = random.randint(0, len(new_sol.Routes) - 1)
    route = list(new_sol.Routes[ri])

    if len(route) < 2 or len(route) >= max_len:
        return sol, 0

    pos = random.randint(1, len(route) - 1)
    u, v = route[pos - 1], route[pos]

    # Candidates w: u->w and w->v exist (directed)
    cand = []
    for w in neighbors(TT_dict, u):
        if w == u or w == v:
            continue
        if w in route:
            continue
        if v in TT_dict.get(w, {}):
            cand.append(w)

    if not cand:
        return sol, 0

    route.insert(pos, random.choice(cand))
    if len(route) > max_len:
        return sol, 0
    if not _route_is_edge_feasible(route, TT_dict):
        return sol, 0

    new_sol.Routes[ri] = route
    new_sol.Routes = _routeset_sanitize(new_sol.Routes)
    return new_sol, 1


def reroute_via_alternative_node(sol, TT_dict, max_len=DEFAULT_MAX_ROUTE_LEN):
    """
    Replace internal node i with w such that u->w and w->v exist.
    Keeps directed feasibility.
    """
    new_sol = copy.deepcopy(sol)
    if not new_sol.Routes:
        return sol, 0
    ri = random.randint(0, len(new_sol.Routes) - 1)
    route = list(new_sol.Routes[ri])

    if len(route) < 3:
        return sol, 0

    pos = random.randint(1, len(route) - 2)
    u, i, v = route[pos - 1], route[pos], route[pos + 1]

    cand = []
    for w in neighbors(TT_dict, u):
        if w in (u, v, i):
            continue
        if v in TT_dict.get(w, {}):
            # allow if w not in route, or it is exactly i (no change)
            if (w not in route) or (w == i):
                cand.append(w)

    if not cand:
        return sol, 0

    route[pos] = random.choice(cand)
    route = _route_sanitize(route)
    if len(route) > max_len:
        return sol, 0
    if not _route_is_edge_feasible(route, TT_dict):
        return sol, 0

    new_sol.Routes[ri] = route
    new_sol.Routes = _routeset_sanitize(new_sol.Routes)
    return new_sol, 1


def extend_route_for_coverage(sol, TT_dict, max_len=DEFAULT_MAX_ROUTE_LEN):
    """
    Extend at one end by ONE node that is a feasible directed neighbor.
    """
    new_sol = copy.deepcopy(sol)
    if not new_sol.Routes:
        return sol, 0
    ri = random.randint(0, len(new_sol.Routes) - 1)
    route = list(new_sol.Routes[ri])

    if len(route) == 0 or len(route) >= max_len:
        return sol, 0

    # Try a few times to avoid dead-ends
    for _ in range(10):
        if random.random() < 0.5:
            end = route[-1]
            cands = [w for w in neighbors(TT_dict, end) if w not in route]
            if not cands:
                continue
            route2 = route + [random.choice(cands)]
        else:
            st = route[0]
            # Need w->st for prepending while keeping direction
            cands = [w for w in TT_dict.keys() if st in TT_dict.get(w, {}) and w not in route]
            if not cands:
                continue
            route2 = [random.choice(cands)] + route

        if len(route2) <= max_len and _route_is_edge_feasible(route2, TT_dict):
            new_sol.Routes[ri] = route2
            new_sol.Routes = _routeset_sanitize(new_sol.Routes)
            return new_sol, 1

    return sol, 0


def prune_low_value_node(sol, TT_dict, min_len=DEFAULT_MIN_ROUTE_LEN):
    """
    Remove an internal node if shortcut edge exists (u->v).
    Keeps minimum length.
    """
    new_sol = copy.deepcopy(sol)
    if not new_sol.Routes:
        return sol, 0
    ri = random.randint(0, len(new_sol.Routes) - 1)
    route = list(new_sol.Routes[ri])

    if len(route) <= min_len:
        return sol, 0

    positions = list(range(1, len(route) - 1))
    random.shuffle(positions)
    for pos in positions:
        u, mid, v = route[pos - 1], route[pos], route[pos + 1]
        if v in TT_dict.get(u, {}):
            route2 = route[:pos] + route[pos + 1:]
            route2 = _route_sanitize(route2)
            if len(route2) < min_len:
                continue
            if _route_is_edge_feasible(route2, TT_dict):
                new_sol.Routes[ri] = route2
                new_sol.Routes = _routeset_sanitize(new_sol.Routes)
                return new_sol, 1

    return sol, 0


ACTION_FUNCS = {
    "add_redundant_segment": add_redundant_segment,
    "reroute_via_alternative_node": reroute_via_alternative_node,
    "extend_route_for_coverage": extend_route_for_coverage,
    "prune_low_value_node": prune_low_value_node,
}


# ====================== 5. Physical distance / cost / energy helpers ======================

def read_coords_latlon(coords_file: str = "Rivera_coords.txt") -> Dict[int, Tuple[float, float]]:
    """Read Rivera node coordinates as {node_id: (lat, lon)} using 1-based ids."""
    coords: Dict[int, Tuple[float, float]] = {}
    with open(coords_file, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue
            node = int(float(parts[0]))
            lat = float(parts[1])
            lon = float(parts[2])
            coords[node] = (lat, lon)
    return coords


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def build_directed_dist_km(TT_dict: Dict[int, Dict[int, float]], coords: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[int, float]]:
    """Directed distance matrix in km for the same directed edges that exist in TT_dict."""
    dist_km: Dict[int, Dict[int, float]] = {}
    for i, nbrs in TT_dict.items():
        dist_km[i] = {}
        if i not in coords:
            continue
        lat1, lon1 = coords[i]
        for j in nbrs:
            if j not in coords:
                continue
            lat2, lon2 = coords[j]
            dist_km[i][j] = haversine_km(lat1, lon1, lat2, lon2)
    return dist_km


def get_route_length_time(route: List[int], TT_dict, dist_km) -> Tuple[float, float]:
    route_key = canonical_route(route)
    cached = ROUTE_METRIC_CACHE.get(route_key)
    if cached is not None:
        return cached

    if len(route_key) < 2:
        ROUTE_METRIC_CACHE[route_key] = (0.0, 0.0)
        return 0.0, 0.0

    total_km = 0.0
    total_min = 0.0
    for k in range(len(route_key) - 1):
        a, b = route_key[k], route_key[k + 1]
        if b in dist_km.get(a, {}) and b in TT_dict.get(a, {}):
            total_km += dist_km[a][b]
            total_min += TT_dict[a][b]
        else:
            c, sp = shortestPath(TT_dict, a, b)
            if c == float("inf") or len(sp) < 2:
                continue
            total_min += c
            for m in range(len(sp) - 1):
                u, v = sp[m], sp[m + 1]
                total_km += dist_km.get(u, {}).get(v, 0.0)

    ROUTE_METRIC_CACHE[route_key] = (total_km, total_min)
    return total_km, total_min


def route_length_km(route: List[int], TT_dict, dist_km) -> float:
    """Directed route length in km; cached per route tuple."""
    total_km, _ = get_route_length_time(route, TT_dict, dist_km)
    return total_km


def route_time_min(route: List[int], TT_dict, dist_km=None) -> float:
    if dist_km is None:
        total = 0.0
        if len(route) < 2:
            return 0.0
        for k in range(len(route) - 1):
            a, b = route[k], route[k + 1]
            if b in TT_dict.get(a, {}):
                total += TT_dict[a][b]
            else:
                c, _ = shortestPath(TT_dict, a, b)
                total += c
        return total
    _, total_min = get_route_length_time(route, TT_dict, dist_km)
    return total_min


def compute_vkm_vh(routeset: List[List[int]], TT_dict, dist_km,
                   operating_hours: float = 12.0,
                   headway_min: float = 10.0,
                   assume_bidirectional: bool = True) -> Tuple[float, float]:
    """Return total vehicle-km and vehicle-hours over the operating horizon."""
    trips_per_route_per_direction = (operating_hours * 60.0) / float(headway_min)
    direction_factor = 2.0 if assume_bidirectional else 1.0
    trips = trips_per_route_per_direction * direction_factor

    vkm = 0.0
    vh = 0.0
    for r in routeset:
        Lr, Tr_min = get_route_length_time(r, TT_dict, dist_km)
        vkm += Lr * trips
        vh += (Tr_min / 60.0) * trips
    return vkm, vh


def compute_operating_cost(routeset: List[List[int]], TT_dict, dist_km,
                           c_km: float, c_h: float,
                           operating_hours: float = 12.0,
                           headway_min: float = 10.0,
                           assume_bidirectional: bool = True) -> Tuple[float, float, float]:
    vkm, vh = compute_vkm_vh(
        routeset, TT_dict, dist_km,
        operating_hours=operating_hours,
        headway_min=headway_min,
        assume_bidirectional=assume_bidirectional,
    )
    cost = float(c_km) * float(vkm) + float(c_h) * float(vh)
    return cost, vkm, vh


def compute_energy_kwh(routeset: List[List[int]], TT_dict, dist_km,
                       e_km: float,
                       operating_hours: float = 12.0,
                       headway_min: float = 10.0,
                       assume_bidirectional: bool = True) -> float:
    vkm, _ = compute_vkm_vh(
        routeset, TT_dict, dist_km,
        operating_hours=operating_hours,
        headway_min=headway_min,
        assume_bidirectional=assume_bidirectional,
    )
    return float(e_km) * float(vkm)


def under_operating_budget(routeset: List[List[int]], TT_dict, dist_km,
                           B: float, c_km: float, c_h: float,
                           operating_hours: float = 12.0,
                           headway_min: float = 10.0,
                           assume_bidirectional: bool = True) -> bool:
    cost, _, _ = compute_operating_cost(
        routeset, TT_dict, dist_km, c_km, c_h,
        operating_hours=operating_hours,
        headway_min=headway_min,
        assume_bidirectional=assume_bidirectional,
    )
    return cost <= (B + 1e-9)


# ====================== 6. RL evaluation / reward ====================

def make_state_key_extended(att: float, cef: float, dun: float, energy_kwh: Optional[float],
                            use_energy: bool, reference: Optional[dict] = None,
                            budget: Optional[float] = None, cost: Optional[float] = None):
    """Διακριτοποιηση της καταστασης ΣΧΕΤΙΚΑ με την αρχικη λυση R0.

    ΠΡΟΒΛΗΜΑ: με απολυτες τιμες, και ειδικα με την ενεργεια στρογγυλοποιημενη
    στο 0.1 kWh (τιμες ~30000), ΚΑΘΕ λυση εδινε μοναδικη κατασταση. Ο πινακας Q
    δεν ξαναεβλεπε ποτε την ιδια κατασταση, αρα max_a Q(s',a) = 0 παντα και
    το S2 ηταν στην πραξη τυχαια αναζητηση. Μετρημενο: 35 επισκεψεις -> 34
    μοναδικες καταστασεις. Με τα bins: 5 -> 81 καταστασεις με επαναληψεις.

    Αν δεν δοθει reference/budget/cost διατηρειται η ΠΑΛΙΑ συμπεριφορα, ωστε
    τα disruption.py / recovery.py να δουλευουν αμεταβλητα.
    """
    if reference is None or budget is None or cost is None:
        if use_energy:
            return (round(float(att), 1), round(float(cef), 2), int(round(float(dun))), round(float(energy_kwh or 0.0), 1))
        return (round(float(att), 1), round(float(cef), 2), int(round(float(dun))))

    return (int(float(att) / max(float(reference["att"]), 1e-9) * STATE_BINS_ATT),
            int(float(cef) / max(float(reference["CEF"]), 1e-9) * STATE_BINS_CEF),
            int(float(cost) / max(float(budget), 1e-9) * STATE_BINS_COST))


def evaluate_solution(sol, TT_dict, TD_dict, tt, dist_km,
                      c_km: float, c_h: float, e_km: float,
                      operating_hours: float = 12.0,
                      headway_min: float = 10.0,
                      assume_bidirectional: bool = True,
                      use_energy_in_state: bool = False,
                      compute_energy: bool = True,
                      state_reference: Optional[dict] = None,
                      state_budget: Optional[float] = None):
    # Τα state_reference / state_budget ειναι ΠΡΟΑΙΡΕΤΙΚΑ. Οταν λειπουν,
    # η συναρτηση συμπεριφερεται ακριβως οπως πριν (συμβατοτητα με τα
    # disruption.py / recovery.py, που την καλουν χωρις αυτα).
    routeset_key = canonical_routeset(sol.Routes)
    cache_key = (
        routeset_key,
        float(c_km), float(c_h), float(e_km),
        float(operating_hours), float(headway_min),
        bool(assume_bidirectional), bool(use_energy_in_state), bool(compute_energy),
        float(state_budget) if state_budget is not None else 0.0,
    )
    cached = SOLUTION_EVAL_CACHE.get(cache_key)
    if cached is not None:
        att, CEF_ave, dun, cost, vkm, vh, energy_kwh, state_key, CEF2_ave, exposure = cached
        sol.att = att
        sol.CEF = CEF_ave
        sol.CEF2 = CEF2_ave
        sol.exposure = exposure
        sol.dun = dun
        sol.cost = cost
        sol.vkm = vkm
        sol.vh = vh
        sol.energy_kwh = energy_kwh
        sol.state_key = state_key
        return att, CEF_ave, dun

    d0, d1, d2, dun, att, CEF_ave, CEF2_ave, exposure = evaluateattacks_ext(
        sol.Routes, TT_dict, TD_dict, tt)
    sol.att = att
    sol.CEF = CEF_ave
    sol.CEF2 = CEF2_ave          # περιορισμενος πλεονασμος
    sol.exposure = exposure      # ζητηση με ΜΙΑ διαδρομη (%)
    sol.dun = dun
    sol.vkm, sol.vh = compute_vkm_vh(
        sol.Routes, TT_dict, dist_km,
        operating_hours=operating_hours,
        headway_min=headway_min,
        assume_bidirectional=assume_bidirectional,
    )
    sol.cost = float(c_km) * float(sol.vkm) + float(c_h) * float(sol.vh)
    sol.energy_kwh = (float(e_km) * float(sol.vkm)) if compute_energy else None
    sol.state_key = make_state_key_extended(att, CEF_ave, dun, sol.energy_kwh, use_energy_in_state,
                                            reference=state_reference, budget=state_budget,
                                            cost=sol.cost)
    SOLUTION_EVAL_CACHE[cache_key] = (att, CEF_ave, dun, sol.cost, sol.vkm, sol.vh,
                                      sol.energy_kwh, sol.state_key, CEF2_ave, exposure)
    return att, CEF_ave, dun


def compute_reward(prev_sol, new_sol, norm: Dict[str, float], scenario_name: str) -> float:
    """Ανταμοιβη κανονικοποιημενη ως προς την αρχικη λυση R0, χωρις χειροκινητα βαρη.

        S1_service   r = dCEF + dATT + dCEF2
        S2_energy    r = dCEF + dATT + dCEF2 + dE

    Ο ορος dCEF2 στοχευει τα ζευγη που εξαρτωνται απο μια μονο διαδρομη. Μενει
    ομως και ο dCEF: ο CEF2 ειναι ανω φραγμενος, οποτε μονος του βαθμολογει με
    μηδεν καθε κινηση σε ηδη καλυμμενο ζευγος και η αναζητηση δεν εχει κλιση να
    ακολουθησει.
    """
    eps = 1e-9
    dCEF = (float(new_sol.CEF) - float(prev_sol.CEF)) / max(norm["CEF"], eps)
    dATT = (float(prev_sol.att) - float(new_sol.att)) / max(norm["ATT"], eps)
    dCEF2 = (float(new_sol.CEF2 or 0.0) - float(prev_sol.CEF2 or 0.0)) / max(norm["CEF2"], eps)
    r = dCEF + dATT + dCEF2
    if scenario_name == "S2_energy":
        dE = (float(prev_sol.energy_kwh) - float(new_sol.energy_kwh)) / max(norm["E"], eps)
        r += dE
    return float(r)


# ====================== 7. Training with operating budget + scenarios =======================


def choose_action(sol):
    """ε-greedy action selection from the state-action Q-table."""
    if sol.state_key is None:
        return random.choice(sol.actions)
    if random.random() < sol.epsilon:
        return random.choice(sol.actions)
    q_row = sol.Q[sol.state_key]
    max_q = np.max(q_row)
    best_candidates = np.flatnonzero(q_row == max_q)
    best_idx = int(random.choice(best_candidates))
    return sol.actions[best_idx]


def decay_epsilon(sol, decay_rate=0.997, min_epsilon=0.05):
    """Gradually reduce exploration while preserving a minimum epsilon."""
    sol.epsilon = max(min_epsilon, sol.epsilon * decay_rate)


def q_update(sol, state_key, action_idx, reward, next_state_key, alpha):
    """Tabular Q-learning update on a state-action table."""
    q_sa = sol.Q[state_key][action_idx]
    target = float(reward) + sol.gamma * float(np.max(sol.Q[next_state_key]))
    sol.Q[state_key][action_idx] = q_sa + alpha * (target - q_sa)


def train_for_budget_scenario(
    p,
    scenario_name,
    TT_dict,
    TD_dict,
    tt,
    dist_km,
    initial_routes,
    actions,
    c_km,
    c_h,
    e_km,
    operating_hours=12.0,
    headway_min=10.0,
    assume_bidirectional=True,
    Episodes=300,
    alpha=0.1,
    seed=42,
    verbose=True,
    warm_start_sol=None,
    reference=None,
    att_ceiling=None,
):
    """Train a separate RL run for a given budget scenario p and scenario name.

    If warm_start_sol is given (the Solution returned as best_sol from the
    previous, smaller p level), training continues from there: it starts
    from that solution's routeset AND its learned Q-table/epsilon, instead
    of resetting to the original baseline routeset with a fresh Q-table.

    Το `reference` ειναι dict {"cost","att","CEF","E"} απο την ΑΡΧΙΚΗ λυση R0.
    Οταν δινεται, το C0 και οι τιμες κανονικοποιησης παιρνονται ΠΑΝΤΑ απο εκει,
    οχι απο τη warm-started λυση. Το warm start παραμενει ακριβως ως ειχε: αλλαζει
    μονο η τιμη αναφορας. Χωρις αυτο, το C0 ανεβαινε σε καθε βημα του p και ο
    προυπολογισμος συσσωρευοταν (το «p=0.20» ηταν στην πραξη +50.8% εως +55.4%,
    και 3 απο τις 4 τελικες λυσεις παραβιαζαν το B του §3.2.4).
    """
    if warm_start_sol is None:
        # Only reseed for the first (smallest) budget level. A warm-started
        # continuation should carry on from wherever the RNG stream and the
        # Q-table already are, not jump back to the same fixed point.
        random.seed(seed)
        np.random.seed(seed)

    use_energy = (scenario_name == "S2_energy")
    compute_energy = use_energy

    if warm_start_sol is not None:
        initial_routes = warm_start_sol.Routes
    initial_routes = _routeset_sanitize(copy.deepcopy(initial_routes))
    sol = Solution(copy.deepcopy(initial_routes), actions)
    if warm_start_sol is not None:
        sol.Q = copy.deepcopy(warm_start_sol.Q)
        sol.epsilon = warm_start_sol.epsilon
    # Το B ειναι γνωστο πριν την αξιολογηση οταν υπαρχει reference, ωστε
    # να μπορει να χρησιμοποιηθει και στη διακριτοποιηση της καταστασης .
    if reference is not None:
        C0 = float(reference["cost"])
        # Ο σχεδιασμος δεσμευει τον προυπολογισμο μειον την εφεδρεια, ποτε ομως
        # λιγοτερα απο το κοστος του υπαρχοντος δικτυου: στα χαμηλα p η εφεδρεια
        # θα εκανε ακομη και την R0 μη εφικτη.
        B = max(C0, C0 * (1.0 + p) * (1.0 - BUDGET_RESERVE))
    else:
        C0 = None
        B = None

    att0, CEF0, dun0 = evaluate_solution(
        sol, TT_dict, TD_dict, tt, dist_km,
        c_km, c_h, e_km,
        operating_hours=operating_hours,
        headway_min=headway_min,
        assume_bidirectional=assume_bidirectional,
        use_energy_in_state=use_energy,
        compute_energy=compute_energy,
        state_reference=reference,
        state_budget=B,
    )

    E0 = sol.energy_kwh if sol.energy_kwh is not None else 0.0
    VKM0 = sol.vkm
    VH0 = sol.vh
    if reference is None:
        C0 = sol.cost
        # Ο σχεδιασμος δεσμευει τον προυπολογισμο μειον την εφεδρεια, ποτε ομως
        # λιγοτερα απο το κοστος του υπαρχοντος δικτυου: στα χαμηλα p η εφεδρεια
        # θα εκανε ακομη και την R0 μη εφικτη.
        B = max(C0, C0 * (1.0 + p) * (1.0 - BUDGET_RESERVE))

    # Κανονικοποιηση των μεταβολων ως προς τις ΑΡΧΙΚΕΣ τιμες αναφορας της R0,
    # οπως οριζει το §3.2.5. Πριν, ο παρονομαστης αλλαζε σε καθε επιπεδο p, αρα
    # οι ανταμοιβες δεν ηταν συγκρισιμες μεταξυ προυπολογισμων.
    _ref_CEF = float(reference["CEF"]) if reference is not None else CEF0
    _ref_ATT = float(reference["att"]) if reference is not None else att0
    _ref_E = float(reference["E"] or 0.0) if reference is not None else E0

    norm = {
        "CEF": max(abs(_ref_CEF), 1e-9),
        "ATT": max(abs(_ref_ATT), 1e-9),
        "E": max(abs(_ref_E), 1e-9),
        # Κανονικοποιηση του cef_new ως προς την ΑΡΧΙΚΗ λυση R0, ακριβως
        # οπως και οι υπολοιποι οροι.
        "CEF2": max(abs(float(reference.get("CEF2", 0.0)) if reference is not None
                        else (sol.CEF2 or 0.0)), 1e-9),
    }

    # Τα ενθετα διπλα εισαγωγικα μεσα σε f-string επιτρεπονται μονο απο
    # Python 3.12 (PEP 701). Σε 3.11 ο κωδικας δεν εκανε καν import.
    _energy_txt = "NA" if sol.energy_kwh is None else f"{E0:.2f}"
    if verbose:
        print("\n" + "=" * 100)
        print(f"RIVERA | {scenario_name} | BUDGET SCENARIO START | p={p:.2f}")
        print(f"Initial: att={att0:.2f}, CEF={CEF0:.4f}, dun={dun0:.3f}, cost={C0:.2f}, energy={_energy_txt}, VKM={VKM0:.2f}, VH={VH0:.2f}, eps={sol.epsilon:.3f}")
        print(f"Budget: C0={C0:.2f} -> B={B:.2f}  |  operating_hours={operating_hours:.1f}, headway={headway_min:.1f} min, bidirectional={assume_bidirectional}")
        print("=" * 100)

    # Best feasible-only solution under both hard constraints: DUN==0 and cost<=B
    best_sol = copy.deepcopy(sol)
    best_att, best_CEF, best_dun = sol.att, sol.CEF, sol.dun
    best_CEF2 = sol.CEF2 or 0.0
    best_exposure = sol.exposure or 0.0
    best_cost, best_energy = sol.cost, sol.energy_kwh
    best_ep = 0
    best_found = (best_dun == 0 and best_cost <= B + 1e-9)

    # Μετρητες για την επανεκκινηση απο την καλυτερη λυση.
    steps_since_improvement = 0
    n_restarts = 0
    Q_shared = sol.Q          # ο ΙΔΙΟΣ πινακας Q επιβιωνει καθε επανεκκινησης

    for ep in range(Episodes):
        state_key = sol.state_key
        action = choose_action(sol)

        new_sol, ok = ACTION_FUNCS[action](sol, TT_dict)
        tried_actions = {action}

        tries = 0
        while (not ok) and tries < DEFAULT_MAX_INSERT_TRIES:
            tries += 1
            remaining = [a for a in actions if a not in tried_actions]
            if not remaining:
                break
            action = random.choice(remaining)
            tried_actions.add(action)
            new_sol, ok = ACTION_FUNCS[action](sol, TT_dict)

        if not ok:
            decay_epsilon(sol)
            continue

        att_new, cef_new, dun_new = evaluate_solution(
            new_sol, TT_dict, TD_dict, tt, dist_km,
            c_km, c_h, e_km,
            operating_hours=operating_hours,
            headway_min=headway_min,
            assume_bidirectional=assume_bidirectional,
            use_energy_in_state=use_energy,
            compute_energy=compute_energy,
            state_reference=reference,
            state_budget=B,
        )

        # Hard feasibility: must satisfy both DUN and budget
        # Η μη εφικτη λυση ΕΞΑΚΟΛΟΥΘΕΙ να απορριπτεται (δεν γινεται ποτε
        # τρεχουσα λυση). Ομως ο πρακτορας μαθαινει πλεον οτι η συγκεκριμενη
        # ενεργεια σε αυτη την κατασταση οδηγει σε αδιεξοδο: μικρη αρνητικη
        # ανταμοιβη με self-loop (s' = s). Το 88% των βηματων απορριπτοταν και
        # δεν παρηγαγε καμια πληροφορια.
        if dun_new > 0 or new_sol.cost > B + 1e-9:
            q_update(sol, state_key, actions.index(action),
                     REJECT_PENALTY, state_key, alpha)
            decay_epsilon(sol)
            steps_since_improvement += 1
            if steps_since_improvement >= STAGNATION_LIMIT:
                _eps_now = sol.epsilon
                sol = copy.deepcopy(best_sol)
                sol.Q = Q_shared
                sol.epsilon = _eps_now
                steps_since_improvement = 0
                n_restarts += 1
            continue

        r = compute_reward(sol, new_sol, norm, scenario_name)

        action_idx = actions.index(action)
        next_state_key = new_sol.state_key
        q_update(sol, state_key, action_idx, r, next_state_key, alpha)

        new_sol.Q = sol.Q
        new_sol.state_key = next_state_key

        # Σκληρο προτυπο εξυπηρετησης: ο μεσος χρονος διαδρομης δεν επιτρεπεται
        # να ξεπερασει το R0 κατα περισσοτερο απο ATT_TOLERANCE.
        if att_ceiling is not None and att_new > att_ceiling + 1e-9:
            better = False
        else:
            # Λεξικογραφικα: CEF2 -> χρονος -> CEF. Καθαρη βελτιωση του CEF2
            # γινεται παντα δεκτη. Μεσα στη ζωνη ανοχης οι δυο λυσεις ειναι
            # ισοδυναμες ως προς τον πλεονασμο και αποφασιζει ο χρονος.
            # Μειωση του CEF2 δεν γινεται ποτε δεκτη.
            c2_new = new_sol.CEF2 or 0.0
            tol = CEF2_TOL * max(best_CEF2, 1e-9)
            d2 = c2_new - best_CEF2
            if not best_found:
                better = True
            elif d2 > tol:
                better = True
            elif d2 >= -1e-12:
                better = (att_new < best_att - 1e-9) or (
                    abs(att_new - best_att) <= 1e-9 and cef_new > best_CEF + 1e-9)
            else:
                better = False

        if better:
            best_sol = copy.deepcopy(new_sol)
            best_att, best_CEF, best_dun = att_new, cef_new, dun_new
            best_CEF2 = new_sol.CEF2 or 0.0
            best_exposure = new_sol.exposure or 0.0
            best_cost, best_energy = new_sol.cost, new_sol.energy_kwh
            best_ep = ep + 1
            best_found = True

        # Οταν η αναζητηση κολλησει, επιστροφη στην καλυτερη λυση αντι για
        # συνεχεια απο τη χειροτερη - αλλιως ο πρακτορας απομακρυνεται.
        if better:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        sol = new_sol
        decay_epsilon(sol)

        if steps_since_improvement >= STAGNATION_LIMIT:
            _eps_now = sol.epsilon
            sol = copy.deepcopy(best_sol)
            sol.Q = Q_shared          # ο πινακας Q ΔΕΝ μηδενιζεται
            sol.epsilon = _eps_now    # ουτε το επιπεδο εξερευνησης
            steps_since_improvement = 0
            n_restarts += 1

        if verbose:
            print(
                f"{scenario_name} | p={p:.2f} | Ep {ep+1:03d}: {action:>28} | "
                f"r={r:+.5f} | CEF={cef_new:.4f} | att={att_new:.2f} | dun={dun_new:.2f} | "
                f"cost={new_sol.cost:.2f}/{B:.2f} | energy={('NA' if new_sol.energy_kwh is None else f'{new_sol.energy_kwh:.2f}')} | "
                f"VKM={new_sol.vkm:.2f} | VH={new_sol.vh:.2f} | eps={sol.epsilon:.3f}"
            )

    if verbose:
        print("\n----- BEST FEASIBLE SOLUTION FOR THIS (scenario, p) -----")
        print(
            f"scenario={scenario_name} | p={p:.2f} | BestEp={best_ep:03d} | "
            f"CEF={best_CEF:.4f} | att={best_att:.2f} | dun={best_dun:.2f} | "
            f"cost={best_cost:.2f}/{B:.2f} | energy={('NA' if best_energy is None else f'{best_energy:.2f}')} | "
            f"restarts={n_restarts}"
        )
        for i, r in enumerate(best_sol.Routes, 1):
            print(f"  Route {i:02d}: {r}")

    return {
        "scenario": scenario_name,
        "p": p,
        "C0": C0,
        "B": B,
        "B_total": (C0 * (1.0 + p)) if C0 is not None else None,
        "budget_reserve": BUDGET_RESERVE,
        "best_ep": best_ep,
        "best_att": best_att,
        "best_CEF": best_CEF,
        "best_CEF2": best_CEF2,              # περιορισμενος πλεονασμος
        "best_exposure": best_exposure,      # % ζητησης με ΜΙΑ διαδρομη
        "best_dun": best_dun,
        "best_cost": best_cost,
        "best_energy": best_energy,
        "n_restarts": n_restarts,
        "best_sol": best_sol,
    }


def _build_transfer_stops(routeset: List[List[int]]) -> Dict[int, Dict[int, List[int]]]:
    trstops = {}
    for id1, r1 in enumerate(routeset):
        trstops[id1] = {}
        s1 = set(r1)
        for id2, r2 in enumerate(routeset):
            trstops[id1][id2] = list(s1.intersection(r2)) if id2 != id1 else []
    return trstops


def _find_unserved_od_pair(routeset, TT_dict, TD_dict, tt_matrix):
    """
    Return the first OD pair that is still unserved under the DUN rule
    (no feasible path, >2 transfers, or path cost >= 1e7).
    """
    routeset = _routeset_sanitize(copy.deepcopy(routeset))
    trstops = _build_transfer_stops(routeset)
    seenpairs = set()

    for ori in sorted(TD_dict):
        for dst in sorted(TD_dict[ori]):
            if ori == dst:
                continue
            pair = (ori, dst)
            if pair in seenpairs:
                continue
            sp = shortestPath(TT_dict, ori, dst)
            paths = seekpathfr(0, ori, dst, routeset, tt_matrix, trstops, sp, 0)
            bad = (not paths) or (paths[0][2] > 2) or (paths[0][0] >= 1e7)
            if bad:
                return ori, dst
            seenpairs.add(pair)
            seenpairs.add((dst, ori))
    return None


def _augment_routeset_until_dun0(routeset, TT_dict, TD_dict, tt, max_routes=80, verbose=True):
    """
    Add directed shortest-path routes for OD pairs that remain unserved until
    DUN becomes 0 or no further progress is possible.
    """
    routes = _routeset_sanitize(copy.deepcopy(routeset))
    routes = make_routeset_feasible(routes, TT_dict)

    prev_signature = None
    while True:
        try:
            _, _, _, dun, _, _ = evaluateattacks(routes, TT_dict, TD_dict, tt)
        except Exception:
            dun = None

        if dun is not None and dun <= 0.0:
            return routes, 0.0

        if len(routes) >= max_routes:
            raise RuntimeError(
                f"ensure_initial_dun0 could not reach DUN=0 before hitting max_routes={max_routes}. "
                f"Current DUN={dun!r} and route_count={len(routes)}."
            )

        od = _find_unserved_od_pair(routes, TT_dict, TD_dict, tt)
        if od is None:
            if dun is None:
                raise RuntimeError("ensure_initial_dun0 could not verify DUN and found no augmenting OD pair.")
            return routes, dun

        ori, dst = od
        _, sp_nodes = shortestPath(TT_dict, ori, dst)
        if not sp_nodes:
            raise RuntimeError(f"ensure_initial_dun0 found unreachable OD pair ({ori}, {dst}) in TT graph.")

        new_route = make_route_feasible(sp_nodes, TT_dict)
        routes2 = _routeset_sanitize(routes + [new_route])

        signature = canonical_routeset(routes2)
        if signature == prev_signature or signature == canonical_routeset(routes):
            raise RuntimeError(
                f"ensure_initial_dun0 could not make progress while repairing OD pair ({ori}, {dst})."
            )
        prev_signature = signature
        routes = routes2

        if verbose:
            print(f"Augmented initial routeset with shortest-path route for OD ({ori}, {dst}); routes={len(routes)}")


# ====================== 7b. Initial routeset cache =======================


INITIAL_FEASIBLE_ROUTESET_CACHE = {}

def _initial_routes_cache_digest(initial_routes: List[List[int]]) -> str:
    payload = json.dumps(initial_routes, sort_keys=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _initial_routes_cache_path(initial_routes: List[List[int]]) -> str:
    payload = json.dumps(initial_routes, sort_keys=False)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"rivera_initial_feasible_cache_{digest}.json"


def ensure_initial_dun0(initial_routes, TT_dict, TD_dict, tt, max_routes=80, verbose=True):
    """
    Guarantee that the initial routeset is both edge-feasible and truly DUN=0.
    A repaired seed is reused from the in-memory cache only if a fresh
    evaluation still confirms DUN=0.
    """
    cache_key = _initial_routes_cache_digest(initial_routes)
    cached = INITIAL_FEASIBLE_ROUTESET_CACHE.get(cache_key)
    if cached is not None:
        cached_routes = _routeset_sanitize(copy.deepcopy(cached))
        try:
            _, _, _, dun_cached, _, _ = evaluateattacks(cached_routes, TT_dict, TD_dict, tt)
        except Exception:
            dun_cached = None
        if dun_cached is not None and dun_cached <= 0.0:
            if verbose:
                print("Loaded initial feasible routeset from in-memory cache (verified DUN=0)")
            return cached_routes
        if verbose:
            print("Ignoring stale in-memory initial routeset cache because DUN=0 was not verified")

    routes = _routeset_sanitize(copy.deepcopy(initial_routes))
    routes = make_routeset_feasible(routes, TT_dict)
    routes, dun = _augment_routeset_until_dun0(
        routes, TT_dict, TD_dict, tt, max_routes=max_routes, verbose=verbose
    )

    if dun is None or dun > 0.0:
        raise RuntimeError(f"ensure_initial_dun0 failed: final DUN={dun!r}")

    INITIAL_FEASIBLE_ROUTESET_CACHE[cache_key] = copy.deepcopy(routes)

    if verbose:
        print(f"Initial routeset after one-time repair/augmentation: DUN={dun:.3f}")

    return routes


def _safe_tag(s: str) -> str:
    return str(s).replace(" ", "_").replace(".", "_").replace("/", "_")


def save_best_scenarios_json(output_dir: str, param_set_name: str, initial_sol, best_by_scenario: Dict[str, dict]):
    """
    Save the overall best solution for S1 and S2 in the SAME JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "param_set": param_set_name,
        "initial_solution": {
            "att": initial_sol.att,
            "CEF": initial_sol.CEF,
            "CEF2": getattr(initial_sol, "CEF2", None),
            "exposure": getattr(initial_sol, "exposure", None),
            "dun": initial_sol.dun,
            "cost": initial_sol.cost,
            "energy": initial_sol.energy_kwh,
            "vkm": initial_sol.vkm,
            "vh": initial_sol.vh,
            "routes": initial_sol.Routes,
        },
        "best_scenarios": {}
    }

    for scenario_name, overall_best in best_by_scenario.items():
        payload["best_scenarios"][scenario_name] = {
            "p": overall_best["p"],
            "C0": overall_best["C0"],
            "B": overall_best["B"],
            "best_ep": overall_best["best_ep"],
            "best_att": overall_best["best_att"],
            "best_CEF": overall_best["best_CEF"],
            # Ο περιορισμενος πλεονασμος και η εκτεθειμενη ζητηση - τα
            # δυο μεγεθη πανω στα οποια κρινεται η ανθεκτικοτητα.
            "best_CEF2": overall_best.get("best_CEF2"),
            "best_exposure": overall_best.get("best_exposure"),
            "best_dun": overall_best["best_dun"],
            "best_cost": overall_best["best_cost"],
            "best_energy": overall_best["best_energy"],
            "best_routes": overall_best["best_sol"].Routes,
        }

    out_path = Path(output_dir) / f"best_scenarios_{_safe_tag(param_set_name)}_{MODEL_TAG}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved best S1/S2 solutions to {out_path}")


def plot_metrics_vs_p(output_dir: str, param_set_name: str, results_by_scenario: Dict[str, List[dict]]):
    """Ενα διαγραμμα ανα δεικτη, με τα δυο σεναρια μαζι.

    Η ταυτοτητα του σεναριου δινεται και με σχημα σημειου και με τυπο γραμμης,
    οχι μονο με χρωμα, ωστε τα σχηματα να διαβαζονται και τυπωμενα ασπρομαυρα.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scenario_order = ["S1_service", "S2_energy"]
    STYLE = {
        "S1_service": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
        "S2_energy":  {"color": "#ea580c", "marker": "s", "linestyle": "--"},
    }
    GRID = "#d4d4d4"
    REF = "#737373"
    P_TICKS = sorted({float(r["p"]) for rows in results_by_scenario.values() for r in rows})

    metrics = [
        ("best_att", "ATT (min)", "att_vs_p"),
        ("best_exposure", "Exposure (% of demand)", "exposure_vs_p"),
        ("best_energy", "Energy (kWh)", "energy_vs_p"),
    ]

    for metric_key, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted_any = False
        for scenario_name in scenario_order:
            rows = sorted(results_by_scenario.get(scenario_name, []), key=lambda r: r["p"])
            if not rows:
                continue
            xs, ys = [], []
            for r in rows:
                val = r.get(metric_key)
                if val is None:
                    xs, ys = [], []
                    break
                xs.append(float(r["p"]))
                ys.append(float(val))
            if not ys:
                continue
            st = STYLE[scenario_name]
            ax.plot(xs, ys, linewidth=2, markersize=6, label=scenario_name, **st)
            plotted_any = True

        ax.set_xlabel("Budget expansion p", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(P_TICKS)
        ax.margins(y=0.10)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if plotted_any:
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out_path = Path(output_dir) / f"{stem}_{_safe_tag(param_set_name)}_{MODEL_TAG}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to {out_path}")

    # Πλεονασμος: ο CEF και ο περιορισμενος CEF2 στους ιδιους αξονες. Ιδιο
    # χρωμα ανα σεναριο, διαφορετικο παχος και διαφανεια ανα δεικτη - η αποσταση
    # των δυο καμπυλων ειναι το μερος του πλεονασμου που ΔΕΝ προσθετει προστασια.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scenario_name in scenario_order:
        rows = sorted(results_by_scenario.get(scenario_name, []), key=lambda r: r["p"])
        if not rows:
            continue
        xs = [float(r["p"]) for r in rows]
        st = STYLE[scenario_name]
        ax.plot(xs, [float(r["best_CEF"]) for r in rows], linewidth=2, markersize=6,
                label=f"{scenario_name} - CEF", **st)
        ax.plot(xs, [float(r["best_CEF2"]) for r in rows], linewidth=1.4, markersize=5,
                alpha=0.55, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                label=f"{scenario_name} - CEF2")
    ax.axhline(CEF_CAP, linewidth=1.0, linestyle=":", color=REF)
    ax.annotate("CEF2 cap", xy=(P_TICKS[0], CEF_CAP), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=REF)
    ax.set_xlabel("Budget expansion p", fontsize=10)
    ax.set_ylabel("Path redundancy", fontsize=10)
    ax.set_xticks(P_TICKS)
    ax.margins(y=0.10)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    out_path = Path(output_dir) / f"cef_vs_p_{_safe_tag(param_set_name)}_{MODEL_TAG}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

    # Κοστος: το μονο διαγραμμα με γραμμες αναφορας. Η αποσταση της καμπυλης
    # απο το B_total ειναι η εφεδρεια που μενει διαθεσιμη για την αποκατασταση.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ref_rows = None
    for scenario_name in scenario_order:
        rows = sorted(results_by_scenario.get(scenario_name, []), key=lambda r: r["p"])
        if not rows:
            continue
        ref_rows = ref_rows or rows
        st = STYLE[scenario_name]
        ax.plot([float(r["p"]) for r in rows], [float(r["best_cost"]) for r in rows],
                linewidth=2, markersize=6, label=scenario_name, **st)
    if ref_rows:
        xs = [float(r["p"]) for r in ref_rows]
        ax.plot(xs, [float(r["B_total"]) for r in ref_rows], linewidth=1.2,
                linestyle=":", color=REF, label="B (approved budget)")
        ax.plot(xs, [float(r["B"]) for r in ref_rows], linewidth=1.2,
                linestyle="-.", color=REF, label="B design (after reserve)")
    ax.set_xlabel("Budget expansion p", fontsize=10)
    ax.set_ylabel("Operating cost (EUR)", fontsize=10)
    ax.set_xticks(P_TICKS)
    ax.margins(y=0.10)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out_path = Path(output_dir) / f"cost_vs_p_{_safe_tag(param_set_name)}_{MODEL_TAG}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

def _build_routes_geodataframes(coords: Dict[int, Tuple[float, float]],
                               TT_dict: Dict[int, Dict[int, float]],
                               routeset: List[List[int]]):
    if gpd is None or Point is None or LineString is None:
        raise ImportError(
            "Geospatial plotting requires geopandas, shapely and contextily. "
            "Install them with: pip install geopandas contextily shapely pyproj"
        )

    node_rows = []
    for node, (lat, lon) in sorted(coords.items()):
        node_rows.append({"node": int(node), "lat": float(lat), "lon": float(lon), "geometry": Point(float(lon), float(lat))})

    edge_rows = []
    seen_edges = set()
    for i, nbrs in TT_dict.items():
        if i not in coords:
            continue
        for j in nbrs:
            if j not in coords:
                continue
            edge_key = tuple(sorted((int(i), int(j))))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            edge_rows.append({
                "u": int(i),
                "v": int(j),
                "geometry": LineString([(float(lon1), float(lat1)), (float(lon2), float(lat2))])
            })

    route_rows = []
    for ridx, route in enumerate(routeset, 1):
        valid_nodes = [int(n) for n in route if int(n) in coords]
        if len(valid_nodes) < 2:
            continue
        pts = [(float(coords[n][1]), float(coords[n][0])) for n in valid_nodes]
        route_rows.append({
            "route_id": int(ridx),
            "geometry": LineString(pts)
        })

    nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:4326")
    routes_gdf = gpd.GeoDataFrame(route_rows, geometry="geometry", crs="EPSG:4326")
    return nodes_gdf, edges_gdf, routes_gdf


def plot_routeset_network_comparison(output_dir: str, param_set_name: str, coords: Dict[int, Tuple[float, float]],
                                     TT_dict: Dict[int, Dict[int, float]], initial_routes: List[List[int]],
                                     best_s1_routes: List[List[int]], best_s2_routes: List[List[int]]):
    """Plot three Rivera routesets on top of a real web basemap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not coords:
        print("Skipping network comparison plot: empty coordinates.")
        return

    try:
        nodes_gdf, edges_gdf, initial_gdf = _build_routes_geodataframes(coords, TT_dict, initial_routes)
        _, _, s1_gdf = _build_routes_geodataframes(coords, TT_dict, best_s1_routes)
        _, _, s2_gdf = _build_routes_geodataframes(coords, TT_dict, best_s2_routes)

        nodes_plot = nodes_gdf.to_crs(epsg=3857)
        edges_plot = edges_gdf.to_crs(epsg=3857)
        initial_plot = initial_gdf.to_crs(epsg=3857)
        s1_plot = s1_gdf.to_crs(epsg=3857)
        s2_plot = s2_gdf.to_crs(epsg=3857)

        panels = [
            ("Initial routeset", initial_plot),
            ("Best S1_service routeset", s1_plot),
            ("Best S2_energy routeset", s2_plot),
        ]

        xmin, ymin, xmax, ymax = nodes_plot.total_bounds
        dx = xmax - xmin
        dy = ymax - ymin
        pad_x = max(dx * 0.08, 500.0)
        pad_y = max(dy * 0.08, 500.0)
        xlim = (xmin - pad_x, xmax + pad_x)
        ylim = (ymin - pad_y, ymax + pad_y)

        fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharex=True, sharey=True)
        basemap_ok = True

        for ax, (title, routes_plot) in zip(axes, panels):
            if not edges_plot.empty:
                edges_plot.plot(ax=ax, linewidth=0.7, alpha=0.20, color="dimgray", zorder=1)

            if not routes_plot.empty:
                cmap = plt.get_cmap("tab20")
                for idx, row in enumerate(routes_plot.itertuples()):
                    gpd.GeoSeries([row.geometry], crs=routes_plot.crs).plot(
                        ax=ax,
                        linewidth=2.2,
                        alpha=0.90,
                        color=cmap(idx % 20),
                        zorder=3,
                    )

            if not nodes_plot.empty:
                nodes_plot.plot(ax=ax, markersize=14, alpha=0.85, color="black", zorder=4)
                for row in nodes_plot.itertuples():
                    ax.text(row.geometry.x, row.geometry.y, str(row.node), fontsize=5.5,
                            ha="center", va="bottom", alpha=0.80, zorder=5)

            if ctx is not None:
                try:
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=5, zoom='auto')
                except Exception as exc:
                    basemap_ok = False
                    print(f"Basemap tiles could not be loaded ({exc}). Saving plain network overlay instead.")

            ax.set_title(title)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_axis_off()

        mode = "with basemap" if basemap_ok and ctx is not None else "without basemap"
        fig.suptitle(f"Rivera network comparison ({mode}) | param_set={param_set_name}")
        fig.tight_layout()
        out_path = Path(output_dir) / f"network_comparison_{_safe_tag(param_set_name)}_{MODEL_TAG}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved network comparison to {out_path}")

    except ImportError as exc:
        print(str(exc))
        print("Falling back to a plain latitude/longitude comparison plot.")

        lats = [coords[n][0] for n in sorted(coords)]
        lons = [coords[n][1] for n in sorted(coords)]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        pad_lat = max((max_lat - min_lat) * 0.08, 0.002)
        pad_lon = max((max_lon - min_lon) * 0.08, 0.002)

        def draw_plain_panel(ax, title, routeset):
            for i, nbrs in TT_dict.items():
                if i not in coords:
                    continue
                x1, y1 = coords[i][1], coords[i][0]
                for j in nbrs:
                    if j not in coords or j < i:
                        continue
                    x2, y2 = coords[j][1], coords[j][0]
                    ax.plot([x1, x2], [y1, y2], linewidth=0.35, alpha=0.15, color="dimgray", zorder=1)

            cmap = plt.get_cmap("tab20")

            for idx, route in enumerate(routeset):
                route = [int(n) for n in route if int(n) in coords]
                if len(route) < 2:
                    continue
                xs = [coords[n][1] for n in route]
                ys = [coords[n][0] for n in route]
                ax.plot(xs, ys, linewidth=2.0, alpha=0.90, color=cmap(idx % 20), zorder=3)

            ax.scatter(lons, lats, s=10, alpha=0.85, color="black", zorder=4)
            for n in sorted(coords):
                ax.text(coords[n][1], coords[n][0], str(n), fontsize=5.5, ha="center", va="bottom", alpha=0.8)
            ax.set_title(title)
            ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
            ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)
            ax.set_axis_off()

        fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharex=True, sharey=True)
        draw_plain_panel(axes[0], "Initial routeset", initial_routes)
        draw_plain_panel(axes[1], "Best S1_service routeset", best_s1_routes)
        draw_plain_panel(axes[2], "Best S2_energy routeset", best_s2_routes)
        fig.suptitle(f"Rivera network comparison | param_set={param_set_name}")
        fig.tight_layout()
        out_path = Path(output_dir) / f"network_comparison_{_safe_tag(param_set_name)}_{MODEL_TAG}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved network comparison to {out_path}")


# ====================== 8. Main =============================================


def main():
    print("MAIN STARTED")
    print("CURRENT WORKING DIRECTORY:", os.getcwd())

    # --- Core methodological parameters discussed in the thesis meetings ---
    operating_hours = 12.0
    headway_min = 10.0
    assume_bidirectional = True

    # Baseline parameter set (recommended main case)
    c_km = 0.8    # EUR / vehicle-km
    c_h = 18.0    # EUR / vehicle-hour
    e_km = 1.2    # kWh / vehicle-km

    # Optional sensitivity sets (documented, not arbitrary)
    PARAM_SETS = [
        {"name": "baseline", "c_km": 0.8, "c_h": 18.0, "e_km": 1.2},
        {"name": "low",      "c_km": 0.6, "c_h": 15.0, "e_km": 1.0},
        {"name": "high",     "c_km": 1.0, "c_h": 22.0, "e_km": 1.5},
    ]

    USE_ALL_PARAM_SETS = False
    param_sets = PARAM_SETS if USE_ALL_PARAM_SETS else [PARAM_SETS[0]]

    p_list = [0.05, 0.10, 0.15, 0.20]
    # ΤΟ ΝΕΟ ΣΕΝΑΡΙΟ. Το S1/S2 μενουν ΑΘΙΚΤΑ - το S3 προστιθεται ως ΤΡΙΤΗ
    # παραλλαγη, ωστε να συγκριθουν τα δυο δικτυα στο πειραμα διαταραχης. Αν το
    # S3 χασει, κρατας τα σημερινα αποτελεσματα ακεραια. Δεν χανεις τιποτα.
    scenario_list = ["S1_service", "S2_energy"]

    # 300 -> 800 βηματα. Εχει νοημα ΜΟΝΟ μαζι με το restart : χωρις
    # αυτο, το 80% των βηματων ηταν ηδη αχρηστο. Μετρημενο με restart:
    # 300 βηματα -> CEF 1.6812 (best_ep 173) | 800 βηματα -> CEF 1.7314 (best_ep 564).
    episodes = 800
    alpha = 0.1
    seed = 42
    verbose = True

    base_dir = Path(__file__).resolve().parent

    print("BASEMAP MODE: OpenStreetMap tiles via contextily (internet required for map tiles)")

    TT_dict, TD_dict, tt, td = load_rivera(
        travel_file="RiveraTravel.txt",
        demand_file="RiveraDemand.txt",
        n_nodes=84
    )
    coords = read_coords_latlon("Rivera_coords.txt")
    dist_km = build_directed_dist_km(TT_dict, coords)

    initial_routes = [
        [1, 2, 3, 7, 9, 14, 18, 22, 26, 34, 67],
        [4, 5, 6, 5, 8, 10, 23, 24, 29, 30, 32, 33, 34, 67],
        [11, 12, 13, 15, 16, 14, 18, 22, 26, 34, 67],
        [17, 21, 20, 22, 26, 34, 67, 68, 69, 65, 61, 58, 57, 55],
        [19, 18, 22, 26, 34, 67, 71, 74, 75, 76, 77, 78, 79, 84],
        [25, 27, 28, 31, 33, 34, 67, 68, 69, 70, 73, 72, 83, 84],
        [35, 36, 37, 38, 30, 32, 33, 34, 67, 68, 69, 65, 80, 81],
        [39, 40, 41, 42, 43, 44, 43, 42, 41, 40, 39, 59, 57, 55],
        [49, 48, 54, 55, 57, 58, 60, 64, 63, 66, 68, 67],
        [50, 51, 52, 53, 54, 55, 56, 55, 57, 58, 61, 65, 69, 68],
        [59, 63, 66, 68, 67, 34, 33, 32, 62, 39],
        [34, 67, 68, 69, 66, 63, 59],
        [33, 32, 62, 39, 59, 57, 55, 54, 48, 47, 46, 45],
        [67, 68, 69, 70, 82],
        [71, 67, 68, 69, 65, 61, 58, 57, 55, 54, 53, 52],
    ]

    initial_routes = ensure_initial_dun0(
        initial_routes, TT_dict, TD_dict, tt, max_routes=80, verbose=True
    )

    actions = [
        "add_redundant_segment",
        "reroute_via_alternative_node",
        "extend_route_for_coverage",
        "prune_low_value_node",
    ]

    all_results = []

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_dir / f"{MODEL_TAG}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("OUTPUT FOLDER:", output_dir)

    print("\n" + "#" * 110)
    print("RIVERA RL v9 | initial=v5.2 | actions=v5.2 | physical operating-cost budget | energy in scenario 2 only")
    print("#" * 110)

    for params in param_sets:
        c_km = params["c_km"]
        c_h = params["c_h"]
        e_km = params["e_km"]

        print("\n" + "*" * 110)
        print(
            f"PARAMETER SET = {params['name']} | "
            f"c_km={c_km:.2f} EUR/VKM | c_h={c_h:.2f} EUR/VH | e_km={e_km:.2f} kWh/VKM"
        )
        print("*" * 110)

        init_sol = Solution(copy.deepcopy(initial_routes), actions)
        evaluate_solution(
            init_sol, TT_dict, TD_dict, tt, dist_km,
            c_km, c_h, e_km,
            operating_hours=operating_hours,
            headway_min=headway_min,
            assume_bidirectional=assume_bidirectional,
            use_energy_in_state=True,
            compute_energy=True,
        )
        print(
            f"Initial metrics -> ATT={init_sol.att:.2f}, CEF={init_sol.CEF:.4f}, DUN={init_sol.dun:.3f}, "
            f"C0={init_sol.cost:.2f}, E0={init_sol.energy_kwh:.2f}, VKM0={init_sol.vkm:.2f}, VH0={init_sol.vh:.2f}"
        )

        # Οι τιμες αναφορας της ΑΡΧΙΚΗΣ λυσης R0. Υπολογιζονται ΜΙΑ φορα
        # και χρησιμοποιουνται σε ΟΛΑ τα επιπεδα p, ωστε B = C0*(1+p) να σημαινει
        # πραγματικα +p% επι της R0 (§3.2.4) και οι ανταμοιβες να κανονικοποιουνται
        # ως προς τις αρχικες τιμες (§3.2.5).
        REFERENCE = {
            "cost": float(init_sol.cost),
            "att": float(init_sol.att),
            "CEF": float(init_sol.CEF),
            # Η αναφορα για τον περιορισμενο δεικτη.
            "CEF2": float(init_sol.CEF2) if init_sol.CEF2 is not None else 0.0,
            "exposure": float(init_sol.exposure) if init_sol.exposure is not None else 0.0,
            "E": float(init_sol.energy_kwh) if init_sol.energy_kwh is not None else 0.0,
        }

        results_by_scenario = {}
        best_by_scenario = {}

        for scenario_name in scenario_list:
            results = []
            prev_best_sol = None   # η καλυτερη λυση του προηγουμενου p
            # Σταθερη οροφη χρονου για ολα τα επιπεδα προυπολογισμου.
            att_cap = float(REFERENCE["att"]) * (1.0 + ATT_TOLERANCE)

            for p in p_list:
                out = train_for_budget_scenario(
                    p=p,
                    scenario_name=scenario_name,
                    TT_dict=TT_dict,
                    TD_dict=TD_dict,
                    tt=tt,
                    dist_km=dist_km,
                    initial_routes=initial_routes,
                    actions=actions,
                    c_km=c_km,
                    c_h=c_h,
                    e_km=e_km,
                    operating_hours=operating_hours,
                    headway_min=headway_min,
                    assume_bidirectional=assume_bidirectional,
                    Episodes=episodes,
                    alpha=alpha,
                    seed=seed,
                    verbose=verbose,
                    warm_start_sol=prev_best_sol,
                    reference=REFERENCE,
                    att_ceiling=att_cap,
                )
                out["param_set"] = params["name"]
                results.append(out)
                all_results.append(out)
                prev_best_sol = out["best_sol"]

            # Τα συνολα εφικτων λυσεων ειναι εμφωλευμενα ως προς p, αρα η
            # βελτιστη τιμη του CEF ειναι μη-φθινουσα εξ ορισμου. Καθε
            # μη-μονοτονια ειναι θορυβος αναζητησης.
            _env = None
            for _r in sorted(results, key=lambda x: x["p"]):
                if _env is None or _r["best_CEF"] > _env["best_CEF"] + 1e-12:
                    _env = _r
                _r["envelope_p"] = _env["p"]
                _r["envelope_CEF"] = _env["best_CEF"]
                _r["envelope_att"] = _env["best_att"]
                _r["envelope_cost"] = _env["best_cost"]
                _r["envelope_energy"] = _env["best_energy"]

            results_by_scenario[scenario_name] = results

            print("\n" + "=" * 110)
            print(f"COMPARISON | param_set={params['name']} | scenario={scenario_name}")
            print("=" * 110)
            for r in results:
                energy_txt = "NA" if r["best_energy"] is None else f"{r['best_energy']:.2f}"
                print(
                    f"p={r['p']:.2f} | B={r['B']:.2f} | BestEp={r['best_ep']:03d} | "
                    f"CEF={r['best_CEF']:.4f} | att={r['best_att']:.2f} | dun={r['best_dun']:.2f} | "
                    f"cost={r['best_cost']:.2f} | energy={energy_txt} | "
                    f"envCEF={r['envelope_CEF']:.4f} (p={r['envelope_p']:.2f}) | "
                    f"restarts={r.get('n_restarts', 0)}"
                )

            feasible = [r for r in results if r["best_dun"] == 0 and r["best_cost"] <= r["B"] + 1e-9]
            if not feasible:
                feasible = results

            if scenario_name == "S1_service":
                overall_best = sorted(feasible, key=lambda r: (-r["best_CEF"], r["best_att"]))[0]
            else:
                overall_best = sorted(feasible, key=lambda r: (-r["best_CEF"], r["best_att"], r["best_energy"]))[0]

            best_by_scenario[scenario_name] = overall_best

            print("\nOVERALL BEST FOR THIS PARAMETER SET AND SCENARIO")
            overall_energy_txt = "NA" if overall_best["best_energy"] is None else f"{overall_best['best_energy']:.2f}"
            print(
                f"param_set={params['name']} | scenario={scenario_name} | BEST p={overall_best['p']:.2f} | "
                f"CEF={overall_best['best_CEF']:.4f} | att={overall_best['best_att']:.2f} | "
                f"dun={overall_best['best_dun']:.2f} | cost={overall_best['best_cost']:.2f}/{overall_best['B']:.2f} | "
                f"energy={overall_energy_txt} | BestEp={overall_best['best_ep']:03d}"
            )
            print("Best routeset (1-based node ids):")
            for k, r in enumerate(overall_best["best_sol"].Routes, 1):
                print(f"  Route {k:02d}: {r}")

        save_best_scenarios_json(
            output_dir=output_dir,
            param_set_name=params["name"],
            initial_sol=init_sol,
            best_by_scenario=best_by_scenario
        )

        plot_metrics_vs_p(
            output_dir=output_dir,
            param_set_name=params["name"],
            results_by_scenario=results_by_scenario
        )

        if "S1_service" in best_by_scenario and "S2_energy" in best_by_scenario:
            plot_routeset_network_comparison(
                output_dir=output_dir,
                param_set_name=params["name"],
                coords=coords,
                TT_dict=TT_dict,
                initial_routes=initial_routes,
                best_s1_routes=best_by_scenario["S1_service"]["best_sol"].Routes,
                best_s2_routes=best_by_scenario["S2_energy"]["best_sol"].Routes,
            )

    print("\nAll output files were saved in:", output_dir)
    print("FILES IN OUTPUT FOLDER:")
    for name in sorted(os.listdir(output_dir)):
        print(" -", name)


if __name__ == "__main__":
    main()
