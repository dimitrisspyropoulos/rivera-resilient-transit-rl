# -*- coding: utf-8 -*-
"""Monte Carlo αναλυση ανθεκτικοτητας - Rivera, Uruguay.

Καθε εκτελεση εφαρμοζει την ΙΔΙΑ τυχαια διαταραχη σε ολα τα δικτυα (common
random numbers): στο αρχικο R0 και στις δυο σχεδιασμενες λυσεις S1 και S2. Η
επιλογη κομβων και ακμων γινεται παντα ΑΠΟ ΤΟ ΟΔΙΚΟ ΔΙΚΤΥΟ και ποτε απο τις
γραμμες ενος συνολου, ωστε η συγκριση να ειναι πραγματικα ζευγαρωτη.

Τυποι διαταραχης: αστοχια συνδεσεων, αφαιρεση κομβων, κλεισιμο περιοχης,
αυξηση ζητησης, και συνδυασμος δυο εξ αυτων.

Δυο χρονικα σημεια καταγραφονται ανα εκτελεση:
  metrics               t0, αμεση εικονα - η γραμμη σπαει εκει που βρισκει τη
                        ζημια, καμια παρεμβαση. Ειναι το κυριο μετρο: δειχνει
                        τι αντεχει ο σχεδιασμος απο μονος του
  metrics_after_detour  t1, τι θα εδινε αυτοματη παρακαμψη. Σημειο αναφορας

Επισης καταγραφεται το coverage_floor_percent: η ζητηση που κανενα συνολο
γραμμων δεν μπορει να εξυπηρετησει μετα τη ζημια. Η αποκατασταση κρινεται ως
προς αυτο το δαπεδο, οχι ως προς το μηδεν.

Οταν κλεινει κομβος, η ζητηση του μεταφερεται στην πλησιεστερη ανοιχτη σταση
εντος 400 m με ταχυτητα βαδισης 4.8 km/h· ο χρονος βαδισης χρεωνεται.

Εισοδος:  best_scenarios_baseline.json απο το v7.py, RiveraTravel.txt,
          RiveraDemand.txt, Rivera_coords.txt, seekallpaths.py
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import heapq
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# =============================================================================
# ============================ MANUAL SETTINGS =================================
# =============================================================================

# Folder where your input files are located.
# Use "." if everything is in the same folder as this script.
DATA_DIR = "."

# Input files. You can put relative or absolute paths.
BEST_JSON_FILE = r"C:\Users\user\Τα αρχεία μου\ΔΙΠΛΩΜΑΤΙΚΗ\2026-08-13_00-23-54\best_scenarios_baseline.json"
TRAINING_MODULE_FILE = "v7.py"
SEEKALLPATHS_FOLDER = "."  # folder containing seekallpaths.py

TRAVEL_FILE = "RiveraTravel.txt"
DEMAND_FILE = "RiveraDemand.txt"
COORDS_FILE = "Rivera_coords.txt"

# Output folder.
OUT_DIR = "monte_carlo_disruption_results_v7"

# Monte Carlo settings.
MONTE_CARLO_RUNS = 150
RANDOM_SEED = 42

# Ποινη μη εξυπηρετουμενης ζητησης, σε λεπτα, για το ΓΕΝΙΚΕΥΜΕΝΟ ΚΟΣΤΟΣ:
#          GC = ATT_served * (1 - u) + P_UNSERVED * u ,   u = DUN/100
#      Ο δεικτης `att` υπολογιζεται μονο πανω στη ΖΗΤΗΣΗ ΠΟΥ ΕΞΥΠΗΡΕΤΕΙΤΑΙ.
#      Οταν μια διαταραχη αποκοπτει τα δυσκολοτερα ζευγη OD, αυτα φευγουν και
#      απο τον αριθμητη και απο τον παρονομαστη, και ο ATT ΠΕΦΤΕΙ: το δικτυο
#      φαινεται ταχυτερο αφου καταστραφει. Μετρημενο: ΔATT = -0.83 λεπτα σε
#      60 διαταραχες, και Pearson(DUN, ΔATT) = -0.290 στα 300 runs.
#      Ο επιβατης που δεν εξυπηρετειται δεν εξαφανιζεται - παιρνει ταξι,
#      περπαταει, η δεν ταξιδευει. Το P_UNSERVED ειναι το κοστος αυτης της
#      εναλλακτικης. Ο `att` ΔΕΝ αλλαζει και ΔΕΝ αφαιρειται· το `gc` προστιθεται.
P_UNSERVED = 60.0

# Τιμες για την αναλυση ευαισθησιας του P_UNSERVED (γραφονται ως gc_P30 κ.λπ.).
P_UNSERVED_SENSITIVITY = [30.0, 60.0, 90.0]

# Common Random Numbers: seed ανα run, ωστε το run #k να δεχεται ΤΗΝ ΙΔΙΑ
#      διαταραχη σε ΟΛΑ τα σεναρια. Ο παλιος κωδικας καλουσε random.seed() μια
#      φορα και το S1 κατανάλωνε τις πρωτες 150 κληρωσεις, το S2 τις επομενες:
#      μονο 15/150 (10%) διαταραχες ηταν κοινες. Ετσι η προταση «το S2 ειναι πιο
#      ανθεκτικο» συγχεοταν με το «το S2 ετυχε ηπιοτερες διαταραχες». Με CRN η
#      συγκριση γινεται ΖΕΥΓΑΡΩΤΗ (Wilcoxon signed-rank) και 150 ζευγαρωτα runs
#      εχουν την ισχυ ~600 μη ζευγαρωτων.
USE_COMMON_RANDOM_NUMBERS = True

# Το §3.4 λεει «ομαδα ΓΕΙΤΟΝΙΚΩΝ κομβων», ο κωδικας ομως κανει random.sample()
#      - τυχαιοι, ΜΗ συνεκτικοι κομβοι διασπαρτοι στην πολη. Δεν ειναι το ιδιο:
#
#        σοβαροτητα  κομβοι   ΔGC τυχαιοι   ΔGC γειτονικη συσταδα
#        mild          2         +2.61            +4.16
#        moderate      4         +6.28            +6.74
#        severe        7        +12.03            +6.80
#
#      Το κλεισιμο 7 ΓΕΙΤΟΝΙΚΩΝ σταςεων κοστιζει σχεδον τη ΜΙΣΗ ζημια απο 7
#      ΔΙΑΣΠΑΡΤΕΣ βλαβες: η συγκεντρωμενη διακοπη αφηνει το υπολοιπο δικτυο
#      ανεπαφο, ενω οι διασπαρτες κοβουν 7 διαφορετικους διαδρομους.
#
#      ΠΡΟΕΠΙΛΟΓΗ = False: η συμπεριφορα μενει ΑΚΡΙΒΩΣ η σημερινη (τυχαιοι
#      κομβοι) και τα αποτελεσματα του Κεφαλαιου 5 δεν αλλαζουν· διορθωνεται
#      αντ' αυτου η μια προταση του §3.4 σε «τυχαια επιλεγμενοι κομβοι εντος
#      της ζωνης». Με True αναπαραγεται η συσταδα, ωστε η παραπανω συγκριση
#      να ειναι ελεγξιμη απο κριτη.
AREA_CLOSURE_CONTIGUOUS = False

# Central area definition.
CENTRAL_NODES = list(range(18, 35))  # 18 έως 34

# Baseline operating assumptions.
C_KM = 0.8
C_H = 18.0
E_KM = 1.2
HEADWAY_MIN = 10.0
OPERATING_HOURS = 12.0
ASSUME_BIDIRECTIONAL = True

# Disruption type probabilities.
# Keep values summing to 1.0.
# ΝΕΟ ΜΕΙΓΜΑ ΔΙΑΤΑΡΑΧΩΝ
# --------------------------------------------------------------------------
# Οι διαταραχες που ΚΑΤΑΣΤΡΕΦΟΥΝ ΚΟΜΒΟΥΣ (node_removal + area_closure) πεφτουν
# απο 42 % σε 18 %. ΔΕΝ καταργουνται: παραμενουν στη μελετη και αναφερονται
# ΧΩΡΙΣΤΑ, ως η οικογενεια αστοχιας οπου ο πλεονασμος ΑΠΟΔΕΔΕΙΓΜΕΝΑ δεν
# μπορει να βοηθησει. Ενα ευρημα που δειχνει και που ΔΕΝ ισχυει ειναι πιο
# πειστικο απο ενα που «κερδιζει» παντου.
#
# Η ΔΙΚΑΙΟΛΟΓΗΣΗ ΕΙΝΑΙ ΡΕΑΛΙΣΜΟΣ, ΟΧΙ ΒΟΛΙΚΟ ΑΠΟΤΕΛΕΣΜΑ: κλειστος δρομος απο
# εργα, ατυχημα, πλημμυρα η διαδηλωση ειναι ασυγκριτα συχνοτερος απο μονιμη
# καταργηση σταση, και ασυγκριτα συχνοτερος απο κλεισιμο ΟΛΟΚΛΗΡΗΣ περιοχης
# οκτω κομβων. Το παλιο μειγμα εδινε στο σπανιοτερο γεγονος τη μεγαλυτερη
# βαρυτητα.
# Το μειγμα εστιαζει σε γεγονοτα που καταστρεφουν υποδομη - εκει εχει νοημα
# ο σχεδιαστικος πλεονασμος. Η αυξηση χρονου διαδρομης αφαιρεθηκε: δεν
# αποσυνδεει τιποτα, αρα ουτε δοκιμαζει την ανθεκτικοτητα ουτε δινει εργο
# αποκαταστασης. Η αυξηση ζητησης παραμενει ως αρνητικος μαρτυρας - ελεγχει
# οτι ο πρακτορας ΔΕΝ δαπανα οταν η υποδομη ειναι ακεραιη.
DISRUPTION_TYPE_PROBS = {
    "link_failure":        0.37,
    "demand_surge":        0.22,
    "combined_disruption": 0.17,
    "node_removal":        0.12,
    "area_closure":        0.12,
}

# Ποσοι δρομοι κλεινουν ταυτοχρονα.
LINK_FAILURE_COUNT = {
    "mild": 1,
    "moderate": 2,
    "severe": 4,
}

# Severity probabilities.
SEVERITY_PROBS = {
    "mild": 0.50,
    "moderate": 0.35,
    "severe": 0.15,
}

# Location probabilities per disruption type.
LOCATION_PROBS = {
    "node_removal": {
        "central": 0.50,
        "peripheral": 0.50,
    },
    "area_closure": {
        "central_area": 0.50,
        "peripheral_area": 0.50,
    },
    "demand_surge": {
        "central_area": 0.50,
        "peripheral_area": 0.50,
    },
    # Η θεση της διακοπης οριζεται στο ΟΔΙΚΟ δικτυο, οχι στις γραμμες.
    "link_failure": {
        "any_road": 1.00,
    },
}

# Severity definitions.
NODE_REMOVAL_COUNT = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

AREA_CLOSURE_NODE_COUNT = {
    "mild": 3,
    "moderate": 5,
    "severe": 8,
}

DEMAND_SURGE_FACTOR = {
    "mild": 1.25,
    "moderate": 1.50,
    "severe": 2.00,
}

# ΠΡΟΣΒΑΣΗ ΜΕ ΠΕΡΠΑΤΗΜΑ ΟΤΑΝ ΚΛΕΙΝΕΙ ΣΤΑΣΗ
# --------------------------------------------------------------------------
# Οταν κλεινει ενας κομβος, ο επιβατης ΔΕΝ εξαφανιζεται: περπαταει στην
# πλησιεστερη ΑΝΟΙΧΤΗ σταση εντος WALK_ACCESS_RADIUS_M και ταξιδευει απο εκει.
# Η ζητηση μεταφερεται στον νεο κομβο και ο χρονος περπατηματος χρεωνεται.
# Οποιος κλειστος κομβος ΔΕΝ εχει ανοιχτο γειτονα εντος της ακτινας, η ζητηση
# του παραμενει ανεξυπηρετητη - οπως και πριν.
#
# Η ακτινα 400 m ειναι η καθιερωμενη ακτινα προσβασης σε σταση (~5 λεπτα με
# 4.8 km/h). Το WALK_ACCESS_RADIUS_M = 300 δινεται ως αναλυση ευαισθησιας.
WALK_ACCESS_ENABLED = True
WALK_ACCESS_RADIUS_M = 400.0
WALK_SPEED_KMH = 4.8

# Combined disruption configuration.
# In combined_disruption, the script randomly combines two of the four base types.
COMBINED_BASE_TYPES = [
    "link_failure",
    "node_removal",
    "area_closure",
    "demand_surge",
]

# Το ΑΡΧΙΚΟ δικτυο R0 δεχεται ΤΙΣ ΙΔΙΕΣ διαταραχες, ως μετρο συγκρισης.
# Χωρις αυτο δεν υπαρχει τροπος να δειχθει οτι ο σχεδιασμος αγορασε
# ανθεκτικοτητα - μονο οτι το σχεδιασμενο δικτυο επιβιωνει «καπως».
INCLUDE_R0_REFERENCE = True

# Η ΔΙΑΤΑΡΑΧΗ ΔΕΝ ΕΠΙΤΡΕΠΕΤΑΙ ΝΑ ΕΞΑΡΤΑΤΑΙ ΑΠΟ ΤΟ ΔΙΚΤΥΟ ΠΟΥ ΕΛΕΓΧΟΥΜΕ.
# --------------------------------------------------------------------------
# Οι `pick_nodes_from_location` και `select_routes_by_location` διαλεγουν
# κομβους/ακμες ΑΠΟ ΤΙΣ ΓΡΑΜΜΕΣ του τρεχοντος συνολου. Με 15 γραμμες (R0) και
# 12 γραμμες (R*), η ιδια σπορα δινει ΔΙΑΦΟΡΕΤΙΚΗ διαταραχη στο καθενα - και
# τοτε η συγκριση των δυο δικτυων ειναι ΑΚΥΡΗ, οσα Common Random Numbers κι αν
# δηλωσουμε.
#
# Με CRN_ACROSS_NETWORKS = True η επιλογη γινεται ΑΠΟ ΤΟ ΔΙΚΤΥΟ (ολοι οι
# κομβοι, ολες οι ακμες), οποτε καθε δικτυο δεχεται ΤΗΝ ΙΔΙΑ ΑΚΡΙΒΩΣ ζημια.
# Ειναι και φυσικα ορθοτερο: μια πλημμυρα δεν ρωταει ποιες σταση χρησιμοποιουν
# οι γραμμες σου.
CRN_ACROSS_NETWORKS = True

# =============================================================================
# =============================== HELPERS ======================================
# =============================================================================


def weighted_choice(prob_dict: Dict[str, float]) -> str:
    keys = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]


def safe_pct(delta: Optional[float], base: Optional[float]) -> Optional[float]:
    if delta is None or base is None:
        return None
    if abs(float(base)) < 1e-12:
        return None
    return float(delta) / float(base)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_import_paths(*folders: Path) -> None:
    for folder in folders:
        folder_str = str(folder.resolve())
        if folder_str not in sys.path:
            sys.path.insert(0, folder_str)


def import_training_module(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Training module not found: {module_path}")
    ensure_import_paths(module_path.parent)
    spec = importlib.util.spec_from_file_location("rivera_v7_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_nodes_from_routes(routes: List[List[int]]) -> List[int]:
    return sorted({int(n) for r in routes for n in r})


def route_central_count(route: List[int], central_set: set[int]) -> int:
    return sum(1 for n in route if n in central_set)


# ΣΧΕΔΙΑΣΤΙΚΗ ΕΠΙΛΟΓΗ, οχι αποτελεσμα βελτιστοποιησης. Μια γραμμη θεωρειται
#      «κυρια» αν διασχιζει τουλαχιστον 3 κεντρικους κομβους (απολυτο κατωφλι,
#      πιανει τις μακριες διαμπερεις γραμμες) Η αν τουλαχιστον το MAIN_ROUTE_SHARE
#      των σταςεων της ειναι κεντρικες (σχετικο κατωφλι, πιανει τις κοντες
#      γραμμες που εξυπηρετουν το κεντρο). Το `or` ειναι σκοπιμο: τα δυο
#      κατωφλια καλυπτουν διαφορετικα ειδη γραμμης. Οι τιμες αναφερονται στο
#      JSON εξοδου ωστε η επιλογη να ειναι ελεγξιμη και αναπαραγωγιμη.
MAIN_ROUTE_MIN_CENTRAL = 3
MAIN_ROUTE_SHARE = 0.25


def route_is_main(route: List[int], central_set: set[int]) -> bool:
    if not route:
        return False
    c = route_central_count(route, central_set)
    return (c >= MAIN_ROUTE_MIN_CENTRAL) or (c / max(1, len(route)) >= MAIN_ROUTE_SHARE)


def pick_nodes_from_location(routes: List[List[int]], location: str, count: int, all_nodes: List[int]) -> List[int]:
    # Οταν συγκρινονται δικτυα, η επιλογη γινεται απο ΟΛΟΥΣ τους κομβους
    # του δικτυου και οχι απο τους κομβους των γραμμων του καθενος.
    if CRN_ACROSS_NETWORKS:
        central_set = set(CENTRAL_NODES)
        if location in ("central", "central_area"):
            pool = [n for n in all_nodes if n in central_set]
        elif location in ("peripheral", "peripheral_area"):
            pool = [n for n in all_nodes if n not in central_set]
        else:
            pool = list(all_nodes)
        if not pool:
            pool = list(all_nodes)
        return random.sample(pool, k=min(count, len(pool)))
    central_set = set(CENTRAL_NODES)
    present_nodes = unique_nodes_from_routes(routes)

    if location in ("central", "central_area"):
        pool = [n for n in present_nodes if n in central_set]
    elif location in ("peripheral", "peripheral_area"):
        pool = [n for n in present_nodes if n not in central_set]
    else:
        pool = present_nodes[:]

    if not pool:
        pool = present_nodes[:]
    if not pool:
        return []
    k = min(count, len(pool))
    return random.sample(pool, k=k)


def shortest_path_avoiding(
    TT_dict: Dict[int, Dict[int, float]],
    blocked: set,
    a: int,
    b: int,
) -> Optional[List[int]]:
    """Dijkstra απο a σε b στον γραφο ΧΩΡΙΣ τους κομβους `blocked`.
    Επιστρεφει τη λιστα κομβων του μονοπατιου, η None αν δεν υπαρχει."""
    if a in blocked or b in blocked:
        return None
    if a == b:
        return [a]
    dist = {a: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, a)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == b:
            path = [b]
            while path[-1] != a:
                path.append(prev[path[-1]])
            return path[::-1]
        for v, w in TT_dict.get(u, {}).items():
            if v in blocked or v in seen:
                continue
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def pick_contiguous_cluster(
    routes: List[List[int]],
    location: str,
    count: int,
    all_nodes: List[int],
    TT_dict: Dict[int, Dict[int, float]],
) -> List[int]:
    """Επιλογη ΓΕΙΤΟΝΙΚΗΣ συσταδας `count` κομβων με BFS απο τυχαιο σπορο,
    οπως περιγραφει το §3.4 («ομαδα γειτονικων κομβων»). Χρησιμοποιειται μονο
    οταν AREA_CLOSURE_CONTIGUOUS = True. Αν η συσταδα δεν συμπληρωνεται (ο
    σπορος ειναι σε απομονωμενο τμημα), επιστρεφει οσους βρηκε."""
    central_set = set(CENTRAL_NODES)
    present_nodes = unique_nodes_from_routes(routes)
    if location in ("central", "central_area"):
        pool = [n for n in present_nodes if n in central_set]
    elif location in ("peripheral", "peripheral_area"):
        pool = [n for n in present_nodes if n not in central_set]
    else:
        pool = present_nodes[:]
    if not pool:
        pool = present_nodes[:]
    if not pool:
        return []

    pool_set = set(pool)
    seed_node = random.choice(pool)
    cluster = [seed_node]
    seen = {seed_node}
    queue = [seed_node]
    while queue and len(cluster) < count:
        u = queue.pop(0)
        for v in TT_dict.get(u, {}):
            if v in seen or v not in pool_set:
                continue
            seen.add(v)
            cluster.append(v)
            queue.append(v)
            if len(cluster) >= count:
                break
    return cluster


def dedup_routes(routes: List[List[int]]) -> List[List[int]]:
    """Αφαιρει ταυτοσημες γραμμες (και ως προς την αντιστροφη φορα).
    Στο σημερινο σχημα δεν εμφανιζονται ποτε (0/300 runs), γινονται ομως
    δυνατες μετα το οταν μια γραμμη σπαει σε τμηματα. Η evaluateattacks
    κανει set() ενω η compute_vkm_vh μετραει καθε γραμμη - χωρις dedup το
    κοστος και η συνδεσιμοτητα θα διαφωνουσαν."""
    seen = set()
    out: List[List[int]] = []
    for r in routes:
        k = tuple(r)
        kr = tuple(reversed(r))
        if k in seen or kr in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def count_invalid_edges(routes: List[List[int]], TT_dict: Dict[int, Dict[int, float]]) -> int:
    """Ποσες ακμες των γραμμων ΔΕΝ υπαρχουν στον γραφο. Μετα το
    πρεπει να ειναι 0 σε καθε run - ειναι η αποδειξη οτι το διαταραγμενο
    δικτυο ειναι φυσικα συνεπες, οχι απλως ισχυρισμος."""
    bad = 0
    for r in routes:
        for k in range(len(r) - 1):
            if r[k + 1] not in TT_dict.get(r[k], {}):
                bad += 1
    return bad


def remove_nodes_from_routes(
    routes: List[List[int]],
    nodes_to_remove: List[int],
    TT_dict: Optional[Dict[int, Dict[int, float]]] = None,
    max_len: int = 40,
) -> List[List[int]]:
    """Αφαιρει κομβους ΚΑΙ επισκευαζει τη συνεχεια της γραμμης.

    ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΔΙΟΡΘΩΝΕΤΑΙ: ο παλιος κωδικας εκανε μονο
        new_r = [n for n in r if n not in remove_set]
    Απο τη γραμμη [1, 2, 3, 7, ...] με κλειστο τον κομβο 2 επαιρνε [1, 3, 7, ...]
    - αλλα η ακμη 1->3 ΔΕΝ ΥΠΑΡΧΕΙ στο δικτυο της Rivera (ο κομβος 1 εχει εναν
    μονο γειτονα, τον 2). Το λεωφορειο περναγε πανω απο τον κλειστο κομβο.
    Μετρημενο σε 60 διαταραχες: 58/60 runs (97%) ειχαν τουλαχιστον μια
    ανυπαρκτη ακμη, 6.62 ανα run κατα μεσο ορο. Η αρχικη λυση εχει 0/163.

    Η ΔΙΟΡΘΩΣΗ: για καθε ζευγος διαδοχικων επιζωντων κομβων a, b
      1. αν η ακμη a->b υπαρχει, εντάξει
      2. αλλιως αναζητειται ΠΡΑΓΜΑΤΙΚΟ συντομοτερο μονοπατι a->b στον γραφο
         ΧΩΡΙΣ τους κλειστους κομβους (παρακαμψη)
      3. αν δεν υπαρχει καμια παρακαμψη, η γραμμη ΣΠΑΕΙ σε δυο ανεξαρτητα
         τμηματα - αυτο ακριβως συμβαινει οταν κλεινει κομβος-αρθρωση.

    Οταν το TT_dict ειναι None η συμπεριφορα ειναι η ΠΑΛΙΑ (συμβατοτητα)."""
    remove_set = set(nodes_to_remove)

    if TT_dict is None:
        new_routes = []
        for r in routes:
            new_r = [n for n in r if n not in remove_set]
            cleaned = []
            for n in new_r:
                if not cleaned or cleaned[-1] != n:
                    cleaned.append(n)
            if len(cleaned) >= 2:
                new_routes.append(cleaned)
        return new_routes

    new_routes: List[List[int]] = []
    for r in routes:
        surv = [n for n in r if n not in remove_set]
        cleaned = []
        for n in surv:
            if not cleaned or cleaned[-1] != n:
                cleaned.append(n)
        if len(cleaned) < 2:
            continue

        seg = [cleaned[0]]
        for k in range(len(cleaned) - 1):
            a, b = cleaned[k], cleaned[k + 1]
            if b in TT_dict.get(a, {}):
                seg.append(b)
                continue
            detour = shortest_path_avoiding(TT_dict, remove_set, a, b)
            if detour is not None and len(seg) + len(detour) - 1 <= max_len:
                seg.extend(detour[1:])
            else:
                if len(seg) >= 2:
                    new_routes.append(seg)
                seg = [b]
        if len(seg) >= 2:
            new_routes.append(seg)

    return dedup_routes(new_routes)


def copy_tt_dict(TT_dict: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
    return {int(i): {int(j): float(v) for j, v in nbrs.items()} for i, nbrs in TT_dict.items()}


def copy_td_dict(TD_dict: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
    return {int(i): {int(j): float(v) for j, v in nbrs.items()} for i, nbrs in TD_dict.items()}


def closed_nodes_from_details(details: Dict[str, Any]) -> set:
    """Ολοι οι κομβοι που εκλεισαν σε αυτη τη διαταραχη (και στα
    combined, απο ολα τα συστατικα)."""
    out: set = set()
    for comp in (details.get("components") or [details]):
        for key in ("closed_nodes", "removed_nodes"):
            for n in (comp.get(key) or []):
                out.add(int(n))
    return out


def nearest_open_node(node: int, open_nodes: List[int], coords, mod,
                      radius_m: float):
    """Πλησιεστερος ΑΝΟΙΧΤΟΣ κομβος εντος `radius_m`, σε ευθεια γραμμη
    πανω στις πραγματικες συντεταγμενες. Επιστρεφει (κομβος, μετρα) η (None, None)."""
    if node not in coords:
        return None, None
    lat1, lon1 = coords[node]
    best = None
    best_d = None
    for j in open_nodes:
        if j == node or j not in coords:
            continue
        lat2, lon2 = coords[j]
        d = mod.haversine_km(lat1, lon1, lat2, lon2) * 1000.0
        if d <= radius_m and (best_d is None or d < best_d):
            best, best_d = j, d
    return best, best_d


def apply_walk_access(TD_dict: Dict[int, Dict[int, float]], closed: set,
                      coords, mod, all_nodes: List[int],
                      radius_m: float = None, speed_kmh: float = None):
    """Ανακατανομη της ζητησης των ΚΛΕΙΣΤΩΝ κομβων στους πλησιεστερους
    ανοιχτους, με χρεωση του χρονου περπατηματος.

    ΓΙΑΤΙ: χωρις αυτο, το μοντελο υποθετει οτι ο επιβατης εξαφανιζεται. Η
    ζητηση του εμενε στον πινακα και μετριοταν ως ανεξυπηρετητη ΓΙΑ ΠΑΝΤΑ,
    ανεξαρτητα απο το τι θα εκανε ο σχεδιαστης η ο πρακτορας αποκαταστασης.
    Μετρημενο: 8.40 % της συνολικης ζητησης, δηλαδη το 77 % ολης της χαμενης
    καλυψης, ηταν «αδυνατη» ΜΟΝΟ εξαιτιας αυτης της παραδοχης.

    ΤΙ ΚΑΝΕΙ: για καθε κλειστο κομβο k βρισκει τον πλησιεστερο ανοιχτο κομβο
    m εντος της ακτινας. Ολη η ζητηση με προελευση η προορισμο τον k
    μεταφερεται στον m. Οσοι κλειστοι κομβοι δεν εχουν ανοιχτο γειτονα εντος
    της ακτινας, η ζητηση τους μενει οπου ηταν και παραμενει ανεξυπηρετητη.

    ΤΙ ΔΕΝ ΚΑΝΕΙ: δεν αδυνατιζει τη διαταραχη. Κλεινουν ΑΚΡΙΒΩΣ οι ιδιοι
    κομβοι. Και δεν ενεργοποιειται ποτε σε αθικτο δικτυο, οπου κανενας
    κομβος δεν ειναι κλειστος - αρα ο ΣΧΕΔΙΑΣΜΟΣ (v6/v7) δεν επηρεαζεται
    καθολου και ΔΕΝ χρειαζεται να ξανατρεξει.

    Επιστρεφει (TD_new, info) οπου το info περιεχει και τα συνολικα
    επιβατο-λεπτα περπατηματος, για τη χρεωση στο ATT."""
    radius_m = WALK_ACCESS_RADIUS_M if radius_m is None else radius_m
    speed_kmh = WALK_SPEED_KMH if speed_kmh is None else speed_kmh
    info = {
        "enabled": bool(WALK_ACCESS_ENABLED),
        "radius_m": float(radius_m),
        "speed_kmh": float(speed_kmh),
        "closed_nodes_count": len(closed),
        "reassigned_nodes": 0,
        "stranded_nodes": 0,
        "reassigned_demand_share": 0.0,
        "stranded_demand_share": 0.0,
        "walk_pax_minutes": 0.0,
        "mean_walk_minutes": 0.0,
        "mapping": {},
    }
    if (not WALK_ACCESS_ENABLED) or (not closed):
        return TD_dict, info

    open_nodes = [n for n in all_nodes if n not in closed]
    m_per_min = float(speed_kmh) * 1000.0 / 60.0

    walk_min: Dict[int, float] = {}
    target: Dict[int, int] = {}
    for k in sorted(closed):
        j, d = nearest_open_node(k, open_nodes, coords, mod, radius_m)
        if j is not None:
            target[k] = j
            walk_min[k] = float(d) / m_per_min
    info["reassigned_nodes"] = len(target)
    info["stranded_nodes"] = len(closed) - len(target)
    info["mapping"] = {str(k): [int(v), round(walk_min[k], 3)] for k, v in target.items()}

    total = sum(sum(v.values()) for v in TD_dict.values())
    TD_new: Dict[int, Dict[int, float]] = {}
    for o, row in TD_dict.items():
        for d, dem in row.items():
            if dem <= 0:
                continue
            o2 = target.get(o, o)
            d2 = target.get(d, d)
            extra = 0.0
            if o in target:
                extra += walk_min[o]
            if d in target:
                extra += walk_min[d]
            if o in closed and o not in target:
                info["stranded_demand_share"] += dem
            elif d in closed and d not in target:
                info["stranded_demand_share"] += dem
            elif extra > 0.0:
                info["reassigned_demand_share"] += dem
                info["walk_pax_minutes"] += dem * extra
            if o2 == d2:
                # Μετα το περπατημα η αφετηρια και ο προορισμος συμπιπτουν:
                # ο επιβατης φτανει με τα ποδια, δεν χρειαζεται λεωφορειο.
                continue
            TD_new.setdefault(o2, {})
            TD_new[o2][d2] = TD_new[o2].get(d2, 0.0) + float(dem)

    if total > 0:
        info["reassigned_demand_share"] = 100.0 * info["reassigned_demand_share"] / total
        info["stranded_demand_share"] = 100.0 * info["stranded_demand_share"] / total
    if info["reassigned_demand_share"] > 0:
        info["mean_walk_minutes"] = (
            info["walk_pax_minutes"] / (info["reassigned_demand_share"] / 100.0 * total)
            if total > 0 else 0.0)
    return TD_new, info


def rebuild_tt_matrix_from_dict(TT_dict: Dict[int, Dict[int, float]], n_nodes: int):
    import numpy as np
    tt = np.full((n_nodes + 1, n_nodes + 1), float("inf"), dtype=float)
    for i in range(1, n_nodes + 1):
        tt[i, i] = 0.0
    for i, nbrs in TT_dict.items():
        for j, val in nbrs.items():
            if 1 <= i <= n_nodes and 1 <= j <= n_nodes:
                tt[i, j] = float(val)
    return tt


def rebuild_td_matrix_from_dict(TD_dict: Dict[int, Dict[int, float]], n_nodes: int):
    import numpy as np
    td = np.zeros((n_nodes + 1, n_nodes + 1), dtype=float)
    for i, nbrs in TD_dict.items():
        for j, val in nbrs.items():
            if 1 <= i <= n_nodes and 1 <= j <= n_nodes:
                td[i, j] = float(val)
    return td


def clear_module_caches(mod) -> None:
    for cache_name in [
        "ROUTE_METRIC_CACHE",
        "ATTACK_EVAL_CACHE",
        "SOLUTION_EVAL_CACHE",
        "INITIAL_FEASIBLE_ROUTESET_CACHE",
    ]:
        obj = getattr(mod, cache_name, None)
        if hasattr(obj, "clear"):
            obj.clear()


def evaluate_routeset(
    mod,
    routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    tt_matrix,
    dist_km: Dict[int, Dict[int, float]],
    compute_energy: bool,
    walk_pax_minutes: float = 0.0,
) -> Dict[str, Any]:
    clear_module_caches(mod)

    # Evaluate ATT / CEF / DUN.
    _, _, _, dun, att, cef, cef2, exposure = mod.evaluateattacks_ext(
        routes, TT_dict, TD_dict, tt_matrix)

    cost, vkm, vh = mod.compute_operating_cost(
        routes,
        TT_dict,
        dist_km,
        c_km=C_KM,
        c_h=C_H,
        operating_hours=OPERATING_HOURS,
        headway_min=HEADWAY_MIN,
        assume_bidirectional=ASSUME_BIDIRECTIONAL,
    )

    energy = None
    if compute_energy:
        energy = mod.compute_energy_kwh(
            routes,
            TT_dict,
            dist_km,
            e_km=E_KM,
            operating_hours=OPERATING_HOURS,
            headway_min=HEADWAY_MIN,
            assume_bidirectional=ASSUME_BIDIRECTIONAL,
        )

    # Γενικευμενο κοστος. Ο `att` μενει ΑΚΡΙΒΩΣ ως ειχε - προστιθεται το `gc`.
    u = max(0.0, min(1.0, float(dun) / 100.0))
    gc = float(att) * (1.0 - u) + P_UNSERVED * u

    # Ο χρονος ΠΕΡΠΑΤΗΜΑΤΟΣ χρεωνεται ΕΠΙΠΛΕΟΝ, σε νεα πεδια. Το `att`
    # και το `gc` μενουν ΑΚΡΙΒΩΣ ως ειχαν, ωστε ολοι οι υπαρχοντες πινακες της
    # διπλωματικης να παραμενουν συγκρισιμοι. Το `att_access` ειναι ο χρονος
    # πορτα-προς-πορτα (λεωφορειο + περπατημα) και το `gc_access` το αντιστοιχο
    # γενικευμενο κοστος.
    _total_demand = sum(sum(v.values()) for v in TD_dict.values())
    _served_mass = max(1e-9, (1.0 - u) * _total_demand)
    _att_walk = float(walk_pax_minutes) / _served_mass if walk_pax_minutes else 0.0
    att_access = float(att) + _att_walk
    gc_access = att_access * (1.0 - u) + P_UNSERVED * u

    out = {
        "att": float(att),
        "att_access": float(att_access),
        "att_walk_component": float(_att_walk),
        "CEF": float(cef),
        "CEF2": float(cef2),
        "exposure": float(exposure),
        "dun": float(dun),
        "gc": float(gc),
        "gc_access": float(gc_access),
        "cost": float(cost),
        "energy": None if energy is None else float(energy),
        "vkm": float(vkm),
        "vh": float(vh),
        "num_routes": len(routes),
        "num_unique_nodes": len(unique_nodes_from_routes(routes)),
        # Αποδειξη φυσικης συνεπειας: μετα το πρεπει να ειναι 0 παντου.
        "invalid_edges": count_invalid_edges(routes, TT_dict),
    }
    # Αναλυση ευαισθησιας ως προς την ποινη P.
    for P in P_UNSERVED_SENSITIVITY:
        out[f"gc_P{int(P)}"] = float(att) * (1.0 - u) + float(P) * u
    return out


def compare_metrics(new: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Το "gc" προστιθεται· κανενα υπαρχον πεδιο δεν αφαιρειται.
    for key in ["att", "att_access", "CEF", "CEF2", "exposure", "dun", "gc", "gc_access",
                "cost", "energy", "vkm", "vh", "num_routes", "num_unique_nodes"]:
        new_val = new.get(key)
        base_val = base.get(key)
        if new_val is None or base_val is None:
            delta = None
            pct = None
        else:
            delta = float(new_val) - float(base_val)
            pct = safe_pct(delta, float(base_val))
        out[f"delta_{key}"] = delta
        out[f"pct_delta_{key}"] = pct
    return out


def select_routes_by_location(routes: List[List[int]], location: str) -> List[int]:
    central_set = set(CENTRAL_NODES)
    main_indices = [i for i, r in enumerate(routes) if route_is_main(r, central_set)]
    secondary_indices = [i for i, r in enumerate(routes) if i not in main_indices]

    if location == "main_routes":
        return main_indices or list(range(len(routes)))
    if location == "secondary_routes":
        return secondary_indices or list(range(len(routes)))
    return list(range(len(routes)))


def route_edges(route: List[int]) -> List[Tuple[int, int]]:
    return [(route[i], route[i + 1]) for i in range(len(route) - 1)]


def apply_demand_surge(
    TD_dict: Dict[int, Dict[int, float]],
    all_nodes: List[int],
    location: str,
    severity: str,
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, Any]]:
    """Η περιοχη οριζεται στο ΔΙΚΤΥΟ, οχι στις γραμμες του καθε συνολου.

    Αν οι κομβοι-στοχοι επιλεγονταν απο τις γραμμες, το R0 (15 γραμμες, 84
    κομβοι) και το σχεδιασμενο δικτυο (12 γραμμες, 82 κομβοι) θα δεχονταν
    ΔΙΑΦΟΡΕΤΙΚΗ αυξηση με την ιδια σπορα και η συγκριση τους θα ηταν ακυρη."""
    TD_new = copy_td_dict(TD_dict)
    factor = DEMAND_SURGE_FACTOR[severity]

    central_set = set(CENTRAL_NODES)
    present_nodes = list(all_nodes)
    if location == "central_area":
        target_nodes = [n for n in present_nodes if n in central_set]
    else:
        target_nodes = [n for n in present_nodes if n not in central_set]

    if not target_nodes:
        target_nodes = present_nodes

    target_set = set(target_nodes)
    changed_pairs = 0
    # Increase OD demand whose destination is inside target area.
    for ori in TD_new:
        for dst in list(TD_new[ori].keys()):
            if dst in target_set:
                TD_new[ori][dst] *= factor
                changed_pairs += 1

    return TD_new, {
        "target_location": location,
        "target_nodes_count": len(target_nodes),
        "factor": factor,
        "changed_od_pairs": changed_pairs,
    }


def apply_link_failure(
    TT_dict: Dict[int, Dict[int, float]],
    severity: str,
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, Any]]:
    """ΔΙΑΚΟΠΗ ΣΥΝΔΕΣΗΣ: κλεινουν k δρομοι.

    ΚΡΙΣΙΜΟ - ΓΙΑΤΙ ΟΙ ΑΚΜΕΣ ΕΠΙΛΕΓΟΝΤΑΙ ΑΠΟ ΤΟ ΟΔΙΚΟ ΔΙΚΤΥΟ ΚΑΙ ΟΧΙ ΑΠΟ ΤΙΣ
    ΓΡΑΜΜΕΣ: η επιλογη ακμων πανω σε μια γραμμη
    του ΤΡΕΧΟΝΤΟΣ συνολου. Αν καναμε το ιδιο εδω, το R0 (15 γραμμες) και το R*
    (12 γραμμες) θα δεχονταν ΔΙΑΦΟΡΕΤΙΚΕΣ διαταραχες ακομη και με το ιδιο seed,
    και η συγκριση τους θα ηταν ΑΚΥΡΗ. Επιλεγοντας απο τις ακμες του δικτυου, η
    ιδια σπορα δινει ΤΟΥΣ ΙΔΙΟΥΣ κλειστους δρομους σε καθε δικτυο - πραγματικα
    Common Random Numbers, και καμια μεροληψια υπερ οποιουδηποτε σχεδιασμου."""
    TT_new = copy_tt_dict(TT_dict)
    undirected = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                         for u, row in TT_new.items() for v in row})
    if not undirected:
        return TT_new, {"failed_edges": [], "failed_edges_count": 0}

    k = min(LINK_FAILURE_COUNT[severity], len(undirected))
    chosen = random.sample(undirected, k=k)
    for a, b in chosen:
        TT_new.get(a, {}).pop(b, None)
        TT_new.get(b, {}).pop(a, None)

    return TT_new, {
        "failed_edges": [[a, b] for a, b in chosen],
        "failed_edges_count": len(chosen),
        "selection_scope": "road_network",
    }


def split_routes_at_closures(routes: List[List[int]],
                            closed: set,
                            TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """ΑΜΕΣΗ ΕΙΚΟΝΑ (t0): η γραμμη σταματα εκει που βρισκει τη ζημια.

    Κανεις δεν εχει προλαβει να σχεδιασει παρακαμψη - αυτο ειναι η πρωτη ωρα
    του συμβαντος. Η γραμμη σπαει σε ανεξαρτητα τμηματα οταν ο επομενος κομβος
    εχει κλεισει η οταν η ακμη προς αυτον δεν υπαρχει πια. Εδω φαινεται τι
    αξιζει ο πλεονασμος: αν αλλη γραμμη εξυπηρετει το ιδιο ζευγος ο επιβατης
    ταξιδευει, αλλιως χανεται."""
    out: List[List[int]] = []
    for r in routes:
        r = [int(n) for n in r]
        cur: List[int] = []
        for k, n in enumerate(r):
            if n in closed:
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
                continue
            if cur and n not in TT_dict.get(cur[-1], {}):
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            if not cur or cur[-1] != n:
                cur.append(n)
        if len(cur) >= 2:
            out.append(cur)
    return [x for x in out if len(x) >= 2]


def repair_routes_after_closures(routes: List[List[int]],
                                 closed: set,
                                 TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """ΜΕΤΑ ΤΗΝ ΠΑΡΑΚΑΜΨΗ (t1): ο φορεας εχει εκδωσει νεα διαδρομη.

    Καθε ζευγος διαδοχικων επιζωντων κομβων επανασυνδεεται με ΠΡΑΓΜΑΤΙΚΟ
    συντομοτερο μονοπατι στο υπολειπομενο οδικο δικτυο· αν δεν υπαρχει, η
    γραμμη σπαει. Καταγραφεται μονο ως σημειο αναφορας - ο πρακτορας
    αποκαταστασης ξεκινα απο το t0 και πληρωνει ο ιδιος την παρακαμψη."""
    base = remove_nodes_from_routes(routes, sorted(closed), TT_dict) if closed else routes
    return repair_routes_after_edge_loss(base, TT_dict)


def compute_coverage_floor(TT_dict: Dict[int, Dict[int, float]],
                           TD_dict: Dict[int, Dict[int, float]],
                           closed: set) -> float:
    """Το δαπεδο καλυψης (%): ζητηση που ΚΑΝΕΝΑ συνολο γραμμων δεν μπορει να
    εξυπηρετησει - ειτε γιατι εμεινε σε κλειστο κομβο, ειτε γιατι τα δυο ακρα
    δεν συνδεονται πια με δρομο. Ειναι κατω φραγμα που αποδεικνυεται, και η
    αποκατασταση κρινεται ως προς αυτο."""
    total = 0.0
    lost = 0.0
    adj = {i: {j for j in row if j not in closed}
           for i, row in TT_dict.items() if i not in closed}
    cache: Dict[int, set] = {}

    def reach(s: int) -> set:
        if s in cache:
            return cache[s]
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        cache[s] = seen
        return seen

    for o, row in (TD_dict or {}).items():
        for d, dem in row.items():
            dem = float(dem)
            if dem <= 0:
                continue
            total += dem
            o_i, d_i = int(o), int(d)
            if o_i in closed or d_i in closed:
                lost += dem
            elif d_i not in reach(o_i):
                lost += dem
    return (100.0 * lost / total) if total > 0 else 0.0


def split_routes_at_missing_edges(routes: List[List[int]],
                                  TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """ΑΜΕΣΗ ΕΙΚΟΝΑ (t0): ο δρομος κοπηκε, η γραμμη σπαει ΕΠΙΤΟΠΟΥ.

    Κανεις δεν εχει προλαβει να σχεδιασει παρακαμψη - αυτο ειναι το πρωτο
    24ωρο. ΕΔΩ φαινεται τι αξιζει ο πλεονασμος: αν υπαρχει αλλη γραμμη που
    εξυπηρετει το ιδιο ζευγος, ο επιβατης ταξιδευει· αλλιως χανεται."""
    out: List[List[int]] = []
    for r in routes:
        r = [int(n) for n in r]
        cur = [r[0]] if r else []
        for k in range(len(r) - 1):
            a, b = r[k], r[k + 1]
            if b in TT_dict.get(a, {}):
                cur.append(b)
            else:
                if len(cur) >= 2:
                    out.append(cur)
                cur = [b]
        if len(cur) >= 2:
            out.append(cur)
    return [x for x in out if len(x) >= 2]


def repair_routes_after_edge_loss(routes: List[List[int]],
                                  TT_dict: Dict[int, Dict[int, float]],
                                  max_len: int = 40) -> List[List[int]]:
    """ΜΕΤΑ ΤΗΝ ΠΑΡΑΚΑΜΨΗ (t1): ιδια λογικη με το , για ακμες.

    Για καθε ζευγος διαδοχικων κομβων που εχασε την ακμη του, αναζητειται
    ΠΡΑΓΜΑΤΙΚΟ συντομοτερο μονοπατι στο υπολειπομενο οδικο δικτυο. Αν δεν
    υπαρχει, η γραμμη σπαει."""
    out: List[List[int]] = []
    for r in routes:
        r = [int(n) for n in r]
        cur = [r[0]] if r else []
        for k in range(len(r) - 1):
            a, b = r[k], r[k + 1]
            if b in TT_dict.get(a, {}):
                cur.append(b)
                continue
            path = shortest_path_avoiding(TT_dict, set(), a, b)
            if path and len(path) >= 2 and len(cur) + len(path) - 1 <= max_len:
                cur.extend(int(x) for x in path[1:])
            else:
                if len(cur) >= 2:
                    out.append(cur)
                cur = [b]
        if len(cur) >= 2:
            out.append(cur)
    return [x for x in out if len(x) >= 2]


def apply_area_closure(
    routes: List[List[int]],
    location: str,
    severity: str,
    all_nodes: List[int],
    TT_dict: Optional[Dict[int, Dict[int, float]]] = None,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    count = AREA_CLOSURE_NODE_COUNT[severity]
    # Προεπιλογη: τυχαια επιλογη κομβων εντος της ζωνης (οπως τρεχει σημερα).
    # Με AREA_CLOSURE_CONTIGUOUS = True επιλεγεται ΓΕΙΤΟΝΙΚΗ συσταδα (BFS).
    if AREA_CLOSURE_CONTIGUOUS and TT_dict is not None:
        nodes_to_remove = pick_contiguous_cluster(routes, location, count, all_nodes, TT_dict)
    else:
        nodes_to_remove = pick_nodes_from_location(routes, location, count, all_nodes)
    return routes, {
        "closed_nodes": nodes_to_remove,
        "closed_nodes_count": len(nodes_to_remove),
    }


def apply_node_removal(
    routes: List[List[int]],
    location: str,
    severity: str,
    all_nodes: List[int],
    TT_dict: Optional[Dict[int, Dict[int, float]]] = None,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    count = NODE_REMOVAL_COUNT[severity]
    nodes_to_remove = pick_nodes_from_location(routes, location, count, all_nodes)
    # Οι γραμμες ΔΕΝ πειραζονται εδω. Το σπασιμο (t0) και η παρακαμψη (t1)
    # γινονται κεντρικα, ωστε καθε τυπος διαταραχης να αντιμετωπιζεται ιδια.
    return routes, {
        "removed_nodes": nodes_to_remove,
        "removed_nodes_count": len(nodes_to_remove),
    }


def edges_lost_from_details(details: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Ολες οι ακμες που κοπηκαν (και στα combined)."""
    out: List[Tuple[int, int]] = []
    for comp in (details.get("components") or [details]):
        for e in (comp.get("failed_edges") or []):
            out.append((int(e[0]), int(e[1])))
    return out


def random_location_for_type(disruption_type: str) -> str:
    probs = LOCATION_PROBS.get(disruption_type)
    if not probs:
        raise ValueError(f"No location probabilities defined for {disruption_type}")
    return weighted_choice(probs)


def apply_single_disruption(
    disruption_type: str,
    routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    severity: str,
    all_nodes: List[int],
) -> Tuple[List[List[int]], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict[str, Any]]:
    routes_new = copy.deepcopy(routes)
    TT_new = copy_tt_dict(TT_dict)
    TD_new = copy_td_dict(TD_dict)

    location = random_location_for_type(disruption_type)
    details: Dict[str, Any] = {
        "type": disruption_type,
        "location": location,
        "severity": severity,
    }

    if disruption_type == "link_failure":
        # Οι γραμμες ΔΕΝ επισκευαζονται εδω. Η επισκευη (t1) γινεται
        # χωριστα στη main, ωστε να καταγραφει ΚΑΙ η αμεση εικονα (t0).
        TT_new, d = apply_link_failure(TT_new, severity)
        details.update(d)

    elif disruption_type == "node_removal":
        routes_new, d = apply_node_removal(routes_new, location, severity, all_nodes, TT_new)
        details.update(d)

    elif disruption_type == "area_closure":
        routes_new, d = apply_area_closure(routes_new, location, severity, all_nodes, TT_new)
        details.update(d)

    elif disruption_type == "demand_surge":
        TD_new, d = apply_demand_surge(TD_new, all_nodes, location, severity)
        details.update(d)

    else:
        raise ValueError(f"Unsupported disruption type: {disruption_type}")

    return routes_new, TT_new, TD_new, details


def apply_monte_carlo_disruption(
    routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    all_nodes: List[int],
) -> Tuple[List[List[int]], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict[str, Any]]:
    disruption_type = weighted_choice(DISRUPTION_TYPE_PROBS)
    severity = weighted_choice(SEVERITY_PROBS)

    if disruption_type != "combined_disruption":
        return apply_single_disruption(disruption_type, routes, TT_dict, TD_dict, severity, all_nodes)

    # Combined disruption = two different base disruptions in the same run.
    first_type, second_type = random.sample(COMBINED_BASE_TYPES, k=2)
    routes_mid, TT_mid, TD_mid, details_1 = apply_single_disruption(
        first_type, routes, TT_dict, TD_dict, severity, all_nodes
    )
    routes_final, TT_final, TD_final, details_2 = apply_single_disruption(
        second_type, routes_mid, TT_mid, TD_mid, severity, all_nodes
    )

    details = {
        "type": "combined_disruption",
        "severity": severity,
        "components": [details_1, details_2],
        "location": f"{details_1.get('location')} + {details_2.get('location')}",
    }
    return routes_final, TT_final, TD_final, details


def summarize_runs(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    # Προστεθηκαν τα gc / delta_gc / pct_delta_gc και το invalid_edges.
    metric_keys = [
        "att", "CEF", "dun", "gc", "cost", "energy", "vkm", "vh", "invalid_edges",
        "delta_att", "delta_CEF", "delta_dun", "delta_gc", "delta_cost", "delta_energy",
        "delta_vkm", "delta_vh",
        "pct_delta_att", "pct_delta_CEF", "pct_delta_gc", "pct_delta_cost", "pct_delta_energy",
    ]

    summaries = []
    for key, items in groups.items():
        summary = {group_keys[i]: key[i] for i in range(len(group_keys))}
        summary["n"] = len(items)
        for mk in metric_keys:
            vals = [x.get(mk) for x in items if x.get(mk) is not None]
            if not vals:
                summary[f"mean_{mk}"] = None
                summary[f"std_{mk}"] = None
                summary[f"min_{mk}"] = None
                summary[f"max_{mk}"] = None
                summary[f"p95_{mk}"] = None
                continue
            vals_sorted = sorted(float(v) for v in vals)
            mean_val = sum(vals_sorted) / len(vals_sorted)
            summary[f"mean_{mk}"] = mean_val
            if len(vals_sorted) > 1:
                variance = sum((v - mean_val) ** 2 for v in vals_sorted) / len(vals_sorted)
                summary[f"std_{mk}"] = math.sqrt(variance)
            else:
                summary[f"std_{mk}"] = 0.0
            summary[f"min_{mk}"] = vals_sorted[0]
            summary[f"max_{mk}"] = vals_sorted[-1]
            p95_idx = min(len(vals_sorted) - 1, math.ceil(0.95 * len(vals_sorted)) - 1)
            summary[f"p95_{mk}"] = vals_sorted[p95_idx]
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# =============================== MAIN =========================================
# =============================================================================


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / DATA_DIR).resolve() if not Path(DATA_DIR).is_absolute() else Path(DATA_DIR).resolve()
    out_dir = (script_dir / OUT_DIR).resolve() if not Path(OUT_DIR).is_absolute() else Path(OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_json_path = Path(BEST_JSON_FILE)
    if not best_json_path.is_absolute():
        best_json_path = data_dir / best_json_path

    # Αν το ρητο μονοπατι δεν υπαρχει (π.χ. επειδη ξανατρεξε η εκπαιδευση και
    # δημιουργηθηκε νεος timestamped φακελος), βρες αυτοματα τον ΠΙΟ ΠΡΟΣΦΑΤΟ.
    # Ο τροπος αποθηκευσης ΔΕΝ αλλαζει - αλλαζει μονο το ΔΙΑΒΑΣΜΑ.
    if not best_json_path.exists():
        import datetime as _dt
        roots = [script_dir, data_dir]
        exact, loose = [], []
        for _r in roots:
            for _pat, _bucket in (("best_scenarios_baseline*.json", exact),
                                  ("*/best_scenarios_baseline*.json", exact),
                                  ("best_scenarios_*.json", loose),
                                  ("*/best_scenarios_*.json", loose)):
                _bucket.extend(_r.glob(_pat))
        # Προτεραιοτητα στο ΑΚΡΙΒΕΣ ονομα που γραφει ο trainer· τα χειροκινητα
        # αντιγραφα (π.χ. "best_scenarios_baseline v7.json") ερχονται μετα.
        exact = sorted({p.resolve() for p in exact}, key=lambda p: p.stat().st_mtime, reverse=True)
        loose = sorted({p.resolve() for p in loose} - set(exact),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = exact + loose
        if candidates:
            _chosen = candidates[0]
            _when = _dt.datetime.fromtimestamp(_chosen.stat().st_mtime)
            _age_days = (_dt.datetime.now() - _when).days
            print(f"Το {best_json_path} δεν βρεθηκε.")
            print(f"Χρηση του: {_chosen}")
            print(f"Ημερομηνια αρχειου: {_when:%Y-%m-%d %H:%M}  ({_age_days} ημερες πριν)")
            if _age_days >= 1:
                print("*** ΠΡΟΣΟΧΗ: το αρχειο ΔΕΝ ειναι σημερινο. Αν μολις ετρεξες")
                print("*** τα v6.py/v7.py, ΔΕΝ αναλυεις τη νεα λυση. Ελεγξε το μονοπατι.")
            if _chosen not in exact:
                print("*** Το ονομα δεν ειναι το ακριβες που γραφει ο trainer")
                print("*** (best_scenarios_baseline.json) - μαλλον χειροκινητο αντιγραφο.")
            best_json_path = _chosen

    training_module_path = Path(TRAINING_MODULE_FILE)
    if not training_module_path.is_absolute():
        training_module_path = data_dir / training_module_path

    # Αν το ονομα δεν ταιριαζει (π.χ. το αρχειο λεγεται v7.r2.py, v7_new.py,
    # v7 (1).py), ψαξε οποιοδηποτε v7*.py και μετα v6*.py στον ιδιο φακελο.
    # ΣΗΜΕΙΩΣΗ: το module χρησιμοποιειται ΜΟΝΟ για τις συναρτησεις αξιολογησης
    # (load_rivera / evaluateattacks / compute_operating_cost / compute_energy_kwh),
    # οι οποιες ειναι ΤΑΥΤΟΣΗΜΕΣ σε v6 και v7 - το disruption δεν χρησιμοποιει
    # ενεργειες. Αυτο που καθοριζει ΤΙ αναλυεται ειναι το best_scenarios JSON.
    if not training_module_path.exists():
        _cands = []
        for _pat in ("v7*.py", "v6*.py"):
            for _r in (script_dir, data_dir):
                _cands.extend(q for q in _r.glob(_pat)
                              if q.name not in ("v7_new.py", "v6_new.py") or True)
        _cands = sorted({q.resolve() for q in _cands}, key=lambda q: q.stat().st_mtime, reverse=True)
        if _cands:
            print(f"Το {training_module_path.name} δεν βρεθηκε. Χρηση του: {_cands[0].name}")
            print("(χρησιμοποιειται μονο για τις συναρτησεις αξιολογησης - ιδιες σε v6/v7)")
            training_module_path = _cands[0]

    seek_folder = Path(SEEKALLPATHS_FOLDER)
    if not seek_folder.is_absolute():
        seek_folder = data_dir / seek_folder

    ensure_import_paths(data_dir, seek_folder, training_module_path.parent)

    random.seed(RANDOM_SEED)

    best_data = load_json(best_json_path)
    mod = import_training_module(training_module_path)

    travel_file = data_dir / TRAVEL_FILE
    demand_file = data_dir / DEMAND_FILE
    coords_file = data_dir / COORDS_FILE

    TT_dict, TD_dict, tt_matrix, _ = mod.load_rivera(
        travel_file=str(travel_file),
        demand_file=str(demand_file),
        coords_file=str(coords_file),
    )
    n_nodes = len(tt_matrix) - 1
    all_nodes = list(range(1, n_nodes + 1))

    coords = mod.read_coords_latlon(str(coords_file))
    dist_km = mod.build_directed_dist_km(TT_dict, coords)

    scenario_routes = {
        "S1_service": best_data["best_scenarios"]["S1_service"]["best_routes"],
        "S2_energy": best_data["best_scenarios"]["S2_energy"]["best_routes"],
    }
    if INCLUDE_R0_REFERENCE:
        _r0 = (best_data.get("initial_solution") or {}).get("routes")
        if _r0:
            scenario_routes["R0_reference"] = _r0
            print(f"Δικτυο αναφορας R0: {len(_r0)} γραμμες - δεχεται τις ΙΔΙΕΣ διαταραχες.")
        else:
            print("ΠΡΟΣΟΧΗ: δεν βρεθηκε initial_solution στο best_scenarios JSON.")

    results: Dict[str, Any] = {
        "methodological_note": (
            "Monte Carlo disruptions are applied post-optimization on v7 S1_service and S2_energy "
            "best routesets. No RL retraining is performed. Each run randomly selects disruption type, "
            "location, and severity. Central nodes are defined as 18..34. "
            "Node removal and area closure rebuild each affected route through a real "
            "shortest path in the graph with the closed nodes deleted; a route is split when no "
            "detour exists, so no route ever traverses a non-existent edge (see invalid_edges). "
            "Alongside att (computed over served demand only) the generalized cost "
            "gc = att*(1-u) + P*u with u = DUN/100 is reported, so that unserved demand is not "
            "silently removed from the average. Common random numbers: run k faces the same "
            "disruption in every scenario, enabling paired comparisons."
        ),
        "inputs": {
            "best_json": str(best_json_path),
            "training_module": str(training_module_path),
            "data_dir": str(data_dir),
            "monte_carlo_runs": MONTE_CARLO_RUNS,
            "random_seed": RANDOM_SEED,
            # παραμετροι των διορθωσεων, ωστε το run να ειναι αναπαραγωγιμο
            "use_common_random_numbers": USE_COMMON_RANDOM_NUMBERS,
            "p_unserved_min": P_UNSERVED,
            "p_unserved_sensitivity": P_UNSERVED_SENSITIVITY,
            "main_route_min_central": MAIN_ROUTE_MIN_CENTRAL,
            "main_route_share": MAIN_ROUTE_SHARE,
            "central_nodes": CENTRAL_NODES,
            "disruption_type_probs": DISRUPTION_TYPE_PROBS,
            "severity_probs": SEVERITY_PROBS,
            "location_probs": LOCATION_PROBS,
            "assumptions": {
                "c_km": C_KM,
                "c_h": C_H,
                "e_km": E_KM,
                "headway_min": HEADWAY_MIN,
                "operating_hours": OPERATING_HOURS,
                "assume_bidirectional": ASSUME_BIDIRECTIONAL,
            },
        },
        "scenarios": {},
    }
    # Καταγραφη των παραμετρων περπατηματος στο JSON εξοδου.
    try:
        results["inputs"]["link_failure_count"] = dict(LINK_FAILURE_COUNT)
        results["inputs"]["include_R0_reference"] = bool(INCLUDE_R0_REFERENCE)
        results["inputs"]["crn_across_networks"] = bool(CRN_ACROSS_NETWORKS)
        results["inputs"]["walk_access_enabled"] = bool(WALK_ACCESS_ENABLED)
        results["inputs"]["walk_access_radius_m"] = float(WALK_ACCESS_RADIUS_M)
        results["inputs"]["walk_speed_kmh"] = float(WALK_SPEED_KMH)
    except Exception:
        pass

    flat_rows: List[Dict[str, Any]] = []

    for scenario_name, base_routes in scenario_routes.items():
        compute_energy = scenario_name == "S2_energy"
        base_routes = copy.deepcopy(base_routes)

        baseline_metrics = evaluate_routeset(
            mod, base_routes, TT_dict, TD_dict, tt_matrix, dist_km, compute_energy=compute_energy
        )

        scenario_obj = {
            "baseline": baseline_metrics,
            "runs": [],
        }

        for run_id in range(1, MONTE_CARLO_RUNS + 1):
            # Common Random Numbers: το run #k δεχεται ΤΗΝ ΙΔΙΑ διαταραχη σε
            # ολα τα σεναρια, ωστε η συγκριση S1 vs S2 να ειναι ΖΕΥΓΑΡΩΤΗ.
            if USE_COMMON_RANDOM_NUMBERS:
                random.seed(RANDOM_SEED * 100000 + run_id)
            disrupted_routes, TT_disrupted, TD_disrupted, disruption_details = apply_monte_carlo_disruption(
                base_routes, TT_dict, TD_dict, all_nodes
            )

            # ΠΡΟΣΒΑΣΗ ΜΕ ΠΕΡΠΑΤΗΜΑ. Εφαρμοζεται ΜΟΝΟ οταν εχουν κλεισει
            # κομβοι· σε καθε αλλη περιπτωση το TD_disrupted μενει αναλλοιωτο.
            _closed = closed_nodes_from_details(disruption_details)
            TD_disrupted, _walk = apply_walk_access(
                TD_disrupted, _closed, coords, mod, all_nodes)
            disruption_details["walk_access"] = _walk

            tt_disrupted_matrix = rebuild_tt_matrix_from_dict(TT_disrupted, n_nodes)

            # Το δαπεδο καλυψης αυτης της διαταραχης - ανεξαρτητο απο το συνολο
            # γραμμων. Καταγραφεται εδω ωστε η αποκατασταση να μην το ξαναβρει.
            disruption_details["coverage_floor_percent"] = compute_coverage_floor(
                TT_disrupted, TD_disrupted, _closed)

            # t0 - ΑΜΕΣΗ ΕΙΚΟΝΑ. Ειναι το κυριο μετρο: δειχνει τι αντεχει ο
            # σχεδιασμος απο μονος του, πριν παρεμβει κανεις.
            routes_t0 = split_routes_at_closures(disrupted_routes, _closed, TT_disrupted)
            metrics = evaluate_routeset(
                mod, routes_t0, TT_disrupted, TD_disrupted, tt_disrupted_matrix,
                dist_km, compute_energy=compute_energy,
                walk_pax_minutes=_walk.get("walk_pax_minutes", 0.0),
            )
            metrics["phase"] = "t0_immediate"
            metrics["broken_routes"] = len(routes_t0)

            # t1 - σημειο αναφορας: τι θα εδινε η αυτοματη παρακαμψη. Δεν
            # τροφοδοτει την αποκατασταση· υπαρχει για συγκριση στο κειμενο.
            routes_t1 = repair_routes_after_closures(disrupted_routes, _closed, TT_disrupted)
            metrics_after_detour = evaluate_routeset(
                mod, routes_t1, TT_disrupted, TD_disrupted, tt_disrupted_matrix,
                dist_km, compute_energy=compute_energy,
                walk_pax_minutes=_walk.get("walk_pax_minutes", 0.0),
            )
            metrics_after_detour["phase"] = "t1_after_detour"
            comparison = compare_metrics(metrics, baseline_metrics)

            run_obj = {
                "run": run_id,
                "scenario": scenario_name,
                "disruption": disruption_details,
                "metrics": metrics,
                "comparison_vs_baseline": comparison,
            }
            run_obj["metrics_after_detour"] = metrics_after_detour
            run_obj["comparison_detour_vs_baseline"] = compare_metrics(
                metrics_after_detour, baseline_metrics)
            scenario_obj["runs"].append(run_obj)

            row = {
                "scenario": scenario_name,
                "run": run_id,
                "type": disruption_details.get("type"),
                "location": disruption_details.get("location"),
                "severity": disruption_details.get("severity"),
            }

            if disruption_details.get("type") == "combined_disruption":
                comps = disruption_details.get("components", [])
                row["component_1_type"] = comps[0].get("type") if len(comps) > 0 else None
                row["component_1_location"] = comps[0].get("location") if len(comps) > 0 else None
                row["component_2_type"] = comps[1].get("type") if len(comps) > 1 else None
                row["component_2_location"] = comps[1].get("location") if len(comps) > 1 else None
            else:
                row["component_1_type"] = None
                row["component_1_location"] = None
                row["component_2_type"] = None
                row["component_2_location"] = None

            row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
            row.update(metrics)
            row.update(comparison)
            flat_rows.append(row)

        results["scenarios"][scenario_name] = scenario_obj

    # Summaries.
    summary_by_type = summarize_runs(flat_rows, ["scenario", "type"])
    summary_by_type_location = summarize_runs(flat_rows, ["scenario", "type", "location"])
    summary_by_type_location_severity = summarize_runs(flat_rows, ["scenario", "type", "location", "severity"])

    results["summaries"] = {
        "by_type": summary_by_type,
        "by_type_location": summary_by_type_location,
        "by_type_location_severity": summary_by_type_location_severity,
    }

    # Write outputs.
    json_path = out_dir / "v7_monte_carlo_disruption_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_csv(out_dir / "v7_monte_carlo_disruption_runs.csv", flat_rows)
    write_csv(out_dir / "v7_monte_carlo_summary_by_type.csv", summary_by_type)
    write_csv(out_dir / "v7_monte_carlo_summary_by_type_location.csv", summary_by_type_location)
    write_csv(out_dir / "v7_monte_carlo_summary_by_type_location_severity.csv", summary_by_type_location_severity)

    print("Monte Carlo disruption analysis completed.")
    print(f"Runs per scenario: {MONTE_CARLO_RUNS}")
    print(f"Output folder: {out_dir}")
    print(f"JSON: {json_path}")
    print("CSV files:")
    print(f" - {out_dir / 'v7_monte_carlo_disruption_runs.csv'}")
    print(f" - {out_dir / 'v7_monte_carlo_summary_by_type.csv'}")
    print(f" - {out_dir / 'v7_monte_carlo_summary_by_type_location.csv'}")
    print(f" - {out_dir / 'v7_monte_carlo_summary_by_type_location_severity.csv'}")


if __name__ == "__main__":
    main()
