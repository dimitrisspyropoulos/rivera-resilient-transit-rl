#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Πρακτορας αποκαταστασης - Rivera, Uruguay.

Για καθε διαταραχη που κατεγραψε το disruption.py, ο πρακτορας αναπαραγει τη
ζημια και συνθετει ΑΚΟΛΟΥΘΙΑ παρεμβασεων που επαναφερει οσο περισσοτερη
ζητηση γινεται. Ξεκινα απο το t0 - τη στιγμη της ζημιας, με τις γραμμες
σπασμενες και καμια παρακαμψη ετοιμη - ωστε η ανασυνταξη να ειναι δικη του
δουλεια και να εχει μετρησιμο κοστος.

Δυο σκληροι περιορισμοι, ιδιοι με τον σχεδιασμο:
  καλυψη   το DUN δεν επιτρεπεται να χειροτερεψει, και στοχος ειναι το δαπεδο
           καλυψης - η ζητηση που κανενα συνολο γραμμων δεν μπορει να
           εξυπηρετησει μετα τη ζημια
  δαπανη   κοστος <= max(B, κοστος μετα τη ζημια) * (1 + περιθωριο εκτακτης
           αναγκης). Ο φορεας ΔΕΝ παιρνει επιπλεον χρηματα: ξοδευει την
           εφεδρεια που αφησε αδιαθετη ο σχεδιασμος

Η ανταμοιβη ειναι ακριβως ο στοχος του σχεδιασμου (dCEF + dATT + dCEF2, συν
dE στο S2), κανονικοποιημενη ως προς το αρχικο δικτυο R0. Ετσι σχεδιασμος και
αποκατασταση μετρανε το ιδιο πραγμα και τα νουμερα τους συγκρινονται.

Εισοδος:  best_scenarios_baseline.json απο το v7.py,
          v7_monte_carlo_disruption_results.json απο το disruption.py,
          RiveraTravel.txt, RiveraDemand.txt, Rivera_coords.txt, seekallpaths.py
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import hashlib
import heapq
import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set


# ============================================================
# MANUAL SETTINGS — CHANGE ONLY THESE IF NEEDED
# ============================================================

DATA_DIR = "."

BEST_JSON_FILE = r"C:\Users\user\Τα αρχεία μου\ΔΙΠΛΩΜΑΤΙΚΗ\2026-08-13_00-23-54\best_scenarios_baseline.json"
MONTE_CARLO_RESULTS_FILE = "monte_carlo_disruption_results_v7/v7_monte_carlo_disruption_results.json"
TRAINING_MODULE_FILE = "v7.py"
SEEKALLPATHS_FOLDER = "."

TRAVEL_FILE = "RiveraTravel.txt"
DEMAND_FILE = "RiveraDemand.txt"
COORDS_FILE = "Rivera_coords.txt"

OUT_DIR = "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_results"

EPISODES_PER_SCENARIO = 150
# Το τριτο σεναριο μπαινει ΜΟΝΟ αν υπαρχει στα δεδομενα - αλλιως
# παραλειπεται σιωπηλα (ο ελεγχος γινεται ηδη στη main).
SCENARIOS_TO_RUN = ["S1_service", "S2_energy"]

# ΑΠΟ ΠΟΙΑ ΣΤΙΓΜΗ ΞΕΚΙΝΑΕΙ Η ΑΠΟΚΑΤΑΣΤΑΣΗ
# --------------------------------------------------------------------------
#   "t0" = ΑΜΕΣΗ ΕΙΚΟΝΑ. Ο δρομος κοπηκε, η γραμμη σπασε επιτοπου, κανεις δεν
#          εχει σχεδιασει ακομη παρακαμψη. Ο πρακτορας καλειται να δρασει ΣΤΗ
#          ΖΗΜΙΑ. Εδω το R0 χανει 8.36 % και το S3 3.07 % - υπαρχει πραγματικο
#          εργο αποκαταστασης και ΜΕΤΡΗΣΙΜΟ κοστος.
#   "t1" = ΜΕΤΑ ΤΗΝ ΠΑΡΑΚΑΜΨΗ. Ιδια λογικη με το των κομβων. Ολα τα
#          δικτυα εχουν ηδη συγκλινει στο 1.007 % (το φυσικο δαπεδο) και ο
#          πρακτορας δεν εχει ουσιαστικα τι να αποκαταστησει.
RECOVERY_START = "t0"

# Ο CEF2 ΣΤΗΝ ΑΝΤΑΜΟΙΒΗ ΤΗΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
# --------------------------------------------------------------------------
# ΤΟ ΠΡΟΒΛΗΜΑ: ο σχεδιασμος του S3 μεγιστοποιει τον ΠΕΡΙΟΡΙΣΜΕΝΟ πλεονασμο
# CEF2 (= ελαχιστοποιει την εκτεθειμενη ζητηση), θυσιαζοντας μαλιστα σκοπιμα
# απλο CEF: 1.9381 εναντι 2.1680 του S1. Αν η αποκατασταση ανταμειβει ΜΟΝΟ
# τον απλο CEF, τοτε ο πρακτορας πιεζει το δικτυο ΠΙΣΩ προς αυτο που ο
# σχεδιασμος απεριψε - πυκνωνει τα ηδη πυκνα ζευγη αντι να σωσει τα
# μονοδρομα. Τα δυο σταδια θα τραβουσαν σε ΑΝΤΙΘΕΤΕΣ κατευθυνσεις.
#
# Η ΔΙΟΡΘΩΣΗ: στο S3 προστιθεται ο ορος dCEF2/CEF2_R0, ΑΚΡΙΒΩΣ οπως το [E4]
# των v6/v7. Ετσι σχεδιασμος και αποκατασταση μετρανε ΤΟ ΙΔΙΟ πραγμα.
#   S1: r = dCEF + dATT
#   S2: r = dCEF + dATT + dE
#   S3: r = dCEF + dATT + dCEF2        <-
# Ο ορος dCEF μενει και στο S3 για τον ιδιο λογο που μενει στον σχεδιασμο: ο
# CEF2 ειναι κορεσμενος και μονος του δινει σχεδον παντου μηδενικη κλιση.
RECOVERY_USE_CEF2 = True

# Ο διαχωρισμος 70/30 ΔΕΝ ειναι train/eval - δεν υπαρχει εκπαιδευση να
#      διαχωριστει. Ο μηχανισμος ειναι ντετερμινιστικος oracle και η πολιτικη
#      ειναι ΙΔΙΑ και στις δυο φασεις· γι' αυτο τα mean rewards ταυτιζονται
#      (0.5314 vs 0.5323). Διατηρειται ως HELD-OUT συνολο διαταραχων: η
#      ταυτιση των δυο μεσων γινεται τοτε ΑΠΟΤΕΛΕΣΜΑ («η αποδοση του oracle
#      ειναι σταθερη σε διαταραχες που δεν χρησιμοποιηθηκαν για βαθμονομηση»).
HELD_OUT_RATIO = 0.30
TRAINING_RATIO = 1.0 - HELD_OUT_RATIO      # 0.70 - ιδια αριθμητικη με πριν
RANDOM_SEED = 42

CENTRAL_NODES = list(range(18, 35))  # 18 έως 34

INTERVENTIONS = [
    "increase_frequency_high_demand_routes",
    "short_turn_service_existing_routes",
    "vehicle_reallocation_critical_routes",
    "transfer_synchronization",
    "emergency_reroute_existing_network",
    # /Οι ΤΕΣΣΕΡΙΣ ΝΕΕΣ ενεργειες. Μαζι με τις δυο υπαρχουσες που
    # αγγιζαν ηδη τις γραμμες, το ρεπερτοριο εχει πλεον ΕΞΙ δομικες ενεργειες
    # και τρεις καθαρα λειτουργικες.
    "emergency_bus_bridge",
    "extend_route_to_orphaned_stops",
    "redundant_overlay_line",
    "suspend_low_value_segment",
]

# The recovery agent selects from recovery plans, not only single actions.
# ΤΑ ΟΚΤΩ ΠΑΚΕΤΑ ΑΠΟΚΑΤΑΣΤΑΣΗΣ, ΞΑΝΑΓΡΑΜΜΕΝΑ
# --------------------------------------------------------------------------
# Καθε πακετο εχει πλεον ΤΟΥΛΑΧΙΣΤΟΝ ΜΙΑ ενεργεια που αλλαζει το συνολο
# γραμμων. Στα παλια οκτω, τρια (minimal_transfer_response, frequency_response,
# fleet_reallocation_response) και το demand_pressure_response δεν αλλαζαν
# ουτε μια γραμμη - αρα ΔCEF = 0.000000 εξ ορισμου.
#
#   1 continuity_repair             αποκατασταση συνεχειας γραμμων
#   2 segmented_operation           λειτουργια των επιζωντων τμηματων
#   3 coverage_restoration          επιστροφη των χαμενων σταση
#   4 redundancy_restoration        επιστροφη ΕΝΑΛΛΑΚΤΙΚΩΝ διαδρομων
#   5 budget_neutral_reconfiguration  αναδιαταξη ΧΩΡΙΣ επιπλεον ευρω
#   6 demand_response               αιχμη ζητησης
#   7 corridor_relief               ανακουφιση φορτωμενου αξονα
#   8 full_structural_recovery      ολα μαζι
NEW_RECOVERY_PLANS = {
    "continuity_repair": [
        "emergency_reroute_existing_network",
    ],
    "segmented_operation": [
        "short_turn_service_existing_routes",
        "increase_frequency_high_demand_routes",
    ],
    "coverage_restoration": [
        "emergency_reroute_existing_network",
        "extend_route_to_orphaned_stops",
    ],
    "redundancy_restoration": [
        "emergency_reroute_existing_network",
        "emergency_bus_bridge",
    ],
    "budget_neutral_reconfiguration": [
        "suspend_low_value_segment",
        "emergency_reroute_existing_network",
        "emergency_bus_bridge",
        "extend_route_to_orphaned_stops",
    ],
    "demand_response": [
        "redundant_overlay_line",
        "increase_frequency_high_demand_routes",
        "vehicle_reallocation_critical_routes",
    ],
    "corridor_relief": [
        "redundant_overlay_line",
        "transfer_synchronization",
    ],
    "full_structural_recovery": [
        "suspend_low_value_segment",
        "emergency_reroute_existing_network",
        "emergency_bus_bridge",
        "extend_route_to_orphaned_stops",
        "redundant_overlay_line",
        "increase_frequency_high_demand_routes",
    ],
}

NEW_FEASIBLE_PLANS_BY_TYPE = {
    "link_failure": [
        "continuity_repair",
        "segmented_operation",
        "redundancy_restoration",
        "budget_neutral_reconfiguration",
        "corridor_relief",
        "full_structural_recovery",
    ],
    "demand_surge": [
        "corridor_relief",
        "demand_response",
        "segmented_operation",
    ],
    "node_removal": [
        "continuity_repair",
        "segmented_operation",
        "coverage_restoration",
        "redundancy_restoration",
        "budget_neutral_reconfiguration",
        "full_structural_recovery",
    ],
    "area_closure": [
        "continuity_repair",
        "segmented_operation",
        "coverage_restoration",
        "redundancy_restoration",
        "budget_neutral_reconfiguration",
        "full_structural_recovery",
    ],
    "combined_disruption": [
        "continuity_repair",
        "coverage_restoration",
        "redundancy_restoration",
        "budget_neutral_reconfiguration",
        "corridor_relief",
        "full_structural_recovery",
    ],
}

RECOVERY_PLANS = NEW_RECOVERY_PLANS

FEASIBLE_PLANS_BY_TYPE = NEW_FEASIBLE_PLANS_BY_TYPE

# ΑΦΑΙΡΕΘΗΚΑΝ: EPSILON_START = 0.35, EPSILON_END = 0.05, LEARNING_RATE = 0.20
#      Εμφανιζονταν ΜΟΝΟ μεσα στις τρεις συναρτησεις choose_action / update_q /
#      epsilon_for_episode, οι οποιες (επαληθευση με αναλυση AST ολου του
#      αρχειου) ΔΕΝ ΚΑΛΟΥΝΤΑΙ ΠΟΤΕ. Ο μηχανισμος που τρεχει ειναι ο
#      εξαντλητικος oracle της select_best_positive_plan. Βλ. .

# ==========================================================================
# ΑΝΑΣΧΕΔΙΑΣΜΟΣ ΕΠΙΠΕΔΟΥ 1
# ==========================================================================
# ΣΥΝΑΡΤΗΣΗ ΑΝΤΑΜΟΙΒΗΣ
#   "design_objective" -> r = dCEF/CEF0 + dATT/ATT0 (+ dE/E0 στο S2_energy),
#       δηλαδη ΑΚΡΙΒΩΣ ο στοχος που βελτιστοποιουν τα v6/v7 (§3.2.5) και το
#       recovery_rl.py. Και τα τρια σταδια της εργασιας μετρανε πλεον το ιδιο
#       πραγμα. Το DUN ΔΕΝ μπαινει στην ανταμοιβη: ειναι ΠΕΡΙΟΡΙΣΜΟΣ, οπως
#       ακριβως και στον σχεδιασμο.
#   "legacy_weights"   -> τα παλια αυθαιρετα βαρη, για συνεχεια/συγκριση.
REWARD_MODE = "design_objective"

# ΣΚΛΗΡΟΣ ΠΡΟΥΠΟΛΟΓΙΣΜΟΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
#   "design" -> B = max( C0*(1+p) , κοστος μετα τη ζημια ). Ο φορεας ΔΕΝ
#       παιρνει επιπλεον χρηματα· οπου η ζημια εχει ΗΔΗ ξεπερασει το B (43 %
#       των τοπολογικων διαταραχων, εως +29.8 %), οριο γινεται αυτο που ηδη
#       ξοδευεται, αλλιως καθε σχεδιο θα ηταν αφετηριακα μη εφικτο.
#   "none"   -> καμια δεσμευση (η παλια συμπεριφορα).
RECOVERY_BUDGET_MODE = "design"

# ΠΕΡΙΘΩΡΙΟ ΠΑΝΩ ΑΠΟ ΤΟΝ ΕΓΚΕΚΡΙΜΕΝΟ ΠΡΟΥΠΟΛΟΓΙΣΜΟ.
# Μηδεν: η ικανοτητα αποκαταστασης ΔΕΝ ερχεται απο επιπλεον χρηματα αλλα απο
# την εφεδρεια που ο σχεδιασμος αφησε αδιαθετη (BUDGET_RESERVE στο v7.py). Ετσι
# το δικτυο ανταπεξερχεται στην ακραια διαταραχη χωρις να ανεβασει τον
# προυπολογισμο - που ειναι και το ζητουμενο.
#
# Το οριο παραμενει max(B, κοστος μετα τη ζημια): οπου η ιδια η ζημια εχει ηδη
# ξεπερασει το B, οριο γινεται αυτο που ηδη ξοδευεται, αλλιως καθε σχεδιο θα
# ηταν αφετηριακα μη εφικτο.
RECOVERY_BUDGET_MARGIN = 0.00

# ΣΥΝΘΕΣΗ ΑΚΟΛΟΥΘΙΑΣ ΑΝΤΙ ΓΙΑ ΟΚΤΩ ΣΤΑΘΕΡΑ ΠΑΚΕΤΑ
# --------------------------------------------------------------------------
# ΤΟ ΟΡΙΟ ΤΟΥ ΠΑΛΙΟΥ ΣΧΗΜΑΤΟΣ: ο πρακτορας δοκιμαζε 8 ΠΡΟΚΑΘΟΡΙΣΜΕΝΑ πακετα,
# ΜΙΑ φορα, και κρατουσε το καλυτερο. Δεν μπορουσε να πει «βαλε μια γεφυρα,
# δες τι εγινε, βαλε και δευτερη, μετα επεκτεινε μια γραμμη».
#
# ΤΩΡΑ: ο πρακτορας ΣΥΝΘΕΤΕΙ ακολουθια. Βημα 1: το καλυτερο απο τα οκτω πακετα
# (μακρο-κινηση - λυνει τη μυωπια, γιατι η αποσυρση μονη της χαλαει τον
# πλεονασμο και θα απορριπτοταν ποτε μονη). Βηματα 2..N: η καλυτερη ΠΡΟΣΘΕΤΙΚΗ
# δομικη ενεργεια καθε φορα, παντα υπο τους ιδιους σκληρους περιορισμους.
# Σταματα μολις καμια κινηση δεν βελτιωνει.
#
# Παραμενει ΕΠΙΠΕΔΟ 1: ντετερμινιστικο, χωρις μαθηση, εξηγησιμο σε χειριστη ως
# λιστα ενεργειων. Η διαφορα με το Επιπεδο 2 μενει: εκει γινεται ΑΝΑΖΗΤΗΣΗ
# 800 βηματων πανω στη δομη των γραμμων.
GREEDY_COMPOSITION = True
GREEDY_MAX_STEPS = 5

# Η ΑΝΤΑΜΟΙΒΗ ΜΕΤΡΑΕΙ ΜΟΝΟ ΟΤΙ ΥΠΟΛΟΓΙΖΕΤΑΙ, ΟΧΙ ΟΤΙ ΥΠΟΘΕΤΟΥΜΕ.
# --------------------------------------------------------------------------
# Οι ενεργειες συχνοτητας / στολου / συγχρονισμου δηλωναν μειωση ATT μεσω
# ΣΤΑΘΕΡΩΝ ΠΑΡΑΔΟΧΗΣ (TRANSFER_SYNC_GAIN_MIN = 1.00, FREQUENCY_INCREASE = 0.30)
# που προστιθενται εκ των υστερων στη μετρικη. Στη σχετικη κλιμακα του στοχου
# αυτο ειναι δωρεαν ανταμοιβη 0.35/17.78 = 0.0197 - μεγαλυτερη απο καθε
# μετρημενο δομικο κερδος. Μετρημενο: το transfer_synchronization κερδιζε σε
# 37 απο 80 επεισοδια ΧΩΡΙΣ να αλλαζει ουτε μια γραμμη.
#
# Δεν βελτιστοποιεις ως προς σταθερα που υπεθεσες ο ιδιος. Οι παραδοχες
# ΠΑΡΑΜΕΝΟΥΝ και αναφερονται στα αποτελεσματα (att_delta_min, additional_cost)·
# απλως ΔΕΝ οδηγουν πλεον την επιλογη. Το κοστος, που ειναι ΠΡΑΓΜΑΤΙΚΟ,
# εξακολουθει να μετραει κανονικα στον περιορισμο του προυπολογισμου.
REWARD_USES_ASSUMED_ATT = False
GREEDY_MIN_GAIN = 1e-6

# Αυτες ΞΑΝΑΧΤΙΖΟΥΝ απο τις ΑΡΧΙΚΕΣ γραμμες, αρα σβηνουν οτι εχει προστεθει.
# Επιτρεπονται ΜΟΝΟ στο πρωτο βημα.
REBUILDS_FROM_BASELINE = {
    "emergency_reroute_existing_network",
    "short_turn_service_existing_routes",
}
# Οι ΠΡΟΣΘΕΤΙΚΕΣ δομικες ενεργειες - μπορουν να εφαρμοστουν επαναληπτικα.
ADDITIVE_STRUCTURAL_ACTIONS = [
    "emergency_bus_bridge",
    "redundant_overlay_line",
    "extend_route_to_orphaned_stops",
    "suspend_low_value_segment",
]
# Καθαρα λειτουργικες - δοκιμαζονται ΜΙΑ φορα στο τελος.
OPERATIONAL_ACTIONS = [
    "increase_frequency_high_demand_routes",
    "vehicle_reallocation_critical_routes",
    "transfer_synchronization",
]

# Η ΚΑΛΥΨΗ ΕΙΝΑΙ ΥΠΟΧΡΕΩΣΗ, ΟΧΙ ΕΠΙΛΟΓΗ.
#   ENFORCE_COVERAGE_FLOOR = True  -> η αποκατασταση ΠΡΕΠΕΙ να επαναφερει το
#       DUN στο δαπεδο (τη ζητηση που κανενα συνολο γραμμων δεν εξυπηρετει).
#       Οποιο σχεδιο αφηνει ανεξυπηρετητη ζητηση που ΘΑ ΜΠΟΡΟΥΣΕ να καλυφθει
#       ειναι ΜΗ ΕΦΙΚΤΟ. Ιδιο κριτηριο με το [RL5] του recovery_rl.py.
#   Ο ασθενεστερος κανονας «να μη χειροτερεψει» μενει ως εφεδρικος.
ENFORCE_COVERAGE_FLOOR = True
COVERAGE_FLOOR_TOL = 1e-6
ENFORCE_NO_COVERAGE_LOSS = True

# Ορια ρεαλισμου των νεων ενεργειων
OVERLAY_MAX_LEN = 18               # μηκος προσωρινης παραλληλης γραμμης
OVERLAY_MAX_CANDIDATES = 6         # ποσους αξονες εξεταζει πριν τα παρατησει
# 6 -> 12 ακρα. Η ΑΡΙΘΜΗΤΙΚΗ ΤΟΥ ΠΡΟΥΠΟΛΟΓΙΣΜΟΥ, μετρημενη:
#   κοστος μετα τη ζημια  39 443 EUR  = 102.4 % του B
#   αποσυρση 6 ακρων      -1 829 EUR
#   γεφυρα λεωφορειων     +2 346 EUR
#   -> 39 960 EUR, ΠΑΝΩ απο το οριο: το πακετο ηταν ΠΑΝΤΑ μη εφικτο.
# Με 16 ακρα ελευθερωνονται ~3 900 EUR και η αναδιαταξη χωραει ΚΑΙ μεσα στον
# αρχικο προυπολογισμο σχεδιασμου. Τα 12 ειναι 12 απο τα 26 ακρα γραμμων, ΟΛΑ
# τους εξυπηρετουμενα και απο αλλη γραμμη - καμια απωλεια καλυψης εκ κατασκευης.
SUSPEND_MAX_NODES = 16
SUSPEND_MIN_ROUTE_LEN = 4          # καμια γραμμη δεν πεφτει κατω απο αυτο

# Γεμιζει απο τη main() () με τις τιμες της ΑΡΧΙΚΗΣ λυσης R0.
REFERENCE_R0: Dict[str, Any] = {}

# Reward weights (χρησιμοποιουνται ΜΟΝΟ στο REWARD_MODE = "legacy_weights")
W_ATT = 1.0
W_CEF = 1.0
W_DUN = 2.5
W_COST = 0.15
W_ENERGY = 0.15

# Operational assumptions
C_KM = 0.8
C_H = 18.0
E_KM = 1.2
HEADWAY_MIN = 10.0
OPERATING_HOURS = 12.0
ASSUME_BIDIRECTIONAL = True

# Filled in main() after load_rivera()
TT_MATRIX_GLOBAL = None
DIST_KM_GLOBAL = None

# Frequency interventions are represented through headway / waiting-time changes.
# They do not reduce edge travel times.
# Extra service is reflected in cost/energy through approximate route-level service work.


# ============================================================
# BASIC UTILITIES
# ============================================================

@dataclass
class DisruptionResult:
    routes: List[List[int]]
    TT_dict: Dict[int, Dict[int, float]]
    TD_dict: Dict[int, Dict[int, float]]
    info: Dict[str, Any]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_training_module(module_path: Path, seek_folder: Path):
    if str(seek_folder.resolve()) not in sys.path:
        sys.path.insert(0, str(seek_folder.resolve()))
    if str(module_path.parent.resolve()) not in sys.path:
        sys.path.insert(0, str(module_path.parent.resolve()))

    if not module_path.exists():
        raise FileNotFoundError(f"Training module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("rivera_v7_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def improvement_decrease(before: Optional[float], after: Optional[float]) -> Optional[float]:
    # For metrics where lower is better: ATT, DUN, cost, energy.
    # improvement = (before - after) / before
    b = safe_float(before)
    a = safe_float(after)
    if b is None or a is None or abs(b) < 1e-12:
        return None
    return (b - a) / b


def improvement_increase(before: Optional[float], after: Optional[float]) -> Optional[float]:
    # For metrics where higher is better: CEF.
    # improvement = (after - before) / before
    b = safe_float(before)
    a = safe_float(after)
    if b is None or a is None or abs(b) < 1e-12:
        return None
    return (a - b) / b


def unique_nodes(routes: List[List[int]]) -> List[int]:
    return sorted({int(n) for r in routes for n in r})


def route_edges(route: List[int]) -> List[Tuple[int, int]]:
    return [(int(route[i]), int(route[i + 1])) for i in range(len(route) - 1)]


def clean_routes(routes: List[List[int]]) -> List[List[int]]:
    cleaned = []
    for r in routes:
        compact = []
        for n in r:
            n = int(n)
            if not compact or compact[-1] != n:
                compact.append(n)
        if len(compact) >= 2:
            cleaned.append(compact)
    return cleaned


# ============================================================
# EVALUATION WRAPPER
# ============================================================

# ΤΑΧΥΤΗΤΑ - ΧΩΡΙΣ ΚΑΜΙΑ ΑΛΛΑΓΗ ΑΠΟΤΕΛΕΣΜΑΤΟΣ
# --------------------------------------------------------------------------
# ΤΟ ΠΡΟΒΛΗΜΑ: η evaluate_routes ΑΔΕΙΑΖΕΙ τις caches των v6/v7 σε ΚΑΘΕ κληση,
# επειδη το κλειδι τους ειναι ΜΟΝΟ το routeset και το δικτυο αλλαζει αναμεσα
# σε επεισοδια. Σωστος συλλογισμος, αλλα δρακοντειο φαρμακο: ΜΕΣΑ σε ενα
# επεισοδιο το δικτυο ειναι ΣΤΑΘΕΡΟ και αλλαζουν μονο οι γραμμες, οποτε η
# cache θα ηταν απολυτως εγκυρη - και πεταγεται.
#
# ΜΕΤΡΗΜΕΝΟ (4 επεισοδια, S3): απο τις κλησεις της evaluate_routes το
#   60 % - 67 % ειναι ΑΚΡΙΒΗ ΔΙΠΛΟΤΥΠΑ (ιδιο routeset ΚΑΙ ιδιο δικτυο) που
#   ξαναυπολογιζονται απο το μηδεν, ~478 ms η καθε μια.
#
# Η ΛΥΣΗ: υπολογιζεται ενα φθηνο αποτυπωμα του (TT, TD, σεναριο) και οι caches
# αδειαζουν ΜΟΝΟ οταν αυτο αλλαξει. Ετσι η cache χρησιμοποιειται ΜΟΝΟ οταν το
# δικτυο ειναι bit-προς-bit ιδιο - δηλαδη ακριβως οταν η αποθηκευμενη τιμη
# ΕΙΝΑΙ σωστη. Τα αποτελεσματα ειναι ΠΑΝΟΜΟΙΟΤΥΠΑ· αλλαζει μονο ο χρονος.
_LAST_NET_FINGERPRINT = None


def _network_fingerprint(TT_dict, TD_dict, scenario_name: str) -> bytes:
    """Φθηνο αποτυπωμα του δικτυου (286 ακμες + 378 ζευγη ~ 1 ms)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(scenario_name.encode("utf-8"))
    h.update(b"|TT|")
    for a in sorted(TT_dict):
        row = TT_dict[a]
        for b in sorted(row):
            h.update(f"{a},{b},{row[b]!r};".encode("utf-8"))
    h.update(b"|TD|")
    for a in sorted(TD_dict):
        row = TD_dict[a]
        for b in sorted(row):
            h.update(f"{a},{b},{row[b]!r};".encode("utf-8"))
    return h.digest()


def evaluate_routes(
    mod: Any,
    routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    assumptions: Dict[str, Any],
    scenario_name: str = "S2_energy",
) -> Dict[str, Any]:
    """
    Correct evaluator for rivera_rl_v7_sna_actions_fixed_bridge.py.

    The v7 script computes ATT, CEF and DUN through:
        Solution(...)
        evaluate_solution(sol, TT_dict, TD_dict, tt, dist_km, ...)

    Therefore this wrapper builds the needed Solution object and calls the
    original v7 evaluation function directly.
    """
    routes = clean_routes(routes)

    # Use global matrices loaded in main()
    global TT_MATRIX_GLOBAL
    global DIST_KM_GLOBAL

    if TT_MATRIX_GLOBAL is None:
        raise RuntimeError("TT_MATRIX_GLOBAL is None. load_rivera() output tt was not stored.")
    if DIST_KM_GLOBAL is None:
        raise RuntimeError("DIST_KM_GLOBAL is None. dist_km was not built from coordinates.")

    # Clear v7 caches because recovery/disruption can modify TT_dict or TD_dict
    # while keeping the same routeset. The original v7 cache key is routeset-based,
    # so without clearing it, changed travel times/demand may be ignored.
    # ...αλλα ΜΟΝΟ οταν το δικτυο ΟΝΤΩΣ αλλαξε. Οσο μενει ιδιο, η cache
    # ειναι εγκυρη και κραταει το 60-67 % των κλησεων που ειναι διπλοτυπα.
    global _LAST_NET_FINGERPRINT
    _fp = _network_fingerprint(TT_dict, TD_dict, scenario_name)
    if _fp != _LAST_NET_FINGERPRINT:
        for cache_name in ["ROUTE_METRIC_CACHE", "ATTACK_EVAL_CACHE", "SOLUTION_EVAL_CACHE"]:
            cache_obj = getattr(mod, cache_name, None)
            if hasattr(cache_obj, "clear"):
                cache_obj.clear()
        _LAST_NET_FINGERPRINT = _fp

    # Minimal action list only needed to instantiate Solution
    dummy_actions = ["do_nothing"]

    sol = mod.Solution(copy.deepcopy(routes), dummy_actions)

    use_energy = (scenario_name == "S2_energy")
    compute_energy_flag = (scenario_name == "S2_energy")

    att, cef, dun = mod.evaluate_solution(
        sol,
        TT_dict,
        TD_dict,
        TT_MATRIX_GLOBAL,
        DIST_KM_GLOBAL,
        assumptions["c_km"],
        assumptions["c_h"],
        assumptions["e_km"],
        operating_hours=assumptions["operating_hours"],
        headway_min=assumptions["headway_min"],
        assume_bidirectional=assumptions["assume_bidirectional"],
        use_energy_in_state=use_energy,
        compute_energy=compute_energy_flag,
    )

    return {
        "att": safe_float(att),
        "CEF": safe_float(cef),
        # Ο evaluate_solution των v6/v7 γεμιζει ηδη τα sol.CEF2 /
        # sol.exposure (ετικετα [E3b]) - εδω απλως διαβαζονται. ΚΑΘΑΡΗ ΜΕΤΡΗΣΗ:
        # αν το φορτωμενο module ειναι ΑΔΙΟΡΘΩΤΟ, βγαινουν None και η
        # συμπεριφορα μενει ΑΚΡΙΒΩΣ η σημερινη.
        "CEF2": safe_float(getattr(sol, "CEF2", None)),
        "exposure": safe_float(getattr(sol, "exposure", None)),
        "dun": safe_float(dun),
        "cost": safe_float(getattr(sol, "cost", None)),
        "energy": safe_float(getattr(sol, "energy_kwh", None)) if scenario_name == "S2_energy" else None,
        "vkm": safe_float(getattr(sol, "vkm", None)),
        "vh": safe_float(getattr(sol, "vh", None)),
        "num_routes": len(routes),
        "num_unique_nodes": len(unique_nodes(routes)),
        "routes": routes,
    }


# ============================================================
# REPLAY EXISTING MONTE CARLO DISRUPTIONS
# ============================================================

def shortest_path_avoiding(
    TT_dict: Dict[int, Dict[int, float]],
    blocked: Set[int],
    a: int,
    b: int,
) -> Optional[List[int]]:
    """Dijkstra απο a σε b στον γραφο ΧΩΡΙΣ τους κομβους `blocked`."""
    if a in blocked or b in blocked:
        return None
    if a == b:
        return [a]
    dist = {a: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, a)]
    seen: Set[int] = set()
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


def count_invalid_edges(routes: List[List[int]], TT_dict: Dict[int, Dict[int, float]]) -> int:
    """Ακμες γραμμων που ΔΕΝ υπαρχουν στον γραφο. Πρεπει να ειναι 0."""
    bad = 0
    for r in routes:
        for k in range(len(r) - 1):
            if int(r[k + 1]) not in TT_dict.get(int(r[k]), {}):
                bad += 1
    return bad


def remove_nodes_from_routes(
    routes: List[List[int]],
    nodes: List[int],
    TT_dict: Optional[Dict[int, Dict[int, float]]] = None,
    max_len: int = 40,
) -> List[List[int]]:
    """ΙΔΙΑ ΔΙΟΡΘΩΣΗ με το [D1] του disruption.py.

    Ο παλιος κωδικας εβγαζε απλως τον κομβο απο τη λιστα: απο [1, 2, 3, 7, ...]
    με κλειστο τον 2 επαιρνε [1, 3, 7, ...] - αλλα η ακμη 1->3 ΔΕΝ ΥΠΑΡΧΕΙ.
    Χειροτερα, το get_route_length_time εκανε fallback σε shortestPath(1,3) =
    [1, 2, 3], δηλαδη δρομολογουσε το λεωφορειο ΜΕΣΑ ΑΠΟ τον κατεστραμμενο
    κομβο: μηκος πριν 16.203 km, μετα 16.203 km - ΜΗΔΕΝΙΚΗ επιδραση στο κοστος.

    Τωρα καθε ζευγος διαδοχικων επιζωντων κομβων επανασυνδεεται με ΠΡΑΓΜΑΤΙΚΟ
    συντομοτερο μονοπατι στον γραφο ΧΩΡΙΣ τους κλειστους κομβους· αν δεν υπαρχει
    παρακαμψη, η γραμμη ΣΠΑΕΙ σε δυο ανεξαρτητα τμηματα.

    ΑΥΤΟ ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ ΤΑΥΤΟΣΗΜΟ με το disruption.py, αλλιως τα δυο σκριπτ
    μετρανε ΔΙΑΦΟΡΕΤΙΚΟ δικτυο για την ιδια διαταραχη.

    Οταν TT_dict is None η συμπεριφορα ειναι η ΠΑΛΙΑ (συμβατοτητα)."""
    rem = {int(x) for x in nodes}

    if TT_dict is None:
        return clean_routes([[int(n) for n in r if int(n) not in rem] for r in routes])

    out: List[List[int]] = []
    for r in routes:
        cleaned: List[int] = []
        for n in (int(x) for x in r):
            if n in rem:
                continue
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
            detour = shortest_path_avoiding(TT_dict, rem, a, b)
            if detour is not None and len(seg) + len(detour) - 1 <= max_len:
                seg.extend(detour[1:])
            else:
                if len(seg) >= 2:
                    out.append(seg)
                seg = [b]
        if len(seg) >= 2:
            out.append(seg)

    # dedup (και ως προς την αντιστροφη φορα), οπως το [D4]
    seen_r = set()
    ded: List[List[int]] = []
    for r in clean_routes(out):
        k1, k2 = tuple(r), tuple(reversed(r))
        if k1 in seen_r or k2 in seen_r:
            continue
        seen_r.add(k1)
        ded.append(r)
    return ded


def multiply_edge_tt(TT_dict: Dict[int, Dict[int, float]], edge: Tuple[int, int], factor: float) -> None:
    u, v = int(edge[0]), int(edge[1])
    try:
        if u in TT_dict and v in TT_dict[u]:
            TT_dict[u][v] = float(TT_dict[u][v]) * factor
    except Exception:
        pass
    try:
        if v in TT_dict and u in TT_dict[v]:
            TT_dict[v][u] = float(TT_dict[v][u]) * factor
    except Exception:
        pass


def get_target_nodes_for_demand_surge(routes: List[List[int]], info: Dict[str, Any]) -> List[int]:
    location = info.get("location") or info.get("target_location")
    nodes = unique_nodes(routes)
    central = set(CENTRAL_NODES)

    if location == "central_area":
        return [n for n in nodes if n in central]
    if location == "peripheral_area":
        return [n for n in nodes if n not in central]
    return nodes


def edges_lost_from_details(details: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Ολες οι ακμες που κοπηκαν - και μεσα σε combined_disruption.
    ΤΑΥΤΟΣΗΜΗ με την edges_lost_from_details του disruption.py [D18]."""
    out: List[Tuple[int, int]] = []
    for comp in (details.get("components") or [details]):
        for e in (comp.get("failed_edges") or []):
            out.append((int(e[0]), int(e[1])))
    return out


def split_routes_at_closures(routes: List[List[int]],
                            closed: Set[int],
                            TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """t0: η γραμμη σταματα εκει που βρισκει τη ζημια - κλειστο κομβο η
    κομμενη ακμη. Ταυτοσημη με τη split_routes_at_closures του disruption.py,
    ωστε η αρχικη κατασταση της αποκαταστασης να αναπαραγει ΑΚΡΙΒΩΣ το
    metrics που εχει ηδη καταγραψει η αναλυση διαταραχων."""
    out: List[List[int]] = []
    for r in routes:
        r = [int(n) for n in r]
        cur: List[int] = []
        for n in r:
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
                                 closed: Set[int],
                                 TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """t1: παρακαμψη μεσω πραγματικου συντομοτερου μονοπατιου. Εναλλακτικη
    αφετηρια, οταν RECOVERY_START = "t1"."""
    base = remove_nodes_from_routes(routes, sorted(closed), TT_dict) if closed else routes
    return repair_routes_after_edge_loss(base, TT_dict)


def split_routes_at_missing_edges(routes: List[List[int]],
                                  TT_dict: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """t0 - η γραμμη σπαει ΕΠΙΤΟΠΟΥ. Αντιγραφο του [D16] του
    disruption.py, ΧΩΡΙΣ dedup, ωστε η αρχικη κατασταση της αποκαταστασης να
    αναπαραγει ΑΚΡΙΒΩΣ το metrics_immediate που εχει ηδη καταγραψει το
    disruption.py."""
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
    """t1 - παρακαμψη μεσω ΠΡΑΓΜΑΤΙΚΟΥ συντομοτερου μονοπατιου στο
    υπολειπομενο οδικο δικτυο. Αντιγραφο του [D17] του disruption.py."""
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


def replay_component(
    routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    info: Dict[str, Any],
) -> Tuple[List[List[int]], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:

    dtype = info.get("type")

    new_routes = copy.deepcopy(routes)
    new_TT = copy.deepcopy(TT_dict)
    new_TD = copy.deepcopy(TD_dict)

    # Οι κλειστοι κομβοι δεν αφαιρουνται απο τις γραμμες εδω: το σπασιμο
    # γινεται κεντρικα, ιδια για καθε τυπο διαταραχης, οπως στο disruption.py.
    if dtype in ("node_removal", "area_closure"):
        return new_routes, new_TT, new_TD

    if dtype == "demand_surge":
        factor = float(info.get("factor", 1.0))
        target_nodes = set(get_target_nodes_for_demand_surge(new_routes, info))
        for i in list(new_TD.keys()):
            for j in list(new_TD[i].keys()):
                try:
                    jj = int(j)
                except Exception:
                    jj = j
                if jj in target_nodes:
                    new_TD[i][j] = float(new_TD[i][j]) * factor
        return new_routes, new_TT, new_TD

    if dtype == "link_failure":
        # ΤΟ ΣΦΑΛΜΑ ΠΟΥ ΕΣΚΑΓΕ: ο τυπος προστεθηκε στο disruption.py [D14]
        # αλλα ΟΧΙ εδω. Εδω αφαιρουνται ΜΟΝΟ οι ακμες απο το οδικο δικτυο· το τι
        # γινονται οι γραμμες αποφασιζεται ΜΙΑ ΦΟΡΑ στην replay_saved_disruption,
        # ΜΕΤΑ απο ολα τα components - ακριβως οπως στο [D18c] του disruption.py.
        for e in (info.get("failed_edges") or []):
            u, v = int(e[0]), int(e[1])
            new_TT.get(u, {}).pop(v, None)
            new_TT.get(v, {}).pop(u, None)
        return new_routes, new_TT, new_TD

    raise ValueError(f"Unknown disruption component type: {dtype}")


COORDS_GLOBAL: Dict[int, Tuple[float, float]] = {}

# Ιδιες παραμετρους με το [D10] του disruption.py. Οταν το JSON της
# διαταραχης φερει το δικο του walk_access, χρησιμοποιουνται ΕΚΕΙΝΕΣ.
WALK_ACCESS_ENABLED = True
WALK_ACCESS_RADIUS_M = 400.0
WALK_SPEED_KMH = 4.8


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _walk_params(info: Dict[str, Any]) -> Tuple[bool, float, float]:
    w = info.get("walk_access") or {}
    if w:
        return (bool(w.get("enabled", WALK_ACCESS_ENABLED)),
                float(w.get("radius_m", WALK_ACCESS_RADIUS_M)),
                float(w.get("speed_kmh", WALK_SPEED_KMH)))
    return WALK_ACCESS_ENABLED, WALK_ACCESS_RADIUS_M, WALK_SPEED_KMH


def apply_walk_access_local(TD_dict, closed: Set[int], info: Dict[str, Any]):
    """ΙΔΙΑ ανακατανομη ζητησης με το [D11] του disruption.py.

    Χωρις αυτο, το Επιπεδο 1 θα δουλευε σε ΑΛΛΗ ζητηση απο εκεινη που παρηγαγε
    το disruption.py: η ζητηση των κλειστων κομβων θα εμενε εκει, θα μετριοταν
    ως μονιμα ανεξυπηρετητη, και καμια αποκατασταση δεν θα μπορουσε ποτε να
    πιασει το δαπεδο."""
    enabled, radius, speed = _walk_params(info)
    out = {"reassigned_nodes": 0, "stranded_nodes": 0, "walk_pax_minutes": 0.0,
           "radius_m": radius, "speed_kmh": speed, "mapping": {}}
    if (not enabled) or (not closed) or (not COORDS_GLOBAL):
        return TD_dict, out, set(closed)

    open_nodes = [n for n in COORDS_GLOBAL if n not in closed]
    m_per_min = float(speed) * 1000.0 / 60.0
    target: Dict[int, int] = {}
    wmin: Dict[int, float] = {}
    for k in sorted(closed):
        if k not in COORDS_GLOBAL:
            continue
        la1, lo1 = COORDS_GLOBAL[k]
        best, bd = None, None
        for j in open_nodes:
            la2, lo2 = COORDS_GLOBAL[j]
            dd = _haversine_m(la1, lo1, la2, lo2)
            if dd <= radius and (bd is None or dd < bd):
                best, bd = j, dd
        if best is not None:
            target[k] = best
            wmin[k] = bd / m_per_min
    out["reassigned_nodes"] = len(target)
    out["stranded_nodes"] = len(closed) - len(target)
    out["mapping"] = {str(k): [int(v), round(wmin[k], 3)] for k, v in target.items()}

    TD_new: Dict[int, Dict[int, float]] = {}
    for o, row in (TD_dict or {}).items():
        for d, dem in row.items():
            try:
                dem = float(dem)
            except Exception:
                continue
            if dem <= 0:
                continue
            o, d = int(o), int(d)
            o2, d2 = target.get(o, o), target.get(d, d)
            extra = (wmin.get(o, 0.0) if o in target else 0.0) + (wmin.get(d, 0.0) if d in target else 0.0)
            if extra > 0:
                out["walk_pax_minutes"] += dem * extra
            if o2 == d2:
                continue
            TD_new.setdefault(o2, {})
            TD_new[o2][d2] = TD_new[o2].get(d2, 0.0) + dem
    stranded = {k for k in closed if k not in target}
    return TD_new, out, stranded


def compute_coverage_floor(TT_dict, TD_dict, closed: Set[int], stranded: Set[int]) -> float:
    """ΤΟ ΔΑΠΕΔΟ ΚΑΛΥΨΗΣ (%): η ζητηση που ΚΑΝΕΝΑ συνολο γραμμων δεν
    μπορει να εξυπηρετησει.

      (α) ζητηση που εμεινε σε κλειστο κομβο χωρις περπατησιμο ανοιχτο γειτονα
      (β) ζητηση μεταξυ ανοιχτων κομβων που ΔΕΝ συνδεονται πια με δρομο

    Ειναι κατω φραγμα που ΑΠΟΔΕΙΚΝΥΕΤΑΙ, οχι εκτιμηση. Η αποκατασταση κρινεται
    ως προς αυτο: οτιδηποτε πανω απο το δαπεδο ειναι ζητηση που ΘΑ ΜΠΟΡΟΥΣΕ να
    εξυπηρετηθει και δεν εξυπηρετηθηκε."""
    total = 0.0
    lost = 0.0
    adj = {i: {j for j in row if j not in closed} for i, row in TT_dict.items() if i not in closed}

    reach_cache: Dict[int, Set[int]] = {}

    def reach(s: int) -> Set[int]:
        if s in reach_cache:
            return reach_cache[s]
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):  # type: ignore[arg-type]
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        reach_cache[s] = seen
        return seen

    for o, row in (TD_dict or {}).items():
        for d, dem in row.items():
            try:
                dem = float(dem)
            except Exception:
                continue
            if dem <= 0:
                continue
            total += dem
            o, d = int(o), int(d)
            if o in closed or d in closed:
                lost += dem
            elif d not in reach(o):
                lost += dem
    return (100.0 * lost / total) if total > 0 else 0.0


def replay_saved_disruption(
    baseline_routes: List[List[int]],
    TT_dict: Dict[int, Dict[int, float]],
    TD_dict: Dict[int, Dict[int, float]],
    disruption_info: Dict[str, Any],
) -> DisruptionResult:

    dtype = disruption_info.get("type")

    routes = copy.deepcopy(baseline_routes)
    tt = copy.deepcopy(TT_dict)
    td = copy.deepcopy(TD_dict)

    if dtype == "combined_disruption":
        for comp in disruption_info.get("components", []):
            routes, tt, td = replay_component(routes, tt, td, comp)
    else:
        routes, tt, td = replay_component(routes, tt, td, disruption_info)

    # Αφου εχουν εφαρμοστει ΟΛΑ τα components, οι γραμμες που εχασαν
    # ακμη σπανε (t0) η παρακαμπτουν (t1). Γινεται ΜΙΑ φορα, στο τελος, ωστε
    # ενα combined_disruption με link_failure + node_removal να δωσει το ιδιο
    # αποτελεσμα με το disruption.py, ανεξαρτητα απο τη σειρα των components.
    info_out = copy.deepcopy(disruption_info)
    closed = affected_nodes_from_disruption(disruption_info)

    # Ο πρακτορας καλειται τη ΣΤΙΓΜΗ της ζημιας: η γραμμη σταματα εκει που
    # βρισκει κλειστο κομβο η κομμενη ακμη και κανεις δεν εχει σχεδιασει ακομη
    # παρακαμψη. Η παρακαμψη ειναι δουλεια ΤΟΥ ΠΡΑΚΤΟΡΑ και πληρωνεται απο τον
    # προυπολογισμο του - αλλιως η βαρια δουλεια θα γινοταν δωρεαν και το
    # κοστος αποκαταστασης δεν θα σημαινε τιποτα. Ταυτοσημο με το t0 του
    # disruption.py.
    if RECOVERY_START == "t0":
        routes = split_routes_at_closures(routes, closed, tt)
    else:
        routes = repair_routes_after_closures(routes, closed, tt)

    td, walk_info, stranded = apply_walk_access_local(td, closed, disruption_info)
    info_out["walk_access_replay"] = walk_info
    # Το δαπεδο ερχεται ετοιμο απο το disruption.py· υπολογιζεται μονο αν λειπει.
    _floor = disruption_info.get("coverage_floor_percent")
    info_out["coverage_floor_percent"] = (
        float(_floor) if _floor is not None
        else compute_coverage_floor(tt, td, closed, stranded))
    info_out["walk_pax_minutes"] = walk_info.get("walk_pax_minutes", 0.0)

    return DisruptionResult(routes=routes, TT_dict=tt, TD_dict=td, info=info_out)


# ============================================================
# RECOVERY INTERVENTIONS — 4 NEW OPERATIONAL ACTIONS ONLY
# ============================================================

FREQUENCY_INCREASE = 0.30          # +30% frequency -> headway 10.00 -> 7.69 min
HIGH_DEMAND_ROUTE_SHARE = 0.35     # top 35% demand routes receive frequency boost
REALLOCATION_INCREASE = 0.30       # +30% frequency on critical routes
REALLOCATION_REDUCTION = 0.15      # -15% frequency on donor low-demand routes
# ΝΕΕΣ ΛΕΙΤΟΥΡΓΙΚΕΣ ΕΝΕΡΓΕΙΕΣ - ορια ρεαλισμου
# --------------------------------------------------------------------------
# Γεφυρα λεωφορειων (bus bridging): προσωρινη γραμμη που ενωνει τις δυο οχθες
# της κλειστης περιοχης, με οχηματα που ελευθερωνονται απο τα ανασταλμενα
# τμηματα. Τρεχει σε ΕΝΑΛΛΑΚΤΙΚΟ διαδρομο - οχι πανω στον ιδιο δρομο που ηδη
# εξυπηρετειται - ωστε να προσθετει ΔΙΑΦΟΡΕΤΙΚΟ μονοπατι, δηλαδη πλεονασμο.
BUS_BRIDGE_MAX_LINES = 3           # το πολυ 3 προσωρινες γραμμες ανα επεισοδιο
# 10 -> 14. Μετρημενο: με οριο 10 κομβους, ΕΝΑΛΛΑΚΤΙΚΟΣ διαδρομος βρισκοταν
# μονο στο 1 απο τα 3 ζευγη οχθων· τα υπολοιπα επεφταν πισω στον ιδιο δρομο, που
# ειναι διπλοτυπο μονοπατι και ΔΕΝ προσθετει πλεονασμο. Γραμμη 14 σταση ειναι
# περιπου 25-30 λεπτα οδηγησης - απολυτως ρεαλιστικη για εκτακτο δρομολογιο.
BUS_BRIDGE_MAX_LEN = 14
# Επεκταση σε ορφανες σταση: κομβοι με ζητηση που εμειναν ΧΩΡΙΣ καμια γραμμη.
ORPHAN_EXTENSION_MAX = 5           # το πολυ 5 ορφανοι κομβοι ανα επεισοδιο
ORPHAN_EXTENSION_MAX_ROUTE_LEN = 25

TRANSFER_SYNC_GAIN_MIN = 1.00      # max average transfer waiting-time gain in minutes
TRANSFER_SYNC_SHARE = 0.35         # assumed OD share affected by synchronization


# Αφαιρεθηκε η classify_routes: οριζοταν και δεν καλουνταν ποτε.


def affected_nodes_from_disruption(info: Dict[str, Any]) -> Set[int]:
    nodes = set()
    nodes.update(int(x) for x in info.get("removed_nodes", []))
    nodes.update(int(x) for x in info.get("closed_nodes", []))

    if info.get("type") == "combined_disruption":
        for comp in info.get("components", []):
            nodes.update(int(x) for x in comp.get("removed_nodes", []))
            nodes.update(int(x) for x in comp.get("closed_nodes", []))

    return nodes


def affected_edges_from_disruption(info: Dict[str, Any]) -> Set[Tuple[int, int]]:
    edges = set()

    for e in info.get("affected_edges", []):
        if len(e) == 2:
            u, v = int(e[0]), int(e[1])
            edges.add((u, v))
            edges.add((v, u))

    if info.get("type") == "combined_disruption":
        for comp in info.get("components", []):
            for e in comp.get("affected_edges", []):
                if len(e) == 2:
                    u, v = int(e[0]), int(e[1])
                    edges.add((u, v))
                    edges.add((v, u))

    return edges


def affected_route_indices(routes: List[List[int]], disruption_info: Dict[str, Any]) -> List[int]:
    nodes = affected_nodes_from_disruption(disruption_info)
    edges = affected_edges_from_disruption(disruption_info)

    affected = []
    for idx, r in enumerate(routes):
        route_node_set = set(int(n) for n in r)
        route_edge_set = set(route_edges(r))
        route_edge_set |= {(v, u) for u, v in route_edge_set}

        if nodes and route_node_set.intersection(nodes):
            affected.append(idx)
            continue
        if edges and route_edge_set.intersection(edges):
            affected.append(idx)
            continue

    return affected


def intervention_do_nothing(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    # Kept only as an internal fallback. It is NOT part of the action space.
    return DisruptionResult(
        routes=copy.deepcopy(disrupted.routes),
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={"intervention": "internal_no_effect_fallback"},
    )


def node_demand(TD_dict: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    demand = defaultdict(float)
    for i, row in TD_dict.items():
        try:
            ii = int(i)
        except Exception:
            ii = i
        for j, val in row.items():
            try:
                jj = int(j)
                f = float(val)
            except Exception:
                continue
            demand[ii] += f
            demand[jj] += f
    return demand


def route_demand_scores(routes: List[List[int]], TD_dict: Dict[int, Dict[int, float]]) -> List[float]:
    nd = node_demand(TD_dict)
    scores = []
    for r in routes:
        nodes = set(int(n) for n in r)
        scores.append(sum(nd.get(n, 0.0) for n in nodes))
    return scores


def top_indices_by_score(scores: List[float], share: float, minimum: int = 1, exclude: Optional[Set[int]] = None) -> List[int]:
    exclude = exclude or set()
    candidates = [(i, s) for i, s in enumerate(scores) if i not in exclude]
    if not candidates:
        return []
    k = max(minimum, int(math.ceil(len(scores) * share)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in candidates[:k]]


def bottom_indices_by_score(scores: List[float], count: int, exclude: Optional[Set[int]] = None) -> List[int]:
    exclude = exclude or set()
    candidates = [(i, s) for i, s in enumerate(scores) if i not in exclude]
    candidates.sort(key=lambda x: x[1])
    return [i for i, _ in candidates[:max(0, count)]]


def route_service_work(
    routes: List[List[int]],
    route_indices: List[int],
    TT_dict: Dict[int, Dict[int, float]],
    headway_min: float,
) -> Tuple[float, float]:
    """Returns approximate daily VKM and VH for selected existing routes."""
    if not route_indices or headway_min <= 0:
        return 0.0, 0.0

    global DIST_KM_GLOBAL
    trips_per_hour = 60.0 / headway_min
    daily_trips = trips_per_hour * OPERATING_HOURS
    if ASSUME_BIDIRECTIONAL:
        daily_trips *= 2.0

    vkm = 0.0
    vh = 0.0
    for ridx in route_indices:
        if ridx < 0 or ridx >= len(routes):
            continue
        route_dist = 0.0
        route_tt_min = 0.0
        for u, v in route_edges(routes[ridx]):
            try:
                if DIST_KM_GLOBAL is not None and u in DIST_KM_GLOBAL and v in DIST_KM_GLOBAL[u]:
                    route_dist += float(DIST_KM_GLOBAL[u][v])
            except Exception:
                pass
            try:
                if u in TT_dict and v in TT_dict[u]:
                    route_tt_min += float(TT_dict[u][v])
            except Exception:
                pass
        vkm += route_dist * daily_trips
        vh += (route_tt_min / 60.0) * daily_trips
    return vkm, vh


def selected_demand_share(scores: List[float], selected: List[int]) -> float:
    total = sum(s for s in scores if s > 0)
    if total <= 1e-12 or not selected:
        return 0.0
    return max(0.0, min(1.0, sum(scores[i] for i in selected if 0 <= i < len(scores)) / total))


def frequency_waiting_delta(headway_min: float, frequency_increase: float, demand_share: float) -> float:
    """
    Negative value means ATT reduction. Expected waiting time = headway / 2.
    The effect is weighted by the demand share served by the selected routes.
    """
    if headway_min <= 0 or frequency_increase <= -0.99:
        return 0.0
    old_wait = headway_min / 2.0
    new_wait = (headway_min / (1.0 + frequency_increase)) / 2.0
    return (new_wait - old_wait) * max(0.0, min(1.0, demand_share))


def intervention_increase_frequency_high_demand_routes(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """
    Increases frequency by 30% on the existing routes with the highest post-disruption demand.
    No nodes, links or spatial paths are added. The ATT effect is applied later as a
    waiting-time adjustment, not as an artificial edge travel-time reduction.
    """
    routes = copy.deepcopy(disrupted.routes)
    scores = route_demand_scores(routes, disrupted.TD_dict)
    selected = top_indices_by_score(scores, HIGH_DEMAND_ROUTE_SHARE, minimum=1)
    share = selected_demand_share(scores, selected)

    vkm, vh = route_service_work(routes, selected, disrupted.TT_dict, HEADWAY_MIN)
    add_vkm = FREQUENCY_INCREASE * vkm
    add_vh = FREQUENCY_INCREASE * vh

    att_delta = frequency_waiting_delta(HEADWAY_MIN, FREQUENCY_INCREASE, share)

    return DisruptionResult(
        routes=routes,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "increase_frequency_high_demand_routes",
            "selected_routes": selected,
            "selected_routes_count": len(selected),
            "demand_share": share,
            "frequency_increase": FREQUENCY_INCREASE,
            "old_headway_min": HEADWAY_MIN,
            "new_headway_min": HEADWAY_MIN / (1.0 + FREQUENCY_INCREASE),
            "att_delta_min": att_delta,
            "additional_vkm": add_vkm,
            "additional_vh": add_vh,
            "additional_cost": C_KM * add_vkm + C_H * add_vh,
            "additional_energy": E_KM * add_vkm,
            "uses_only_existing_nodes_and_routes": True,
        },
    )


def split_route_into_available_segments(route: List[int], forbidden: Set[int]) -> List[List[int]]:
    segments = []
    current = []
    for n in route:
        n = int(n)
        if n in forbidden:
            if len(current) >= 2:
                segments.append(current)
            current = []
        else:
            current.append(n)
    if len(current) >= 2:
        segments.append(current)
    return segments


def intervention_short_turn_service_existing_routes(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """
    Keeps operating the available parts of the same existing routes when nodes/areas are closed.
    This creates short-turn services on already existing route segments only. It does not add
    any node, link, or new spatial corridor.
    """
    forbidden = affected_nodes_from_disruption(disrupted.info)
    if not forbidden:
        return DisruptionResult(
            routes=copy.deepcopy(disrupted.routes),
            TT_dict=copy.deepcopy(disrupted.TT_dict),
            TD_dict=copy.deepcopy(disrupted.TD_dict),
            info={
                "intervention": "short_turn_service_existing_routes",
                "short_turn_routes_count": 0,
                "changed_routes": 0,
                "att_delta_min": 0.0,
                "additional_cost": 0.0,
                "additional_energy": 0.0,
                "uses_only_existing_route_segments": True,
            },
        )

    new_routes = []
    changed_routes = 0
    short_turn_segments = 0
    for r in baseline_routes:
        r = [int(n) for n in r]
        if any(n in forbidden for n in r):
            segments = split_route_into_available_segments(r, forbidden)
            if segments:
                new_routes.extend(segments)
                short_turn_segments += len(segments)
                changed_routes += 1
        else:
            if len(r) >= 2:
                new_routes.append(r)

    new_routes = clean_routes(new_routes)
    if not new_routes:
        new_routes = copy.deepcopy(disrupted.routes)

    # Small operational penalty for terminal turn-back management. This keeps the action realistic.
    extra_cost = 0.01 * (safe_float(disrupted.info.get("closed_nodes_count")) or len(forbidden)) * (safe_float(0.0) or 1.0)

    return DisruptionResult(
        routes=new_routes,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "short_turn_service_existing_routes",
            "forbidden_nodes": sorted(forbidden),
            "short_turn_routes_count": short_turn_segments,
            "changed_routes": changed_routes,
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_route_segments": True,
        },
    )


def intervention_vehicle_reallocation_critical_routes(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """
    Reallocates available vehicles from low-demand routes to critical/high-demand routes.
    Spatial layout remains fixed. The intervention is approximately fleet-neutral: added
    frequency on critical routes is partly offset by reduced frequency on donor routes.
    """
    routes = copy.deepcopy(disrupted.routes)
    scores = route_demand_scores(routes, disrupted.TD_dict)
    affected = [i for i in affected_route_indices(baseline_routes, disrupted.info) if i < len(routes)]

    high_demand = top_indices_by_score(scores, HIGH_DEMAND_ROUTE_SHARE, minimum=1)
    critical = sorted(set(affected + high_demand))
    if not critical:
        critical = high_demand

    donors = bottom_indices_by_score(scores, count=len(critical), exclude=set(critical))

    critical_share = selected_demand_share(scores, critical)
    donor_share = selected_demand_share(scores, donors)

    att_gain = frequency_waiting_delta(HEADWAY_MIN, REALLOCATION_INCREASE, critical_share)
    # Donor routes lose some frequency, so their waiting time increases slightly.
    donor_old_wait = HEADWAY_MIN / 2.0
    donor_new_wait = (HEADWAY_MIN / max(0.05, 1.0 - REALLOCATION_REDUCTION)) / 2.0
    att_loss = (donor_new_wait - donor_old_wait) * donor_share
    att_delta = att_gain + att_loss

    vkm_crit, vh_crit = route_service_work(routes, critical, disrupted.TT_dict, HEADWAY_MIN)
    vkm_don, vh_don = route_service_work(routes, donors, disrupted.TT_dict, HEADWAY_MIN)
    net_vkm = REALLOCATION_INCREASE * vkm_crit - REALLOCATION_REDUCTION * vkm_don
    net_vh = REALLOCATION_INCREASE * vh_crit - REALLOCATION_REDUCTION * vh_don

    return DisruptionResult(
        routes=routes,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "vehicle_reallocation_critical_routes",
            "critical_routes": critical,
            "critical_routes_count": len(critical),
            "donor_routes": donors,
            "donor_routes_count": len(donors),
            "critical_demand_share": critical_share,
            "donor_demand_share": donor_share,
            "frequency_increase_critical": REALLOCATION_INCREASE,
            "frequency_reduction_donor": REALLOCATION_REDUCTION,
            "att_delta_min": att_delta,
            "additional_vkm": net_vkm,
            "additional_vh": net_vh,
            "additional_cost": C_KM * net_vkm + C_H * net_vh,
            "additional_energy": E_KM * net_vkm,
            "uses_only_existing_nodes_and_routes": True,
        },
    )


def transfer_hub_nodes(routes: List[List[int]]) -> Set[int]:
    counts = defaultdict(int)
    for r in routes:
        for n in set(int(x) for x in r):
            counts[n] += 1
    return {n for n, c in counts.items() if c >= 2}


def intervention_transfer_synchronization(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """
    Synchronizes arrivals/departures at existing transfer hubs. No route geometry is changed.
    The effect is modelled as a small waiting-time reduction for the OD share that is likely
    to use transfer hubs.
    """
    routes = copy.deepcopy(disrupted.routes)
    hubs = transfer_hub_nodes(routes)
    nd = node_demand(disrupted.TD_dict)
    total_node_demand = sum(nd.values())
    hub_demand = sum(nd.get(h, 0.0) for h in hubs)
    hub_share = 0.0 if total_node_demand <= 1e-12 else max(0.0, min(1.0, hub_demand / total_node_demand))

    affected_share = max(0.10, min(TRANSFER_SYNC_SHARE, hub_share)) if hubs else 0.0
    att_delta = -TRANSFER_SYNC_GAIN_MIN * affected_share

    return DisruptionResult(
        routes=routes,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "transfer_synchronization",
            "transfer_hubs": sorted(hubs),
            "transfer_hubs_count": len(hubs),
            "assumed_affected_transfer_share": affected_share,
            "transfer_sync_gain_min": TRANSFER_SYNC_GAIN_MIN,
            "att_delta_min": att_delta,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_transfer_nodes": True,
        },
    )


# ============================================================
# EMERGENCY REROUTING + ACTION BUNDLES
# ============================================================

def build_available_adjacency(
    TT_dict: Dict[int, Dict[int, float]],
    forbidden_nodes: Set[int],
    forbidden_edges: Set[Tuple[int, int]],
) -> Dict[int, Dict[int, float]]:
    """Builds a directed graph from the available road network."""
    adj: Dict[int, Dict[int, float]] = defaultdict(dict)
    for u, row in TT_dict.items():
        try:
            uu = int(u)
        except Exception:
            continue
        if uu in forbidden_nodes:
            continue
        for v, val in row.items():
            try:
                vv = int(v)
                w = float(val)
            except Exception:
                continue
            if vv in forbidden_nodes:
                continue
            if (uu, vv) in forbidden_edges:
                continue
            if math.isfinite(w) and w > 0:
                adj[uu][vv] = w
    return adj


def shortest_path_available(
    adj: Dict[int, Dict[int, float]],
    source: int,
    target: int,
    max_expansions: int = 5000,
) -> Optional[List[int]]:
    """Dijkstra shortest path on the available road network."""
    import heapq

    source = int(source)
    target = int(target)
    if source == target:
        return [source]
    if source not in adj:
        return None

    heap = [(0.0, source, [source])]
    best = {source: 0.0}
    expansions = 0

    while heap and expansions < max_expansions:
        dist, u, path = heapq.heappop(heap)
        expansions += 1
        if u == target:
            return path
        if dist > best.get(u, float("inf")) + 1e-12:
            continue
        for v, w in adj.get(u, {}).items():
            nd = dist + w
            if nd + 1e-12 < best.get(v, float("inf")):
                best[v] = nd
                heapq.heappush(heap, (nd, v, path + [v]))
    return None


def repair_route_via_available_network(
    route: List[int],
    adj: Dict[int, Dict[int, float]],
    forbidden_nodes: Set[int],
    forbidden_edges: Set[Tuple[int, int]],
) -> Tuple[List[int], bool, int]:
    """
    Repairs a route by reconnecting consecutive available stops through the
    existing available road network. No new node is created; inserted nodes are
    nodes that already exist in the road graph.
    """
    available_stops = [int(n) for n in route if int(n) not in forbidden_nodes]
    if len(available_stops) < 2:
        return [], False, 0

    repaired = [available_stops[0]]
    changed = len(available_stops) != len(route)
    inserted_nodes = 0

    for target in available_stops[1:]:
        source = repaired[-1]
        direct_ok = target in adj.get(source, {}) and (source, target) not in forbidden_edges
        if direct_ok:
            repaired.append(target)
            continue

        path = shortest_path_available(adj, source, target)
        if path and len(path) >= 2:
            # Append path without repeating source.
            repaired.extend(path[1:])
            inserted_nodes += max(0, len(path) - 2)
            changed = True
        else:
            # If no connection exists, start a new continuous segment by skipping
            # this target. This avoids creating invalid edges.
            changed = True
            continue

    repaired = clean_routes([repaired])
    if not repaired:
        return [], False, 0
    return repaired[0], changed, inserted_nodes


def intervention_emergency_reroute_existing_network(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """
    Temporary rerouting on the existing available road network.
    It is designed for node_removal, area_closure, link_failure and
    combined disruptions. It avoids closed/removed nodes and affected edges,
    and reconnects route gaps using shortest available paths.
    """
    forbidden_nodes = affected_nodes_from_disruption(disrupted.info)
    forbidden_edges = affected_edges_from_disruption(disrupted.info)

    adj = build_available_adjacency(disrupted.TT_dict, forbidden_nodes, forbidden_edges)

    repaired_routes = []
    changed_routes = 0
    inserted_nodes_total = 0
    failed_routes = 0

    for r in baseline_routes:
        repaired, changed, inserted_nodes = repair_route_via_available_network(
            r,
            adj,
            forbidden_nodes,
            forbidden_edges,
        )
        if len(repaired) >= 2:
            repaired_routes.append(repaired)
            if changed:
                changed_routes += 1
            inserted_nodes_total += inserted_nodes
        else:
            failed_routes += 1

    repaired_routes = clean_routes(repaired_routes)
    if not repaired_routes:
        repaired_routes = copy.deepcopy(disrupted.routes)

    return DisruptionResult(
        routes=repaired_routes,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "emergency_reroute_existing_network",
            "forbidden_nodes": sorted(forbidden_nodes),
            "forbidden_edges_count": len(forbidden_edges) // 2 if forbidden_edges else 0,
            "changed_routes": changed_routes,
            "repaired_routes_count": len(repaired_routes),
            "failed_routes_count": failed_routes,
            "inserted_existing_nodes_count": inserted_nodes_total,
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "additional_vkm": 0.0,
            "additional_vh": 0.0,
            "uses_only_existing_road_network": True,
            "adds_new_nodes": False,
        },
    )


def _bridge_shores(baseline_routes: List[List[int]], forbidden: Set[int]) -> List[Tuple[int, int]]:
    """Οι «οχθες»: για καθε γραμμη, οι επιζωντες κομβοι εκατερωθεν καθε
    συνεχομενης ομαδας κλειστων κομβων. Αυτα ειναι ακριβως τα ζευγη που
    χρειαζονται γεφυρα."""
    pairs: List[Tuple[int, int]] = []
    for r in baseline_routes:
        r = [int(n) for n in r]
        i = 0
        while i < len(r):
            if r[i] in forbidden:
                j = i
                while j < len(r) and r[j] in forbidden:
                    j += 1
                u = r[i - 1] if i - 1 >= 0 else None
                v = r[j] if j < len(r) else None
                if u is not None and v is not None and u not in forbidden and v not in forbidden:
                    pairs.append((int(u), int(v)))
                i = j
            else:
                i += 1
    return pairs


def intervention_emergency_bus_bridge(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """ΓΕΦΥΡΑ ΛΕΩΦΟΡΕΙΩΝ (bus bridging).

    Η κλασικη πραγματικη αποκριση σε διακοπη: προσωρινη γραμμη-λεωφορειο που
    ενωνει τις δυο πλευρες της κλειστης περιοχης, με οχηματα που ελευθερωνονται
    απο τα ανασταλμενα τμηματα.

    ΚΡΙΣΙΜΗ ΛΕΠΤΟΜΕΡΕΙΑ: η γεφυρα ΔΕΝ τρεχει στον ιδιο δρομο που χρησιμοποιει
    ηδη η επισκευασμενη γραμμη. Αν το εκανε, θα ηταν ΑΚΡΙΒΩΣ διπλοτυπο μονοπατι
    και ο δεικτης πλεονασμου, που μετραει ΔΙΑΚΡΙΤΑ μονοπατια, δεν θα αλλαζε
    καθολου. Τρεχει σε ΕΝΑΛΛΑΚΤΙΚΟ διαδρομο: υπολογιζεται πρωτα ο συντομοτερος
    (αυτος που ηδη εξυπηρετειται) και μετα ο συντομοτερος που ΤΟΝ ΑΠΟΦΕΥΓΕΙ.
    Ετσι η γεφυρα προσθετει πραγματικα δευτερη επιλογη διαδρομης - δηλαδη
    πλεονασμο - οπως ακριβως κανει ενας φορεας που δεν βαζει το εκτακτο
    δρομολογιο πανω στον ηδη φορτωμενο δρομο.

    Το κοστος ΔΕΝ δηλωνεται ως additional_cost: οι νεες γραμμες μπαινουν στο
    routeset, αρα το κοστος τους υπολογιζεται αυτομοντα απο τα VKM/VH."""
    forbidden = affected_nodes_from_disruption(disrupted.info)
    forbidden_edges = affected_edges_from_disruption(disrupted.info)

    if not forbidden:
        return DisruptionResult(
            routes=copy.deepcopy(disrupted.routes),
            TT_dict=copy.deepcopy(disrupted.TT_dict),
            TD_dict=copy.deepcopy(disrupted.TD_dict),
            info={
                "intervention": "emergency_bus_bridge",
                "bridge_lines_count": 0,
                "att_delta_min": 0.0,
                "additional_cost": 0.0,
                "additional_energy": 0.0,
                "uses_only_existing_road_network": True,
                "adds_new_nodes": False,
            },
        )

    adj = build_available_adjacency(disrupted.TT_dict, forbidden, forbidden_edges)

    shuttles: List[List[int]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    alt_found = 0
    same_corridor = 0
    for u, v in _bridge_shores(baseline_routes, forbidden):
        if len(shuttles) >= BUS_BRIDGE_MAX_LINES:
            break
        key = (min(u, v), max(u, v))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        primary = shortest_path_available(adj, u, v)
        if not primary or len(primary) < 2:
            continue

        # Ο ΕΝΑΛΛΑΚΤΙΚΟΣ διαδρομος: ιδιες οχθες, αποφευγοντας τους ενδιαμεσους
        # κομβους του συντομοτερου (που ηδη εξυπηρετειται).
        blocked = set(forbidden) | set(primary[1:-1])
        adj_alt = build_available_adjacency(disrupted.TT_dict, blocked, forbidden_edges)
        alt = shortest_path_available(adj_alt, u, v)

        chosen = None
        if alt and 2 <= len(alt) <= BUS_BRIDGE_MAX_LEN:
            chosen = alt
            alt_found += 1
        elif 2 <= len(primary) <= BUS_BRIDGE_MAX_LEN:
            # Δεν υπαρχει εναλλακτικος διαδρομος: η γεφυρα τρεχει στον
            # μοναδικο διαθεσιμο. Προσθετει συχνοτητα, οχι πλεονασμο.
            chosen = primary
            same_corridor += 1
        if chosen:
            shuttles.append([int(x) for x in chosen])

    routes_new = copy.deepcopy(disrupted.routes)
    routes_new.extend(shuttles)
    routes_new = clean_routes(routes_new)
    if not routes_new:
        routes_new = copy.deepcopy(disrupted.routes)

    return DisruptionResult(
        routes=routes_new,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "emergency_bus_bridge",
            "forbidden_nodes": sorted(forbidden),
            "bridge_lines_count": len(shuttles),
            "bridge_alternative_corridor": alt_found,
            "bridge_same_corridor": same_corridor,
            "bridge_lines": [list(s) for s in shuttles],
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_road_network": True,
            "adds_new_nodes": False,
        },
    )


def intervention_extend_route_to_orphaned_stops(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """ΕΠΕΚΤΑΣΗ ΣΕ ΟΡΦΑΝΕΣ ΣΤΑΣΕΙΣ.

    Μετα τη ζημια καποιοι ΑΝΟΙΧΤΟΙ κομβοι με ζητηση μενουν χωρις καμια γραμμη -
    ειτε γιατι η μοναδικη γραμμη τους κοπηκε, ειτε γιατι εκει προσγειωθηκε η
    ζητηση επιβατων που περπατησαν απο κλειστη σταση ([D11] του disruption.py).
    Η ενεργεια επεκτεινει το ΑΚΡΟ της πλησιεστερης επιζωσας γραμμης ωστε να
    τους πιασει, μεσω του συντομοτερου διαθεσιμου δρομου.

    Ειναι λειτουργικη επεμβαση: υπαρχον οδικο δικτυο, υπαρχων στολος, μερικες
    ωρες. Το κοστος προκυπτει αυτομοντα απο τα VKM/VH του διευρυμενου routeset."""
    forbidden = affected_nodes_from_disruption(disrupted.info)
    forbidden_edges = affected_edges_from_disruption(disrupted.info)
    routes_new = [list(map(int, r)) for r in copy.deepcopy(disrupted.routes)]

    served: Set[int] = {n for r in routes_new for n in r}
    demand_nodes: Dict[int, float] = defaultdict(float)
    for o, row in (disrupted.TD_dict or {}).items():
        for d, dem in row.items():
            try:
                dem = float(dem)
            except Exception:
                continue
            if dem <= 0:
                continue
            demand_nodes[int(o)] += dem
            demand_nodes[int(d)] += dem

    orphans = [n for n, _ in sorted(demand_nodes.items(), key=lambda kv: -kv[1])
               if n not in served and n not in forbidden]
    orphans = orphans[:ORPHAN_EXTENSION_MAX]

    if not orphans:
        return DisruptionResult(
            routes=copy.deepcopy(disrupted.routes),
            TT_dict=copy.deepcopy(disrupted.TT_dict),
            TD_dict=copy.deepcopy(disrupted.TD_dict),
            info={
                "intervention": "extend_route_to_orphaned_stops",
                "orphaned_nodes_found": 0,
                "orphaned_nodes_served": 0,
                "att_delta_min": 0.0,
                "additional_cost": 0.0,
                "additional_energy": 0.0,
                "uses_only_existing_road_network": True,
                "adds_new_nodes": False,
            },
        )

    adj = build_available_adjacency(disrupted.TT_dict, forbidden, forbidden_edges)
    connected = 0
    extended_routes: Set[int] = set()

    for orphan in orphans:
        best = None            # (μηκος μονοπατιου, index γραμμης, "head"/"tail", path)
        for idx, r in enumerate(routes_new):
            if len(r) >= ORPHAN_EXTENSION_MAX_ROUTE_LEN:
                continue
            for end, node in (("tail", r[-1]), ("head", r[0])):
                p = shortest_path_available(adj, node, orphan) if end == "tail" \
                    else shortest_path_available(adj, orphan, node)
                if not p or len(p) < 2:
                    continue
                if len(r) + len(p) - 1 > ORPHAN_EXTENSION_MAX_ROUTE_LEN:
                    continue
                if best is None or len(p) < best[0]:
                    best = (len(p), idx, end, p)
        if best is None:
            continue
        _, idx, end, p = best
        if end == "tail":
            routes_new[idx] = routes_new[idx] + [int(x) for x in p[1:]]
        else:
            routes_new[idx] = [int(x) for x in p[:-1]] + routes_new[idx]
        extended_routes.add(idx)
        connected += 1

    routes_new = clean_routes(routes_new)
    if not routes_new:
        routes_new = copy.deepcopy(disrupted.routes)

    return DisruptionResult(
        routes=routes_new,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "extend_route_to_orphaned_stops",
            "forbidden_nodes": sorted(forbidden),
            "orphaned_nodes_found": len(orphans),
            "orphaned_nodes_served": connected,
            "extended_routes_count": len(extended_routes),
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_road_network": True,
            "adds_new_nodes": False,
        },
    )


def _node_demand_weights(TD_dict) -> Dict[int, float]:
    w: Dict[int, float] = defaultdict(float)
    for o, row in (TD_dict or {}).items():
        for d, dem in row.items():
            try:
                dem = float(dem)
            except Exception:
                continue
            if dem <= 0:
                continue
            w[int(o)] += dem
            w[int(d)] += dem
    return w


def intervention_redundant_overlay_line(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """ΠΑΡΑΛΛΗΛΗ ΓΡΑΜΜΗ ΑΝΑΚΟΥΦΙΣΗΣ.

    Προσωρινη γραμμη που ενωνει τα ΙΔΙΑ ακρα με τον πιο φορτωμενο αξονα, αλλα
    απο ΑΛΛΟΝ δρομο. Δινει δευτερη επιλογη διαδρομης στους επιβατες που σημερα
    εξαρτωνται απο εναν και μονο αξονα.

    ΓΙΑΤΙ ΕΙΝΑΙ ΑΠΑΡΑΙΤΗΤΗ: ειναι η μονη ενεργεια που δουλευει σε ΚΑΘΕ τυπο
    διαταραχης. Στην αιχμη ζητησης και στην αυξηση χρονων διαδρομης δεν
    κλεινει κανενας κομβος, αρα καμια απο τις υπολοιπες δομικες ενεργειες δεν
    εχει τι να πιασει - και το Επιπεδο 1 επεφτε πισω στις τρεις ενεργειες που
    δεν αγγιζουν καθολου το συνολο γραμμων.

    Οπως και η γεφυρα, ΔΕΝ τρεχει στον ιδιο δρομο: ο δεικτης πλεονασμου
    μετραει ΔΙΑΚΡΙΤΑ μονοπατια, αρα διπλοτυπη γραμμη δεν προσθετει τιποτα."""
    forbidden = affected_nodes_from_disruption(disrupted.info)
    forbidden_edges = affected_edges_from_disruption(disrupted.info)
    routes = [list(map(int, r)) for r in copy.deepcopy(disrupted.routes)]
    w = _node_demand_weights(disrupted.TD_dict)

    overlay = None
    axis_idx = None
    order = sorted(range(len(routes)), key=lambda i: -sum(w.get(n, 0.0) for n in routes[i]))
    for i in order[:OVERLAY_MAX_CANDIDATES]:
        r = routes[i]
        if len(r) < 3:
            continue
        # Δεν δοκιμαζεται ΜΟΝΟ το ανοιγμα ακρο-σε-ακρο. Σε αραιο δικτυο
        # οπως της Rivera, εναλλακτικος διαδρομος που παρακαμπτει ΟΛΟΥΣ τους
        # ενδιαμεσους κομβους μιας γραμμης 18 σταση συχνα δεν υπαρχει. Δοκιμαζονται
        # και τα δυο ΜΙΣΑ του αξονα, οπου η παρακαμψη ειναι πολυ πιθανοτερη.
        # Μετρημενο: παραλληλη γραμμη παραγοταν μονο στο 5 απο 11 διαταραχες.
        mid = len(r) // 2
        spans = [(0, len(r) - 1)]
        if len(r) >= 5:
            spans.extend([(0, mid), (mid, len(r) - 1)])
        for a, b in spans:
            if b - a < 2:
                continue
            blocked = set(forbidden) | set(r[a + 1:b])
            adj_alt = build_available_adjacency(disrupted.TT_dict, blocked, forbidden_edges)
            alt = shortest_path_available(adj_alt, r[a], r[b])
            if alt and 2 <= len(alt) <= OVERLAY_MAX_LEN:
                overlay = [int(x) for x in alt]
                axis_idx = i
                break
        if overlay:
            break

    routes_new = copy.deepcopy(routes)
    if overlay:
        routes_new.append(overlay)
    routes_new = clean_routes(routes_new)
    if not routes_new:
        routes_new = copy.deepcopy(disrupted.routes)

    return DisruptionResult(
        routes=routes_new,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "redundant_overlay_line",
            "overlay_added": bool(overlay),
            "overlay_line": overlay or [],
            "overlay_parallel_to_route_index": axis_idx,
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_road_network": True,
            "adds_new_nodes": False,
        },
    )


def intervention_suspend_low_value_segment(disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    """ΑΠΟΣΥΡΣΗ ΜΗ ΠΑΡΑΓΩΓΙΚΩΝ ΑΚΡΩΝ.

    Κοβει τα ακρα γραμμων που (α) εξυπηρετουν τη χαμηλοτερη ζητηση και (β)
    καλυπτονται ΚΑΙ απο αλλη γραμμη - αρα η αποσυρση τους ΔΕΝ χανει καλυψη.

    ΓΙΑΤΙ: με σκληρο προυπολογισμο , η αποκατασταση δεν μπορει να
    προσθεσει γεφυρα η παραλληλη γραμμη αν δεν βρει που να τα πληρωσει. Αυτη η
    ενεργεια ειναι ο ΧΡΗΜΑΤΟΔΟΤΗΣ: ελευθερωνει οχηματα-ωρες απο εκει που
    αποδιδουν λιγοτερο. Ειναι ακριβως αυτο που κανει ενας φορεας σε κριση -
    αραιωνει τα περιφερειακα ακρα για να κρατησει τον κορμο."""
    routes = [list(map(int, r)) for r in copy.deepcopy(disrupted.routes)]
    w = _node_demand_weights(disrupted.TD_dict)

    cover: Dict[int, int] = defaultdict(int)
    for r in routes:
        for n in set(r):
            cover[n] += 1

    suspended: List[int] = []
    for _ in range(SUSPEND_MAX_NODES):
        best = None            # (ζητηση, index γραμμης, ακρο)
        for idx, r in enumerate(routes):
            if len(r) <= SUSPEND_MIN_ROUTE_LEN:
                continue
            for end in (0, -1):
                n = r[end]
                if cover.get(n, 0) <= 1:
                    continue
                val = w.get(n, 0.0)
                if best is None or val < best[0]:
                    best = (val, idx, end)
        if best is None:
            break
        _, idx, end = best
        n = routes[idx][end]
        routes[idx] = routes[idx][1:] if end == 0 else routes[idx][:-1]
        cover[n] -= 1
        suspended.append(int(n))

    routes_new = clean_routes(routes)
    if not routes_new:
        routes_new = copy.deepcopy(disrupted.routes)

    return DisruptionResult(
        routes=routes_new,
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info={
            "intervention": "suspend_low_value_segment",
            "suspended_terminal_nodes": suspended,
            "suspended_count": len(suspended),
            "coverage_preserved_by_construction": True,
            "att_delta_min": 0.0,
            "additional_cost": 0.0,
            "additional_energy": 0.0,
            "uses_only_existing_road_network": True,
            "adds_new_nodes": False,
        },
    )


def feasible_plan_names(disruption_info: Dict[str, Any]) -> List[str]:
    dtype = disruption_info.get("type", "unknown")
    names = FEASIBLE_PLANS_BY_TYPE.get(dtype)
    if names:
        return names
    return list(RECOVERY_PLANS.keys())


def combine_intervention_infos(plan_name: str, plan_actions: List[str], infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "intervention": plan_name,
        "recovery_plan": plan_name,
        "plan_actions": plan_actions,
        "plan_length": len(plan_actions),
        "action_infos": infos,
        "att_delta_min": 0.0,
        "additional_cost": 0.0,
        "additional_energy": 0.0,
        "additional_vkm": 0.0,
        "additional_vh": 0.0,
    }

    # Sum additive operational effects.
    for info in infos:
        for key in ["att_delta_min", "additional_cost", "additional_energy", "additional_vkm", "additional_vh"]:
            combined[key] += safe_float(info.get(key)) or 0.0

    # Keep useful count fields for CSV summaries.
    count_keys = [
        "changed_routes",
        "short_turn_routes_count",
        "selected_routes_count",
        "critical_routes_count",
        "donor_routes_count",
        "transfer_hubs_count",
        "repaired_routes_count",
        "failed_routes_count",
        "inserted_existing_nodes_count",
    ]
    for key in count_keys:
        vals = [safe_float(info.get(key)) for info in infos]
        vals = [v for v in vals if v is not None]
        combined[key] = sum(vals) if vals else None

    # Keep the most relevant shares/frequencies when available.
    for key in [
        "frequency_increase",
        "frequency_increase_critical",
        "frequency_reduction_donor",
        "old_headway_min",
        "new_headway_min",
        "demand_share",
        "critical_demand_share",
        "donor_demand_share",
        "assumed_affected_transfer_share",
    ]:
        for info in reversed(infos):
            if info.get(key) is not None:
                combined[key] = info.get(key)
                break

    combined["uses_only_existing_network"] = True
    combined["positive_reward_filter"] = True
    return combined


def apply_recovery_plan(
    plan_name: str,
    disrupted: DisruptionResult,
    baseline_routes,
    baseline_TT,
    baseline_TD,
) -> DisruptionResult:
    """Applies a sequential recovery plan/action bundle."""
    plan_actions = RECOVERY_PLANS[plan_name]
    original_disruption_info = copy.deepcopy(disrupted.info)

    current = DisruptionResult(
        routes=copy.deepcopy(disrupted.routes),
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info=copy.deepcopy(original_disruption_info),
    )
    infos: List[Dict[str, Any]] = []

    for action in plan_actions:
        # Each action sees the original disruption metadata, while using the
        # current routes/TT/TD produced by previous actions in the bundle.
        action_input = DisruptionResult(
            routes=copy.deepcopy(current.routes),
            TT_dict=copy.deepcopy(current.TT_dict),
            TD_dict=copy.deepcopy(current.TD_dict),
            info=copy.deepcopy(original_disruption_info),
        )
        current = apply_intervention(action, action_input, baseline_routes, baseline_TT, baseline_TD)
        infos.append(copy.deepcopy(current.info))

    current.info = combine_intervention_infos(plan_name, plan_actions, infos)
    current.info["original_disruption_type"] = original_disruption_info.get("type")
    current.info["original_disruption_location"] = original_disruption_info.get("location")
    current.info["original_disruption_severity"] = original_disruption_info.get("severity")
    return current


def evaluate_recovery_plan(
    mod: Any,
    plan_name: str,
    disrupted: DisruptionResult,
    disrupted_metrics: Dict[str, Any],
    baseline_routes,
    baseline_TT,
    baseline_TD,
    assumptions: Dict[str, Any],
    scenario_name: str,
) -> Tuple[DisruptionResult, Dict[str, Any], float]:
    recovered = apply_recovery_plan(
        plan_name=plan_name,
        disrupted=disrupted,
        baseline_routes=baseline_routes,
        baseline_TT=baseline_TT,
        baseline_TD=baseline_TD,
    )
    recovered_metrics = evaluate_routes(mod, recovered.routes, recovered.TT_dict, recovered.TD_dict, assumptions, scenario_name=scenario_name)
    recovered_metrics = apply_operational_metric_adjustments(recovered_metrics, recovered.info, scenario_name)
    reward = compute_reward(disrupted_metrics, recovered_metrics, scenario_name)
    return recovered, recovered_metrics, reward


def _apply_move(move: str, state: "DisruptionResult", original_info: Dict[str, Any],
                baseline_routes, baseline_TT, baseline_TD) -> "DisruptionResult":
    """Εφαρμογη ΜΙΑΣ κινησης (ενεργειας η ολοκληρου πακετου) πανω στην
    ΤΡΕΧΟΥΣΑ κατασταση, με το info της ΑΡΧΙΚΗΣ διαταραχης - αλλιως οι ενεργειες
    δεν βλεπουν ποιοι κομβοι εκλεισαν."""
    inp = DisruptionResult(
        routes=copy.deepcopy(state.routes),
        TT_dict=copy.deepcopy(state.TT_dict),
        TD_dict=copy.deepcopy(state.TD_dict),
        info=copy.deepcopy(original_info),
    )
    if move in RECOVERY_PLANS:
        return apply_recovery_plan(move, inp, baseline_routes, baseline_TT, baseline_TD)
    return apply_intervention(move, inp, baseline_routes, baseline_TT, baseline_TD)


def _accumulate_adjustments(acc: Dict[str, float], info: Dict[str, Any]) -> Dict[str, Any]:
    """Οι λειτουργικες προσαρμογες (att_delta_min, additional_cost, ...)
    ΑΘΡΟΙΖΟΝΤΑΙ κατα μηκος της ακολουθιας. Αλλιως θα μετρουσε μονο η τελευταια."""
    for k in ("att_delta_min", "additional_cost", "additional_energy",
              "additional_vkm", "additional_vh"):
        v = safe_float(info.get(k))
        if v is not None:
            acc[k] = acc.get(k, 0.0) + float(v)
    return dict(acc)


def select_best_greedy_sequence(
    mod: Any,
    disruption_info: Dict[str, Any],
    disrupted: DisruptionResult,
    disrupted_metrics: Dict[str, Any],
    baseline_routes,
    baseline_TT,
    baseline_TD,
    assumptions: Dict[str, Any],
    scenario_name: str,
):
    """ΑΠΛΗΣΤΗ ΣΥΝΘΕΣΗ ΑΚΟΛΟΥΘΙΑΣ ΠΑΡΕΜΒΑΣΕΩΝ.

    Βημα 1  : το καλυτερο απο τα οκτω πακετα.
    Βηματα 2+: η καλυτερη ΠΡΟΣΘΕΤΙΚΗ δομικη ενεργεια (γεφυρα, παραλληλη γραμμη,
               επεκταση σε ορφανες, αποσυρση) - επαναληπτικα.
    Τελος   : οι τρεις λειτουργικες, μια φορα η καθεμια.

    Σε καθε βημα ισχυουν ΟΙ ΙΔΙΟΙ σκληροι περιορισμοι: κοστος <= B και
    DUN <= δαπεδο. Οταν υπαρχει ΑΝΑΚΤΗΣΙΜΗ ζητηση, προτεραιοτητα εχει η κινηση
    που ΜΕΙΩΝΕΙ περισσοτερο το DUN - η καλυψη πριν απο τον στοχο."""
    dun_floor = safe_float(disrupted.info.get("coverage_floor_percent"))
    B = recovery_budget(disrupted_metrics, assumptions)

    state = DisruptionResult(
        routes=copy.deepcopy(disrupted.routes),
        TT_dict=copy.deepcopy(disrupted.TT_dict),
        TD_dict=copy.deepcopy(disrupted.TD_dict),
        info=copy.deepcopy(disruption_info),
    )
    cur_metrics = copy.deepcopy(disrupted_metrics)
    acc: Dict[str, float] = {}
    sequence: List[str] = []
    candidate_rows: List[Dict[str, Any]] = []
    used: set = set()

    def score(metrics):
        return compute_reward(disrupted_metrics, metrics, scenario_name)

    for step in range(int(GREEDY_MAX_STEPS)):
        if step == 0:
            moves = [p for p in feasible_plan_names(disruption_info)]
        else:
            moves = [a for a in ADDITIVE_STRUCTURAL_ACTIONS if a not in used]
        if not moves:
            break

        best = None
        for move in moves:
            try:
                nxt = _apply_move(move, state, disruption_info, baseline_routes, baseline_TT, baseline_TD)
            except Exception:
                continue
            m = evaluate_routes(mod, nxt.routes, nxt.TT_dict, nxt.TD_dict, assumptions,
                                scenario_name=scenario_name)
            adj = _accumulate_adjustments(dict(acc), nxt.info)
            m = apply_operational_metric_adjustments(m, adj, scenario_name)
            ok, why = plan_is_feasible(disrupted_metrics, m, assumptions, dun_floor)
            r = score(m)
            candidate_rows.append({
                "step": step + 1, "move": move, "reward": r,
                "feasible": bool(ok), "infeasible_reason": (None if ok else why),
                "budget": B, "dun_floor": dun_floor,
                "att": m.get("att"), "CEF": m.get("CEF"),
                "dun": m.get("dun"), "cost": m.get("cost"), "energy": m.get("energy"),
            })
            if why == "over_budget":
                continue
            d_now = safe_float(cur_metrics.get("dun")) or 0.0
            d_new = safe_float(m.get("dun")) or 0.0
            # Καμια κινηση που ΧΕΙΡΟΤΕΡΕΥΕΙ την καλυψη δεν γινεται υποψηφια,
            # οσο θετικη κι αν ειναι η ανταμοιβη της. Το DUN ειναι περιορισμος,
            # οχι ορος που μπορει να αντισταθμιστει απο χρονο η πλεονασμο.
            if d_new > d_now + 1e-9:
                continue
            gap = (d_now - float(dun_floor)) if dun_floor is not None else 0.0
            # Η ΚΑΛΥΨΗ ΠΡΩΤΑ: οταν υπαρχει ανακτησιμη ζητηση, κριτηριο ειναι το DUN.
            key = ((d_new, -r) if gap > COVERAGE_FLOOR_TOL else (0.0, -r))
            if (not ok) and gap <= COVERAGE_FLOOR_TOL:
                continue
            if best is None or key < best[0]:
                best = (key, move, nxt, m, adj, r)

        if best is None:
            break
        key, move, nxt, m, adj, r = best
        improved = (r > score(cur_metrics) + GREEDY_MIN_GAIN)
        d_now = safe_float(cur_metrics.get("dun")) or 0.0
        d_new = safe_float(m.get("dun")) or 0.0
        closes_gap = (dun_floor is not None and d_now > float(dun_floor) + COVERAGE_FLOOR_TOL
                      and d_new < d_now - 1e-9)
        if not (improved or closes_gap):
            break
        state, cur_metrics, acc = nxt, m, adj
        sequence.append(move)
        used.add(move)
        if move in RECOVERY_PLANS:
            used.update(RECOVERY_PLANS[move])

    # τελικο περασμα: οι λειτουργικες, μια φορα η καθεμια
    for move in OPERATIONAL_ACTIONS:
        if move in used:
            continue
        try:
            nxt = _apply_move(move, state, disruption_info, baseline_routes, baseline_TT, baseline_TD)
        except Exception:
            continue
        m = evaluate_routes(mod, nxt.routes, nxt.TT_dict, nxt.TD_dict, assumptions,
                            scenario_name=scenario_name)
        adj = _accumulate_adjustments(dict(acc), nxt.info)
        m = apply_operational_metric_adjustments(m, adj, scenario_name)
        ok, why = plan_is_feasible(disrupted_metrics, m, assumptions, dun_floor)
        if not ok:
            continue
        if score(m) > score(cur_metrics) + GREEDY_MIN_GAIN:
            state, cur_metrics, acc = nxt, m, adj
            sequence.append(move)
            used.add(move)

    if not sequence:
        rec = intervention_do_nothing(disrupted, baseline_routes, baseline_TT, baseline_TD)
        floor_ok0 = (dun_floor is None or
                     (safe_float(disrupted_metrics.get("dun")) or 0.0) <= float(dun_floor) + COVERAGE_FLOOR_TOL)
        return "do_nothing", rec, copy.deepcopy(disrupted_metrics), 0.0, candidate_rows, bool(floor_ok0)

    name = "greedy:" + "+".join(sequence)
    state.info = combine_intervention_infos(name, sequence, [state.info])
    state.info["greedy_sequence"] = list(sequence)
    state.info["greedy_steps"] = len(sequence)
    state.info.update({k: v for k, v in acc.items()})
    floor_ok = (dun_floor is None or
                (safe_float(cur_metrics.get("dun")) or 0.0) <= float(dun_floor) + COVERAGE_FLOOR_TOL)
    return name, state, cur_metrics, score(cur_metrics), candidate_rows, bool(floor_ok)


def select_best_positive_plan(
    mod: Any,
    disruption_info: Dict[str, Any],
    disrupted: DisruptionResult,
    disrupted_metrics: Dict[str, Any],
    baseline_routes,
    baseline_TT,
    baseline_TD,
    assumptions: Dict[str, Any],
    scenario_name: str,
) -> Tuple[str, DisruptionResult, Dict[str, Any], float, List[Dict[str, Any]]]:
    """
    Greedy model-based recovery-agent step:
    evaluate all feasible recovery plans and choose the one with the highest
    reward. If the best reward is <= 0, caller will reject it as do_nothing.
    """
    candidate_rows: List[Dict[str, Any]] = []
    best_name = "do_nothing"
    best_recovered = intervention_do_nothing(disrupted, baseline_routes, baseline_TT, baseline_TD)
    best_metrics = copy.deepcopy(disrupted_metrics)
    best_reward = -float("inf")
    # Το δαπεδο καλυψης αυτης της διαταραχης, υπολογισμενο στο .
    dun_floor = safe_float(disrupted.info.get("coverage_floor_percent"))
    best_found = False
    pool: List[Tuple[str, DisruptionResult, Dict[str, Any], float, bool]] = []

    for plan_name in feasible_plan_names(disruption_info):
        recovered, metrics, reward = evaluate_recovery_plan(
            mod=mod,
            plan_name=plan_name,
            disrupted=disrupted,
            disrupted_metrics=disrupted_metrics,
            baseline_routes=baseline_routes,
            baseline_TT=baseline_TT,
            baseline_TD=baseline_TD,
            assumptions=assumptions,
            scenario_name=scenario_name,
        )
        # +ΣΚΛΗΡΗ ΕΦΙΚΤΟΤΗΤΑ πριν απο την ανταμοιβη.
        ok, why = plan_is_feasible(disrupted_metrics, metrics, assumptions, dun_floor)
        candidate_rows.append({
            "plan": plan_name,
            "plan_actions": RECOVERY_PLANS[plan_name],
            "reward": reward,
            "feasible": bool(ok),
            "infeasible_reason": (None if ok else why),
            "budget": recovery_budget(disrupted_metrics, assumptions),
            "dun_floor": dun_floor,
            "att": metrics.get("att"),
            "CEF": metrics.get("CEF"),
            "dun": metrics.get("dun"),
            "cost": metrics.get("cost"),
            "energy": metrics.get("energy"),
        })
        # Στη δεξαμενη της εφεδρικης επιλογης μπαινουν μονο σχεδια που δεν
        # χειροτερευουν την καλυψη. Το «οσο πιο κοντα στο δαπεδο γινεται» δεν
        # επιτρεπεται ποτε να καταληξει χειροτερα απο την ιδια τη ζημια.
        if why not in ("over_budget", "coverage_loss"):
            pool.append((plan_name, recovered, metrics, reward, bool(ok)))
        if not ok:
            continue
        if reward > best_reward:
            best_name = plan_name
            best_recovered = recovered
            best_metrics = metrics
            best_reward = reward
            best_found = True

    # ΛΕΞΙΚΟΓΡΑΦΙΚΗ ΕΠΙΛΟΓΗ. Πρωτα η καλυψη, μετα ο στοχος.
    # Αν ΚΑΝΕΝΑ σχεδιο δεν πιανει το δαπεδο, δεν επιστρεφουμε σιωπηλα στο
    # do_nothing: επιλεγεται το σχεδιο με το ΧΑΜΗΛΟΤΕΡΟ DUN (και με την
    # καλυτερη ανταμοιβη σε ισοπαλια) και καταγραφεται οτι το δαπεδο δεν
    # πιαστηκε. Η ανεξυπηρετητη ζητηση δεν επιτρεπεται να αγνοηθει.
    floor_reached = bool(best_found)
    if not best_found and pool:
        pool.sort(key=lambda t: (safe_float(t[2].get("dun")) if t[2].get("dun") is not None else 1e9,
                                 -t[3]))
        best_name, best_recovered, best_metrics, best_reward, _ = pool[0]
    return best_name, best_recovered, best_metrics, best_reward, candidate_rows, floor_reached


def apply_intervention(action: str, disrupted: DisruptionResult, baseline_routes, baseline_TT, baseline_TD) -> DisruptionResult:
    if action == "increase_frequency_high_demand_routes":
        return intervention_increase_frequency_high_demand_routes(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "short_turn_service_existing_routes":
        return intervention_short_turn_service_existing_routes(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "vehicle_reallocation_critical_routes":
        return intervention_vehicle_reallocation_critical_routes(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "transfer_synchronization":
        return intervention_transfer_synchronization(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "emergency_reroute_existing_network":
        return intervention_emergency_reroute_existing_network(disrupted, baseline_routes, baseline_TT, baseline_TD)
    #
    if action == "emergency_bus_bridge":
        return intervention_emergency_bus_bridge(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "extend_route_to_orphaned_stops":
        return intervention_extend_route_to_orphaned_stops(disrupted, baseline_routes, baseline_TT, baseline_TD)
    #
    if action == "redundant_overlay_line":
        return intervention_redundant_overlay_line(disrupted, baseline_routes, baseline_TT, baseline_TD)
    if action == "suspend_low_value_segment":
        return intervention_suspend_low_value_segment(disrupted, baseline_routes, baseline_TT, baseline_TD)

    raise ValueError(f"Unknown intervention: {action}")


def apply_operational_metric_adjustments(metrics: Dict[str, Any], intervention_info: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """
    Applies operational effects that the original static evaluator does not capture directly:
    - frequency/headway changes affect ATT through waiting time, not edge travel time;
    - extra/redistributed service affects cost and energy.
    """
    adjusted = copy.deepcopy(metrics)

    att = safe_float(adjusted.get("att"))
    att_delta = safe_float(intervention_info.get("att_delta_min")) or 0.0
    if att is not None:
        adjusted["att"] = max(0.0, att + att_delta)
    # Κρατιεται χωριστα ωστε η ανταμοιβη να μπορει να ΜΗΝ τη μετραει.
    adjusted["att_assumed_delta_min"] = float(att_delta)

    cost = safe_float(adjusted.get("cost"))
    add_cost = safe_float(intervention_info.get("additional_cost")) or 0.0
    if cost is not None:
        adjusted["cost"] = max(0.0, cost + add_cost)

    energy = safe_float(adjusted.get("energy"))
    add_energy = safe_float(intervention_info.get("additional_energy")) or 0.0
    if scenario_name == "S2_energy" and energy is not None:
        adjusted["energy"] = max(0.0, energy + add_energy)
    elif scenario_name != "S2_energy":
        adjusted["energy"] = None

    vkm = safe_float(adjusted.get("vkm"))
    add_vkm = safe_float(intervention_info.get("additional_vkm")) or 0.0
    if vkm is not None:
        adjusted["vkm"] = max(0.0, vkm + add_vkm)

    vh = safe_float(adjusted.get("vh"))
    add_vh = safe_float(intervention_info.get("additional_vh")) or 0.0
    if vh is not None:
        adjusted["vh"] = max(0.0, vh + add_vh)

    adjusted["operational_adjustment"] = {
        "att_delta_min": att_delta,
        "additional_cost": add_cost,
        "additional_energy": add_energy if scenario_name == "S2_energy" else None,
        "additional_vkm": add_vkm,
        "additional_vh": add_vh,
    }
    return adjusted


# ============================================================
# RL AGENT
# ============================================================

def discretize_metric(value: Optional[float], baseline: Optional[float], thresholds=(0.05, 0.20)) -> int:
    v = safe_float(value)
    b = safe_float(baseline)
    if v is None or b is None or abs(b) < 1e-12:
        return 0

    rel = (v - b) / b
    if rel < thresholds[0]:
        return 0
    if rel < thresholds[1]:
        return 1
    return 2


def state_key(disruption_info: Dict[str, Any], disrupted_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Tuple[Any, ...]:
    dtype = disruption_info.get("type", "unknown")
    location = disruption_info.get("location", "unknown")
    severity = disruption_info.get("severity", "unknown")

    att_level = discretize_metric(disrupted_metrics.get("att"), baseline_metrics.get("att"))

    dun = safe_float(disrupted_metrics.get("dun")) or 0.0
    if dun <= 0.0:
        dun_level = 0
    elif dun < 10:
        dun_level = 1
    else:
        dun_level = 2

    cef_base = safe_float(baseline_metrics.get("CEF"))
    cef_now = safe_float(disrupted_metrics.get("CEF"))

    if cef_base is None or cef_now is None or abs(cef_base) < 1e-12:
        cef_loss_level = 0
    else:
        loss = (cef_base - cef_now) / cef_base
        if loss < 0.05:
            cef_loss_level = 0
        elif loss < 0.20:
            cef_loss_level = 1
        else:
            cef_loss_level = 2

    return (dtype, location, severity, att_level, cef_loss_level, dun_level)


# ΑΦΑΙΡΕΘΗΚΑΝ ΟΙ ΣΥΝΑΡΤΗΣΕΙΣ choose_action / update_q / epsilon_for_episode.
#      Οριζονταν εδω αλλα ΔΕΝ ΚΑΛΟΥΝΤΑΝ ΠΟΤΕ - επαληθευτηκε με αναλυση AST
#      ολοκληρου του αρχειου (κλησεις: 0, 0, 0). Ο μηχανισμος που εκτελειται
#      ειναι η select_best_positive_plan: εξαντλητικη ντετερμινιστικη αναζητηση
#      πανω σε ΟΛΑ τα εφικτα πακετα παρεμβασεων. Βλ. .
#      (Αφαιρεθηκε επισης η classify_routes, που επισης δεν καλουνταν ποτε.)


def compute_reward(disrupted_metrics: Dict[str, Any], recovered_metrics: Dict[str, Any],
                   scenario_name: str = "S1_service") -> float:
    """Η ανταμοιβη ειναι Ο ΣΤΟΧΟΣ ΤΟΥ ΣΧΕΔΙΑΣΜΟΥ.

        r = (CEF_μετα - CEF_πριν)/CEF_R0  +  (ATT_πριν - ATT_μετα)/ATT_R0
        (+ (E_πριν - E_μετα)/E_R0  μονο στο S2_energy)

    Ειναι ΑΚΡΙΒΩΣ η compute_reward των v6/v7 (§3.2.5) και του recovery_rl.py:
    ιδιος στοχος, ιδια κανονικοποιηση ως προς την αρχικη λυση R0. Ετσι τα τρια
    σταδια - σχεδιασμος, λειτουργικη αποκριση, τακτικη ανασυνταξη - μετρανε το
    ιδιο πραγμα και τα νουμερα τους συγκρινονται μεταξυ τους.

    Το DUN και το κοστος ΔΕΝ ειναι οροι της ανταμοιβης· ειναι ΣΚΛΗΡΟΙ
    ΠΕΡΙΟΡΙΣΜΟΙ (), οπως ακριβως και στον σχεδιασμο. Ετσι δεν μπορει
    πλεον ενα σχεδιο να «αγορασει» χρονο ταξιδιου με απεριοριστο χρημα."""
    if REWARD_MODE == "legacy_weights":
        imp_att = improvement_decrease(disrupted_metrics.get("att"), recovered_metrics.get("att")) or 0.0
        imp_cef = improvement_increase(disrupted_metrics.get("CEF"), recovered_metrics.get("CEF")) or 0.0
        imp_dun = improvement_decrease(disrupted_metrics.get("dun"), recovered_metrics.get("dun")) or 0.0
        imp_cost = improvement_decrease(disrupted_metrics.get("cost"), recovered_metrics.get("cost")) or 0.0
        imp_energy = improvement_decrease(disrupted_metrics.get("energy"), recovered_metrics.get("energy")) or 0.0
        return (W_ATT * imp_att + W_CEF * imp_cef + W_DUN * imp_dun
                + W_COST * imp_cost + W_ENERGY * imp_energy)

    ref_cef = abs(safe_float(REFERENCE_R0.get("CEF")) or 0.0)
    ref_att = abs(safe_float(REFERENCE_R0.get("att")) or 0.0)
    ref_e = abs(safe_float(REFERENCE_R0.get("energy")) or 0.0)
    ref_cef = ref_cef if ref_cef > 1e-9 else 1.0
    ref_att = ref_att if ref_att > 1e-9 else 1.0
    ref_e = ref_e if ref_e > 1e-9 else 1.0

    c0 = safe_float(disrupted_metrics.get("CEF")) or 0.0
    c1 = safe_float(recovered_metrics.get("CEF")) or 0.0
    a0 = safe_float(disrupted_metrics.get("att")) or 0.0
    a1 = safe_float(recovered_metrics.get("att")) or 0.0
    if not REWARD_USES_ASSUMED_ATT:
        # Αφαιρειται η ΥΠΟΤΙΘΕΜΕΝΗ μειωση χρονου, ωστε να μεινει μονο η
        # ΜΕΤΡΗΜΕΝΗ μεταβολη που προκυπτει απο αλλαγη των γραμμων.
        _assumed = safe_float(recovered_metrics.get("att_assumed_delta_min")) or 0.0
        a1 = a1 - _assumed

    r = (c1 - c0) / ref_cef + (a0 - a1) / ref_att

    # Ο ιδιος ορος με τον σχεδιασμο. Χωρις αυτον ο πρακτορας θα ανταμειβοταν
    # για να ξανακανει πυκνο ο,τι ο σχεδιασμος αραιωσε σκοπιμα, και τα δυο
    # σταδια θα τραβουσαν σε αντιθετες κατευθυνσεις.
    if RECOVERY_USE_CEF2:
        ref_c2 = abs(safe_float(REFERENCE_R0.get("CEF2")) or 0.0)
        ref_c2 = ref_c2 if ref_c2 > 1e-9 else 1.0
        x0 = safe_float(disrupted_metrics.get("CEF2"))
        x1 = safe_float(recovered_metrics.get("CEF2"))
        if x0 is not None and x1 is not None:
            r += (x1 - x0) / ref_c2

    if scenario_name == "S2_energy":
        e0 = safe_float(disrupted_metrics.get("energy"))
        e1 = safe_float(recovered_metrics.get("energy"))
        if e0 is not None and e1 is not None:
            r += (e0 - e1) / ref_e
    return float(r)


def recovery_budget(disrupted_metrics: Dict[str, Any], assumptions: Dict[str, Any]) -> Optional[float]:
    """Το σκληρο οριο δαπανης της αποκαταστασης."""
    if RECOVERY_BUDGET_MODE != "design":
        return None
    b_design = safe_float(assumptions.get("budget_total"))
    c_dis = safe_float(disrupted_metrics.get("cost"))
    if b_design is None:
        return None
    if c_dis is None:
        return float(b_design) * (1.0 + RECOVERY_BUDGET_MARGIN)
    return float(max(b_design, c_dis)) * (1.0 + RECOVERY_BUDGET_MARGIN)


def plan_is_feasible(disrupted_metrics: Dict[str, Any], recovered_metrics: Dict[str, Any],
                     assumptions: Dict[str, Any], dun_floor: Optional[float] = None) -> Tuple[bool, str]:
    """+ΣΚΛΗΡΗ ΕΦΙΚΤΟΤΗΤΑ - ιδια λογικη με τον σχεδιασμο.

    (α) Το κοστος δεν ξεπερνα τον προυπολογισμο.
    (β) Η ΚΑΛΥΨΗ ΕΠΑΝΕΡΧΕΤΑΙ ΣΤΟ ΔΑΠΕΔΟ: dun <= dun_floor. Ειναι το
        αυστηροτερο κριτηριο που ειναι φυσικα δυνατο - καλυτερα απο το δαπεδο
        δεν γινεται - και ταυτιζεται με το [RL5] του recovery_rl.py, ωστε τα
        δυο επιπεδα να κρινονται με τον ιδιο κανονα. Οταν το δαπεδο δεν ειναι
        γνωστο, πεφτει πισω στο ασθενεστερο «να μη χειροτερεψει».

    Χωρις το (α), το σχεδιο `full_emergency_recovery` κερδιζε ξοδευοντας
    +10 550 EUR για -1.57 λεπτα, με τον πλεονασμο να μενει στο μηδεν."""
    B = recovery_budget(disrupted_metrics, assumptions)
    cost = safe_float(recovered_metrics.get("cost"))
    if B is not None and cost is not None and cost > B + 1e-6:
        return False, "over_budget"

    d1 = safe_float(recovered_metrics.get("dun"))
    d0 = safe_float(disrupted_metrics.get("dun"))

    # Η καλυψη δεν επιτρεπεται ΠΟΤΕ να χειροτερεψει σε σχεση με τη ζημια. Ο
    # ελεγχος αυτος προηγειται του δαπεδου: οταν το δαπεδο ειναι απροσιτο, το
    # «οσο πιο κοντα γινεται» δεν σημαινει «χειροτερα απο εκει που ξεκινησαμε».
    if ENFORCE_NO_COVERAGE_LOSS and d0 is not None and d1 is not None and d1 > d0 + 1e-9:
        return False, "coverage_loss"

    if ENFORCE_COVERAGE_FLOOR and dun_floor is not None and d1 is not None:
        if d1 > float(dun_floor) + COVERAGE_FLOOR_TOL:
            return False, "above_coverage_floor"
    return True, "ok"


# ============================================================
# OUTPUT / SUMMARY
# ============================================================

def flatten_episode(
    scenario: str,
    ep: int,
    original_mc_run: int,
    phase: str,
    disruption_info: Dict[str, Any],
    action: str,
    intervention_info: Dict[str, Any],
    disrupted_metrics: Dict[str, Any],
    recovered_metrics: Dict[str, Any],
    reward: float,
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "episode": ep,
        "original_mc_run": original_mc_run,
        "phase": phase,
        # Παλια ετικετα, παραγομενη απο τη νεα - ωστε καμια υπαρχουσα
        # αναλυση των CSV που φιλτραρει σε train/eval να μη σπασει.
        "phase_legacy": ("train" if phase == "calibration" else "eval"),
        "type": disruption_info.get("type"),
        "location": disruption_info.get("location"),
        "severity": disruption_info.get("severity"),
        "action": action,
        "recovery_plan": intervention_info.get("recovery_plan", action),
        "plan_length": intervention_info.get("plan_length"),
        "plan_actions": json.dumps(intervention_info.get("plan_actions", []), ensure_ascii=False),
        "applied": intervention_info.get("applied"),
        "rejected_plan": intervention_info.get("rejected_plan"),
        "rejected_reward": intervention_info.get("rejected_reward"),

        "att_disrupted": disrupted_metrics.get("att"),
        "att_recovered": recovered_metrics.get("att"),
        "improvement_att": improvement_decrease(disrupted_metrics.get("att"), recovered_metrics.get("att")),

        "CEF_disrupted": disrupted_metrics.get("CEF"),
        "CEF_recovered": recovered_metrics.get("CEF"),
        "improvement_CEF": improvement_increase(disrupted_metrics.get("CEF"), recovered_metrics.get("CEF")),

        # Ο περιορισμενος πλεονασμος και η εκτεθειμενη ζητηση, ΠΡΙΝ και
        # ΜΕΤΑ την αποκατασταση. Το exposure ειναι το νουμερο του κεφαλαιου:
        # «η αποκατασταση επανεφερε την εκτεθειμενη ζητηση απο X % σε Y %».
        "CEF2_disrupted": disrupted_metrics.get("CEF2"),
        "CEF2_recovered": recovered_metrics.get("CEF2"),
        "improvement_CEF2": improvement_increase(disrupted_metrics.get("CEF2"),
                                                 recovered_metrics.get("CEF2")),

        "exposure_disrupted": disrupted_metrics.get("exposure"),
        "exposure_recovered": recovered_metrics.get("exposure"),
        "improvement_exposure": improvement_decrease(disrupted_metrics.get("exposure"),
                                                     recovered_metrics.get("exposure")),

        "dun_disrupted": disrupted_metrics.get("dun"),
        "dun_recovered": recovered_metrics.get("dun"),
        "improvement_dun": improvement_decrease(disrupted_metrics.get("dun"), recovered_metrics.get("dun")),

        "cost_disrupted": disrupted_metrics.get("cost"),
        "cost_recovered": recovered_metrics.get("cost"),
        "improvement_cost": improvement_decrease(disrupted_metrics.get("cost"), recovered_metrics.get("cost")),

        "energy_disrupted": disrupted_metrics.get("energy"),
        "energy_recovered": recovered_metrics.get("energy"),
        "improvement_energy": improvement_decrease(disrupted_metrics.get("energy"), recovered_metrics.get("energy")),

        "reward": reward,

        "changed_routes": intervention_info.get("changed_routes"),
        "short_turn_routes_count": intervention_info.get("short_turn_routes_count"),
        "selected_routes_count": intervention_info.get("selected_routes_count"),
        "critical_routes_count": intervention_info.get("critical_routes_count"),
        "donor_routes_count": intervention_info.get("donor_routes_count"),
        "transfer_hubs_count": intervention_info.get("transfer_hubs_count"),
        "repaired_routes_count": intervention_info.get("repaired_routes_count"),
        "failed_routes_count": intervention_info.get("failed_routes_count"),
        "inserted_existing_nodes_count": intervention_info.get("inserted_existing_nodes_count"),
        "frequency_increase": intervention_info.get("frequency_increase"),
        "frequency_increase_critical": intervention_info.get("frequency_increase_critical"),
        "frequency_reduction_donor": intervention_info.get("frequency_reduction_donor"),
        "old_headway_min": intervention_info.get("old_headway_min"),
        "new_headway_min": intervention_info.get("new_headway_min"),
        "coverage_floor_percent": intervention_info.get("coverage_floor_percent"),
        "coverage_gap_before": intervention_info.get("coverage_gap_before"),
        "coverage_gap_after": intervention_info.get("coverage_gap_after"),
        "coverage_floor_reached": intervention_info.get("coverage_floor_reached"),
        "applied_for_coverage": intervention_info.get("applied_for_coverage"),
        "att_delta_min": intervention_info.get("att_delta_min"),
        "demand_share": intervention_info.get("demand_share"),
        "critical_demand_share": intervention_info.get("critical_demand_share"),
        "donor_demand_share": intervention_info.get("donor_demand_share"),
        "assumed_affected_transfer_share": intervention_info.get("assumed_affected_transfer_share"),
        "additional_vkm": intervention_info.get("additional_vkm"),
        "additional_vh": intervention_info.get("additional_vh"),
        "additional_cost": intervention_info.get("additional_cost"),
        "additional_energy": intervention_info.get("additional_energy"),
    }


def summarize(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)

    metrics = [
        "improvement_att",
        "improvement_CEF",
        "improvement_CEF2",          #
        "improvement_exposure",      #
        "improvement_dun",
        "improvement_cost",
        "improvement_energy",
        "reward",
    ]

    out = []
    for key, vals in grouped.items():
        s = {k: v for k, v in zip(group_keys, key)}
        s["n"] = len(vals)

        for m in metrics:
            nums = [safe_float(r.get(m)) for r in vals]
            nums = [x for x in nums if x is not None]

            if nums:
                nums_sorted = sorted(nums)
                mean = sum(nums_sorted) / len(nums_sorted)
                std = math.sqrt(sum((x - mean) ** 2 for x in nums_sorted) / len(nums_sorted)) if len(nums_sorted) > 1 else 0.0
                idx95 = min(len(nums_sorted) - 1, int(math.ceil(0.95 * len(nums_sorted))) - 1)

                s[f"mean_{m}"] = mean
                s[f"std_{m}"] = std
                s[f"min_{m}"] = nums_sorted[0]
                s[f"max_{m}"] = nums_sorted[-1]
                s[f"p95_{m}"] = nums_sorted[idx95]
            else:
                s[f"mean_{m}"] = None
                s[f"std_{m}"] = None
                s[f"min_{m}"] = None
                s[f"max_{m}"] = None
                s[f"p95_{m}"] = None

        out.append(s)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    random.seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    root = (script_dir / DATA_DIR).resolve() if not Path(DATA_DIR).is_absolute() else Path(DATA_DIR).resolve()
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    module_path = root / TRAINING_MODULE_FILE
    # Αν το ονομα δεν ταιριαζει (π.χ. v7.r2.py), ψαξε οποιοδηποτε v7*.py
    # και μετα v6*.py. Το module χρησιμοποιειται ΜΟΝΟ για τις συναρτησεις
    # αξιολογησης, που ειναι ΤΑΥΤΟΣΗΜΕΣ σε v6 και v7.
    if not module_path.exists():
        _c = []
        for _pat in ("v7*.py", "v6*.py"):
            for _r in (script_dir, root):
                _c.extend(_r.glob(_pat))
        _c = sorted({q.resolve() for q in _c}, key=lambda q: q.stat().st_mtime, reverse=True)
        if _c:
            print(f"Το {module_path.name} δεν βρεθηκε. Χρηση του: {_c[0].name}")
            print("(μονο για τις συναρτησεις αξιολογησης - ιδιες σε v6/v7)")
            module_path = _c[0]
    seek_folder = root / SEEKALLPATHS_FOLDER
    mod = import_training_module(module_path, seek_folder)

    # Αν το ρητο μονοπατι δεν υπαρχει (π.χ. νεος timestamped φακελος μετα
    # απο επανεκτελεση των v6/v7, η νεος φακελος monte_carlo_* μετα το
    # disruption.py), βρες αυτοματα το ΠΙΟ ΠΡΟΣΦΑΤΟ. Ο τροπος ΑΠΟΘΗΚΕΥΣΗΣ δεν
    # αλλαζει - αλλαζει μονο το ΔΙΑΒΑΣΜΑ.
    def _resolve(explicit: str, patterns: List[str], label: str) -> Path:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
        cands: List[Path] = []
        for pat in patterns:
            cands.extend(script_dir.glob(pat))
            cands.extend(root.glob(pat))
        cands = sorted(set(cands), key=lambda q: q.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(
                f"Δεν βρεθηκε {label}: ουτε στο {p} ουτε με τα μοτιβα {patterns}."
            )
        import datetime as _dt
        _chosen = cands[0]
        _when = _dt.datetime.fromtimestamp(_chosen.stat().st_mtime)
        _age = (_dt.datetime.now() - _when).days
        print(f"Το {p} δεν βρεθηκε.")
        print(f"Χρηση του: {_chosen}")
        print(f"Ημερομηνια αρχειου: {_when:%Y-%m-%d %H:%M}  ({_age} ημερες πριν)")
        if _age >= 1:
            print("*** ΠΡΟΣΟΧΗ: το αρχειο ΔΕΝ ειναι σημερινο - ελεγξε το μονοπατι.")
        return _chosen

    best_json_path = _resolve(
        BEST_JSON_FILE,
        ["best_scenarios_*.json", "*/best_scenarios_*.json"],
        "best_scenarios JSON")
    mc_json_path = _resolve(
        MONTE_CARLO_RESULTS_FILE,
        ["*monte_carlo_disruption_results.json", "*/*monte_carlo_disruption_results.json"],
        "Monte Carlo JSON")

    best_data = load_json(best_json_path)

    # Η αναφορα R0 - ΑΚΡΙΒΩΣ η ιδια που χρησιμοποιουν τα v6/v7 για την
    # κανονικοποιηση της ανταμοιβης. Χωρις αυτην, η ανταμοιβη του Επιπεδου 1
    # δεν ειναι συγκρισιμη ουτε με τον σχεδιασμο ουτε με το recovery_rl.py.
    _init = best_data.get("initial_solution", {}) or {}
    REFERENCE_R0.clear()
    REFERENCE_R0.update({
        "att": safe_float(_init.get("att")),
        "CEF": safe_float(_init.get("CEF")),
        "CEF2": safe_float(_init.get("CEF2")),        #
        "cost": safe_float(_init.get("cost")),
        "energy": safe_float(_init.get("energy")),
        "dun": safe_float(_init.get("dun")),
    })
    print(f"Αναφορα R0: ATT={REFERENCE_R0.get('att')}  CEF={REFERENCE_R0.get('CEF')}  "
          f"C0={REFERENCE_R0.get('cost')}")
    # Χωρις αυτη την αναφορα ο ορος dCEF2 δεν μπορει να κανονικοποιηθει
    # οπως στον σχεδιασμο - και τοτε ΔΕΝ πρεπει να μπει καθολου.
    if RECOVERY_USE_CEF2:
        _c2ref = REFERENCE_R0.get("CEF2")
        if _c2ref is None:
            print("*** ΠΡΟΣΟΧΗ: το best_scenarios JSON ΔΕΝ εχει CEF2 στο")
            print("*** initial_solution. Ο ορος dCEF2 ΔΕΝ θα μπει -")
            print("*** σχεδιασμος και αποκατασταση θα μετρανε ΑΛΛΟ πραγμα.")
            print("*** Ξανατρεξε το v7.py.")
        else:
            print(f"Ανταμοιβη: + dCEF2 / CEF2_R0 και στα δυο σεναρια   (CEF2_R0 = {_c2ref})")
    print(f"Εκτακτο περιθωριο προυπολογισμου: {RECOVERY_BUDGET_MARGIN:.0%}")
    print(f"Ανταμοιβη: {REWARD_MODE}   |   προυπολογισμος: {RECOVERY_BUDGET_MODE}   "
          f"|   πακετα: {len(RECOVERY_PLANS)}")
    mc_data = load_json(mc_json_path)

    global TT_MATRIX_GLOBAL
    global DIST_KM_GLOBAL

    TT_dict, TD_dict, tt_matrix, _td_matrix = mod.load_rivera(
        travel_file=str(root / TRAVEL_FILE),
        demand_file=str(root / DEMAND_FILE),
        coords_file=str(root / COORDS_FILE),
    )

    coords = mod.read_coords_latlon(str(root / COORDS_FILE))
    # Οι συντεταγμενες χρειαζονται για την προσβαση με περπατημα.
    COORDS_GLOBAL.clear()
    COORDS_GLOBAL.update({int(k): (float(v[0]), float(v[1])) for k, v in coords.items()})
    DIST_KM_GLOBAL = mod.build_directed_dist_km(TT_dict, coords)
    TT_MATRIX_GLOBAL = tt_matrix

    assumptions = {
        "c_km": C_KM,
        "c_h": C_H,
        "e_km": E_KM,
        "headway_min": HEADWAY_MIN,
        "operating_hours": OPERATING_HOURS,
        "assume_bidirectional": ASSUME_BIDIRECTIONAL,
    }

    all_rows = []

    results = {
        "methodological_note": (
            "Recovery-agent analysis applied to the already generated v7 Monte Carlo disruptions. "
            "No new random disruptions are created. The recovery agent evaluates feasible action bundles/recovery plans per existing disruption run. "
            "Five base interventions are available: increase_frequency_high_demand_routes, short_turn_service_existing_routes, "
            "vehicle_reallocation_critical_routes, transfer_synchronization, and emergency_reroute_existing_network. "
            "The emergency rerouting action uses only existing nodes and the available road network. No new nodes are introduced. "
            "The best plan is finally applied only when its reward is positive; otherwise the episode is recorded as do_nothing with zero operational change. Energy is reported only for S2_energy; for S1_service it is set to null."
        ),
        "inputs": {
            "best_json": str(best_json_path),          #
            "monte_carlo_results": str(mc_json_path),  #
            "training_module": str(module_path),
            "episodes_per_scenario": EPISODES_PER_SCENARIO,
            # Διατηρειται το ονομα για συμβατοτητα με τα παλια αρχεια εξοδου,
            # αλλα σημαινει «ποσοστο επεισοδιων εκτος του held-out συνολου».
            "training_ratio": TRAINING_RATIO,
            "random_seed": RANDOM_SEED,
            "central_nodes": CENTRAL_NODES,
            "scenarios": SCENARIOS_TO_RUN,
            "interventions": INTERVENTIONS,
            "recovery_plans": RECOVERY_PLANS,
            "feasible_plans_by_type": FEASIBLE_PLANS_BY_TYPE,
            "reward_mode": REWARD_MODE,
            "recovery_use_cef2": bool(RECOVERY_USE_CEF2),      #
            "recovery_start": RECOVERY_START,        #
            "recovery_budget_mode": RECOVERY_BUDGET_MODE,
            "recovery_budget_margin": RECOVERY_BUDGET_MARGIN,
            "greedy_composition": bool(GREEDY_COMPOSITION),
            "greedy_max_steps": GREEDY_MAX_STEPS,
            "reward_uses_assumed_att": bool(REWARD_USES_ASSUMED_ATT),
            "enforce_no_coverage_loss": bool(ENFORCE_NO_COVERAGE_LOSS),
            "reference_R0": dict(REFERENCE_R0),
            # ΤΙΜΙΑ ΟΝΟΜΑΣΙΑ. Δεν εκτελειται Q-learning: οι choose_action /
            # update_q / epsilon_for_episode οριζονταν αλλα ΔΕΝ καλουνταν ποτε
            # (επαληθευση με αναλυση AST). Ο μηχανισμος ειναι εξαντλητικη
            # ντετερμινιστικη αναζητηση: για καθε διαταραχη αξιολογουνται ΟΛΑ
            # τα εφικτα πακετα παρεμβασεων και επιλεγεται το μεγιστο θετικο
            # reward. Λειτουργει ως ΑΝΩ ΦΡΑΓΜΑ (oracle) της αποδοσης
            # οποιασδηποτε μαθημενης πολιτικης ΕΝΟΣ ΒΗΜΑΤΟΣ.
            "agent_policy": "exhaustive_one_step_recovery_oracle",
            "agent_policy_note": (
                "Deterministic exhaustive search over all feasible one-step intervention "
                "bundles; the bundle with the maximum positive reward is applied, otherwise "
                "do_nothing. No reinforcement learning is performed. This is an upper bound "
                "on the performance of any learned one-step recovery policy."
            ),
            "held_out_ratio": HELD_OUT_RATIO,
            "frequency_increase": FREQUENCY_INCREASE,
            "high_demand_route_share": HIGH_DEMAND_ROUTE_SHARE,
            "reallocation_increase": REALLOCATION_INCREASE,
            "reallocation_reduction": REALLOCATION_REDUCTION,
            "transfer_sync_gain_min": TRANSFER_SYNC_GAIN_MIN,
            "positive_reward_filter": True,
            "fallback_action": "do_nothing",
            "assumptions": assumptions,
        },
        "scenarios": {},
    }

    for scenario_name in SCENARIOS_TO_RUN:
        # Ο προυπολογισμος του ΣΧΕΔΙΑΣΜΟΥ για αυτο το σεναριο: B = C0(1+p),
        # με το p να διαβαζεται απο το ιδιο το best_scenarios JSON. Ο σκληρος
        # ελεγχος του γινεται πανω σε αυτο.
        _p = safe_float(best_data["best_scenarios"][scenario_name].get("p")) or 0.0
        _c0 = safe_float(REFERENCE_R0.get("cost"))
        assumptions["budget_p"] = float(_p)
        # Ο ΠΛΗΡΗΣ εγκεκριμενος προυπολογισμος - οχι ο μειωμενος που δεσμευει ο
        # σχεδιασμος. Η διαφορα των δυο ειναι η εφεδρεια που ξοδευει ο πρακτορας.
        assumptions["budget_total"] = (float(_c0) * (1.0 + float(_p))) if _c0 else None
        print(f"{scenario_name}: p={_p:.2f}  B_total={assumptions['budget_total']}")

        baseline_routes = copy.deepcopy(best_data["best_scenarios"][scenario_name]["best_routes"])
        baseline_metrics = evaluate_routes(mod, baseline_routes, TT_dict, TD_dict, assumptions, scenario_name=scenario_name)

        saved_runs = mc_data["scenarios"][scenario_name]["runs"][:EPISODES_PER_SCENARIO]
        # Ονομα «calibration / held_out» αντι για «train / eval»: δεν γινεται
        # εκπαιδευση, αρα δεν υπαρχει τιποτα να εκπαιδευτει. Το held-out συνολο
        # δειχνει οτι η αποδοση του oracle ειναι σταθερη.
        training_episodes = int(round(len(saved_runs) * TRAINING_RATIO))

        # Ο πινακας Q μενει ΚΕΝΟΣ γιατι ΔΕΝ γινεται Q-learning. Διατηρειται
        # στην εξοδο μονο για συμβατοτητα δομης με τα παλια αρχεια.
        q_table = {}
        scenario_episodes = []

        for idx, saved_run in enumerate(saved_runs, start=1):
            # "calibration" = τα πρωτα 70%, "held_out" = τα υπολοιπα 30%.
            # Οι παλιες ετικετες train/eval διατηρουνται ως δευτερο πεδιο ωστε
            # να μη σπασει καμια υπαρχουσα αναλυση των CSV.
            phase = "calibration" if idx <= training_episodes else "held_out"
            phase_legacy = "train" if idx <= training_episodes else "eval"

            disruption_info = saved_run["disruption"]
            original_mc_run = saved_run.get("run", idx)

            disrupted = replay_saved_disruption(
                baseline_routes=baseline_routes,
                TT_dict=TT_dict,
                TD_dict=TD_dict,
                disruption_info=disruption_info,
            )

            disrupted_metrics = evaluate_routes(mod, disrupted.routes, disrupted.TT_dict, disrupted.TD_dict, assumptions, scenario_name=scenario_name)

            state = state_key(disruption_info, disrupted_metrics, baseline_metrics)

            _selector = (select_best_greedy_sequence if GREEDY_COMPOSITION
                         else select_best_positive_plan)   #
            candidate_plan, recovered, recovered_metrics, candidate_reward, candidate_rows, floor_reached = _selector(
                mod=mod,
                disruption_info=disruption_info,
                disrupted=disrupted,
                disrupted_metrics=disrupted_metrics,
                baseline_routes=baseline_routes,
                baseline_TT=TT_dict,
                baseline_TD=TD_dict,
                assumptions=assumptions,
                scenario_name=scenario_name,
            )

            # ΚΑΝΟΝΑΣ ΑΠΟΔΟΧΗΣ - Η ΚΑΛΥΨΗ ΥΠΕΡΙΣΧΥΕΙ ΤΗΣ ΑΝΤΑΜΟΙΒΗΣ.
            #
            # Παλια: το σχεδιο εφαρμοζοταν ΜΟΝΟ αν reward > 0. Ετσι ενα σχεδιο
            # που ΕΠΑΝΕΦΕΡΕ ανεξυπηρετητη ζητηση αλλα εχανε λιγο σε χρονο η
            # πλεονασμο απορριπτοταν, και η ζητηση εμενε ακαλυπτη.
            #
            # Τωρα: εφαρμοζεται αν (α) βελτιωνει τον στοχο, Η (β) υπαρχει
            # ΑΝΑΚΤΗΣΙΜΗ ζητηση (DUN πανω απο το δαπεδο) και το σχεδιο την
            # μειωνει. Η καλυψη ειναι υποχρεωση εξυπηρετησης, οχι ορος
            # βελτιστοποιησης.
            candidate_info = copy.deepcopy(recovered.info)

            _floor = safe_float(disrupted.info.get("coverage_floor_percent"))
            _d0 = safe_float(disrupted_metrics.get("dun"))
            _d1 = safe_float(recovered_metrics.get("dun"))
            _gap_before = (None if (_floor is None or _d0 is None)
                           else max(0.0, _d0 - float(_floor)))
            _gap_after = (None if (_floor is None or _d1 is None)
                          else max(0.0, _d1 - float(_floor)))
            _restores_coverage = bool(
                _gap_before is not None and _gap_before > COVERAGE_FLOOR_TOL
                and _d0 is not None and _d1 is not None and _d1 < _d0 - 1e-9)

            # Τελικο διχτυ ασφαλειας: ο περιορισμος της καλυψης ελεγχεται και
            # στο σημειο εφαρμογης, ανεξαρτητα απο τον επιλογεα. Οποιο σχεδιο
            # αφηνει περισσοτερη ζητηση ανεξυπηρετητη απ' οσο η ιδια η ζημια
            # δεν εφαρμοζεται ποτε, οσο θετικη κι αν ειναι η ανταμοιβη του.
            _worsens_coverage = bool(_d0 is not None and _d1 is not None and _d1 > _d0 + 1e-9)

            if (candidate_reward > 0.0 or _restores_coverage) and not _worsens_coverage:
                action = candidate_plan
                reward = candidate_reward
                recovered.info["candidate_plan"] = candidate_plan
                recovered.info["candidate_reward"] = candidate_reward
                recovered.info["candidate_evaluations"] = candidate_rows
                recovered.info["applied"] = True
                recovered.info["applied_for_coverage"] = bool(
                    _restores_coverage and candidate_reward <= 0.0)
                recovered.info["coverage_floor_percent"] = _floor
                recovered.info["coverage_gap_before"] = _gap_before
                recovered.info["coverage_gap_after"] = _gap_after
                recovered.info["coverage_floor_reached"] = bool(floor_reached)
            else:
                action = "do_nothing"
                recovered = intervention_do_nothing(disrupted, baseline_routes, TT_dict, TD_dict)
                recovered.info.update({
                    "intervention": "do_nothing",
                    "recovery_plan": "do_nothing",
                    "plan_actions": [],
                    "plan_length": 0,
                    "applied": False,
                    "rejected_plan": candidate_plan,
                    "rejected_reward": candidate_reward,
                    "rejected_reason": ("coverage_loss" if _worsens_coverage
                                        else "non_positive_reward"),
                    "rejected_intervention_info": candidate_info,
                    "candidate_evaluations": candidate_rows,
                    "positive_reward_filter": True,
                    # Διαφανεια: το do_nothing επιλεγεται μονο οταν δεν
                    # υπηρχε ανακτησιμη ζητηση να καλυφθει.
                    "coverage_floor_percent": _floor,
                    "coverage_gap_before": _gap_before,
                    "coverage_gap_after": _gap_before,
                    "coverage_floor_reached": bool(
                        _gap_before is not None and _gap_before <= COVERAGE_FLOOR_TOL),
                })
                recovered_metrics = copy.deepcopy(disrupted_metrics)
                reward = 0.0

            row = flatten_episode(
                scenario=scenario_name,
                ep=idx,
                original_mc_run=original_mc_run,
                phase=phase,
                disruption_info=disruption_info,
                action=action,
                intervention_info=recovered.info,
                disrupted_metrics=disrupted_metrics,
                recovered_metrics=recovered_metrics,
                reward=reward,
            )

            all_rows.append(row)

            scenario_episodes.append({
                "episode": idx,
                "original_mc_run": original_mc_run,
                "phase": phase,
                "phase_legacy": phase_legacy,   # συμβατοτητα με τα παλια JSON
                "state": list(state),
                "disruption": disruption_info,
                "action": action,
                "intervention_info": recovered.info,
                "metrics_after_disruption": {k: v for k, v in disrupted_metrics.items() if k != "routes"},
                "metrics_after_intervention": {k: v for k, v in recovered_metrics.items() if k != "routes"},
                "improvements": {
                    "att": row["improvement_att"],
                    "CEF": row["improvement_CEF"],
                    "dun": row["improvement_dun"],
                    "cost": row["improvement_cost"],
                    "energy": row["improvement_energy"],
                },
                "reward": reward,
            })

        q_json = q_table

        results["scenarios"][scenario_name] = {
            "baseline": {k: v for k, v in baseline_metrics.items() if k != "routes"},
            "episodes": scenario_episodes,
            "q_table": q_json,
        }

    summary_all = summarize(all_rows, ["scenario", "phase"])
    summary_by_type = summarize(all_rows, ["scenario", "phase", "type"])
    summary_by_type_location = summarize(all_rows, ["scenario", "phase", "type", "location"])
    summary_by_action = summarize(all_rows, ["scenario", "phase", "action"])

    write_json(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_results.json", results)
    write_csv(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_episodes.csv", all_rows)
    write_csv(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_all.csv", summary_all)
    write_csv(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_type.csv", summary_by_type)
    write_csv(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_type_location.csv", summary_by_type_location)
    write_csv(out_dir / "v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_action.csv", summary_by_action)

    print("Ολοκληρωθηκε. Ο πρακτορας αποκαταστασης εφαρμοστηκε στις καταγεγραμμενες διαταραχες.")
    print("Files written:")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_results.json'}")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_episodes.csv'}")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_all.csv'}")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_type.csv'}")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_type_location.csv'}")
    print(f" - {out_dir / 'v7_recovery_agent_5actions_plan_positive_reward_existing_mc_summary_by_action.csv'}")


if __name__ == "__main__":
    main()
