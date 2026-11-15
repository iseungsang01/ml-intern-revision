"""Fix the FAIR-MAST shot lists from the catalogue alone (no measurement is read).

Population: level-1 shots that carry the `act` source (time-resolved charge exchange).
Excluded: the 13 pilot shots already inspected (venue_plan/PLAN_review.md sec. 1).
gate_shots.txt : 300 shots, 75 per campaign M6-M9, drawn with a fixed seed from the
                 `act` shots whose catalogue quality is "Not Checked".
all_shots.txt  : every remaining `act` shot (download only; not part of the gate).
Run from the repo root:  py ces_prediction/experiments/mast/select_shots.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

META = Path("data/mast/meta")
OUT = Path(__file__).parent
PILOT = [17703, 18330, 18064, 21770, 24411, 22622, 26581, 27244, 27288, 30448, 30013, 28991, 30147]
SEED = 20261116
PER_CAMPAIGN = 75

src = pd.read_parquet(META / "l1_sources.parquet")
shots = pd.read_parquet(META / "l1_shots.parquet")[["shot_id", "campaign"]]
act = src[src.name == "act"][["shot_id", "quality"]].merge(shots, on="shot_id")
act = act[~act.shot_id.isin(PILOT)].sort_values("shot_id").reset_index(drop=True)

rng = np.random.default_rng(SEED)
gate = []
for camp in ["M6", "M7", "M8", "M9"]:
    ids = act[(act.campaign == camp) & (act.quality == "Not Checked")].shot_id.to_numpy()
    gate += sorted(int(x) for x in rng.choice(ids, PER_CAMPAIGN, replace=False))
gate_set = set(gate)
rest = [int(x) for x in act.shot_id if int(x) not in gate_set]

(OUT / "gate_shots.txt").write_text("\n".join(map(str, gate)) + "\n")
(OUT / "all_shots.txt").write_text("\n".join(map(str, rest)) + "\n")
print("act shots after pilot exclusion:", len(act))
print(act.groupby(["campaign", "quality"]).size().to_string())
print("gate:", len(gate), "rest:", len(rest))
