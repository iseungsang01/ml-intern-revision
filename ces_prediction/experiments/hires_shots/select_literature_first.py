"""Literature-first selection: published shots take slots, data fills the rest.

This is the ordering the Notion request document already uses -- three of its twelve slots
(#31873, #32027, #31359) are published discharges that FAIL the data gate and were kept
anyway. The scan's job is therefore to find published shots to ADD, not a V_rot floor to
filter them out with.
"""
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def A(x):
    return str(x).encode("ascii", "replace").decode("ascii")


d = pd.read_csv(HERE / "shot_scored_v2.csv").set_index("shot")
answered = set(json.loads((HERE / "fulltext_index_hits.json").read_text(encoding="utf-8"))["queried"])

# Published discharges disqualified on data grounds that literature CANNOT override:
# the MC "signal" is an electrical spike, so a microsecond fetch buys nothing to analyse.
ARTIFACT_OUT = {31276: "MC RMS collapses to 32% on trimming (kurt 363) -- spike, not a mode",
                31888: "MC carried by spikes (trim 0.36, kurt 105)"}
COMPANION = {31923: 31921, 31357: 31359}      # same-session halves of a published pair

LIT = {31921: "P1/P2 FIRE mode (fig.10 / fig.3,7-9)",
       31873: "P3 ELM suppression (fig.5 + SI)",
       31359: "P4 error field, ERMP OFF (fig.6)",
       32027: "P5 PanoMHD, L/H transition (fig.7)",
       31097: "NEW Phys.Plasmas 2025 RMP-induced edge kink-like modes",
       31747: "NEW EPJ Web Conf 313 02005, NTM stabilisation by ECCD",
       31923: "P1/P2 FIRE mode companion (fig.11-13 / fig.2)",
       31357: "P4 error field, ERMP ON -- the paired control of 31359",
       31276: "P4 fig.3", 31888: "Bayesian NN disruption prediction"}

print(A("=" * 96))
print(A("TIER 1 -- every published shot we hold, and what happens to it"))
print(A("=" * 96))
print(A(f"{'shot':>6} {'s42':>6} {'vt':>5} {'ti':>5} {'gate':>5} {'artf':>5} {'score':>6}  disposition"))
roles_lit, comps = [], []
for s in sorted(LIT):
    r = d.loc[s]
    if s in ARTIFACT_OUT:
        dispo = "EXCLUDED: " + ARTIFACT_OUT[s]
    elif s in COMPANION:
        dispo = f"companion of #{COMPANION[s]} (fetch, never train/test)"
        comps.append(s)
    else:
        dispo = "ROLE"
        roles_lit.append(s)
    print(A(f"{s:>6} {r.split_s42:>6} {int(r.vt_clean_n):>5} {int(r.ti_clean_n):>5} "
            f"{'Y' if r.pass_gate else 'n':>5} {'Y' if r.artifact_free else 'n':>5} "
            f"{r.score_v2:>6.3f}  {dispo}"))

print(A(f"\nliterature role shots: {roles_lit}  ({len(roles_lit)})"))
print(A(f"companions: {comps}"))

# ---- Tier 2: fill the remaining role slots from the data score -------------------------
N_ROLES = 10
need = N_ROLES - len(roles_lit)
pool = d[(d.pass_gate) & (d.artifact_free)].sort_values("score_v2", ascending=False)
cand = pool[~pool.index.isin(LIT)]

# test must be s42-test, and -- the k=2 fix -- every test shot must carry V_rot.
lit_test = [s for s in roles_lit if d.loc[s].split_s42 == "test"]
lit_test_with_vt = [s for s in lit_test if d.loc[s].vt_clean_n >= 200]
n_test_needed = 3 - len(lit_test)
print(A(f"\nliterature shots already on the s42-test side: {lit_test} "
        f"(of which carrying V_rot >= 200: {lit_test_with_vt})"))

test_fill = [int(s) for s in cand[(cand.split_s42 == "test")
                                  & (cand.vt_clean_n >= 200)].index[:n_test_needed]]
rest_fill = [int(s) for s in cand[(cand.split_s42 != "test")].index[:need - len(test_fill)]]

print(A("\n" + "=" * 96))
print(A(f"LITERATURE-FIRST {N_ROLES} + {len(comps)} companions"))
print(A("=" * 96))
print(A(f"{'shot':>6} {'role':>6} {'src':>4} {'vt':>5} {'ti':>5} {'gate':>5} {'score':>6}  why"))
final = []
for s in roles_lit:
    role = "test" if d.loc[s].split_s42 == "test" else "pool"
    final.append((s, role, "LIT", LIT[s]))
for s in test_fill:
    final.append((s, "test", "data", "score_v2 fill, carries V_rot"))
for s in rest_fill:
    final.append((s, "pool", "data", "score_v2 fill"))
for s, role, src, why in final:
    r = d.loc[s]
    print(A(f"{s:>6} {role:>6} {src:>4} {int(r.vt_clean_n):>5} {int(r.ti_clean_n):>5} "
            f"{'Y' if r.pass_gate else 'n':>5} {r.score_v2:>6.3f}  {why[:44]}"))
for s in comps:
    r = d.loc[s]
    print(A(f"{s:>6} {'comp':>6} {'LIT':>4} {int(r.vt_clean_n):>5} {int(r.ti_clean_n):>5} "
            f"{'Y' if r.pass_gate else 'n':>5} {r.score_v2:>6.3f}  {LIT[s][:44]}"))

tests = [s for s, role, _, _ in final if role == "test"]
print(A(f"\nliterature share of the {N_ROLES} roles: {len(roles_lit)}/{N_ROLES} "
        f"(the Notion 12-list had 4/10 + 2 companions)"))
print(A(f"test triple: {tests} -> valid V_rot {[int(d.loc[s].vt_clean_n) for s in tests]}"))
dead = [s for s in tests if d.loc[s].vt_clean_n < 200]
print(A(f"test shots without usable V_rot: {dead}  "
        f"-> CES_VT effective clusters = {len(tests) - len(dead)}"))
print(A(f"total valid V_rot over the {N_ROLES} roles: "
        f"{int(sum(d.loc[s].vt_clean_n for s, _, _, _ in final))}"))
print(A(f"unscanned among them (could still gain a citation): "
        f"{[s for s, _, _, _ in final if s not in answered]}"))
