"""B.73: the fit units of the four validation designs (registration section 5, Addendum 1).

A unit = (name, design, sub, fit discharges, test discharges); discharges are file indices of the D1 discharges.
"""
from __future__ import annotations

import common as C

import numpy as np  # noqa: E402

TREE_SPEC = {
    # design -> {learner: targets}
    "V1": {"T0": C.TARGETS, "TX": C.TARGETS, "TB": ("F", "L5"), "TE": ("F", "L5"), "TXlag": ("F", "L5")},
    "V2": {"T0": ("F", "L5"), "TX": ("F", "L5")},
    "V3": {"T0": ("F", "L5"), "TX": ("F", "L5")},
    "V4": {"T0": ("F", "L5"), "TX": ("F", "L5"), "TB": ("F", "L5"), "TE": ("F", "L5"), "TXlag": ("F", "L5")},
}
FILTER_TARGETS = ("F", "L5")
CLF_DESIGNS = ("V1", "V4")


def discharge_table(R):
    """Per D1 discharge: file index, shot, cluster, V1 fold in each re-partition."""
    u, first = np.unique(R["fidx"], return_index=True)
    tab = {"fidx": u, "shot": R["shot"][first], "clus": R["clus"][first]}
    for r in C.REPEATS:
        kk = R[f"r{r}_k"]
        fold = np.full(len(u), -1)
        for j, i in enumerate(u):
            v = np.unique(kk[R["fidx"] == i])
            if len(v) != 1:
                raise SystemExit(f"FATAL: discharge {i} is in more than one fold of r{r}")
            fold[j] = v[0]
        tab[f"fold_r{r}"] = fold
    return tab


def v2_assignment(tab, seed):
    """Whole shot clusters to five folds: clusters in a random order (seed), each to the fold with the fewest D1
    discharges so far (ties: lowest fold index); Addendum 1."""
    cl, cnt = np.unique(tab["clus"], return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cl))
    load = np.zeros(C.K, np.int64)
    fold_of = {}
    for j in order:
        f = int(np.argmin(load))
        fold_of[int(cl[j])] = f
        load[f] += cnt[j]
    return np.array([fold_of[int(c)] for c in tab["clus"]]), load


def units(R=None):
    R = R if R is not None else C.rows()
    tab = discharge_table(R)
    u = tab["fidx"]
    out = []
    for r in C.REPEATS:
        f = tab[f"fold_r{r}"]
        for k in range(C.K):
            out.append({"name": f"V1_r{r}_k{k}", "design": "V1", "sub": r, "fit": u[f != k], "test": u[f == k]})
    for s in C.V2_SEEDS:
        f, _ = v2_assignment(tab, s)
        for k in range(C.K):
            out.append({"name": f"V2_s{s}_k{k}", "design": "V2", "sub": s, "fit": u[f != k], "test": u[f == k]})
    o = np.argsort(tab["shot"], kind="stable")
    fifth = np.zeros(len(u), np.int64)
    for q, part in enumerate(np.array_split(o, 5)):
        fifth[part] = q
    for q in range(1, 5):
        out.append({"name": f"V3_q{q}", "design": "V3", "sub": q, "fit": u[fifth < q], "test": u[fifth == q]})
    sh = tab["shot"]
    out.append({"name": "V4a", "design": "V4", "sub": "a", "fit": u[sh <= C.V4_FIT_A], "test": u[sh >= C.V4_TEST_FROM]})
    out.append({"name": "V4b", "design": "V4", "sub": "b", "fit": u[sh <= C.V4_FIT_B], "test": u[sh >= C.V4_TEST_FROM]})
    return out, tab, fifth


def describe():
    R = C.rows()
    us, tab, fifth = units(R)
    o = {"d1_discharges": int(len(tab["fidx"])), "clusters_d1": int(len(np.unique(tab["clus"])))}
    cl, cnt = np.unique(tab["clus"], return_counts=True)
    o["cluster_sizes_d1_discharges"] = {"max": int(cnt.max()), "median": float(np.median(cnt)), "min": int(cnt.min()),
                                        "largest_five": sorted(cnt.tolist())[-5:]}
    for s in C.V2_SEEDS:
        _, load = v2_assignment(tab, s)
        o[f"v2_seed{s}_fold_discharges"] = load.tolist()
    sh = tab["shot"]
    o["v3_fifths"] = [{"shots": [int(sh[fifth == q].min()), int(sh[fifth == q].max())], "discharges": int((fifth == q).sum())}
                      for q in range(5)]
    o["v4"] = {"fit_a": int((sh <= C.V4_FIT_A).sum()), "fit_b": int((sh <= C.V4_FIT_B).sum()),
               "test": int((sh >= C.V4_TEST_FROM).sum()),
               "between_31991_and_32002": int(((sh > C.V4_FIT_A) & (sh < 32002)).sum()),
               "shot_32311": int((sh == 32311).sum())}
    o["units"] = [{"name": x["name"], "fit": int(len(x["fit"])), "test": int(len(x["test"]))} for x in us]
    return o


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=1))
