"""B.86 readout (registration sections 4-5, Addendum 1 sections 7-8): statements, per-band tables, value of the frame,
counts, into the JSON; then the markdown.  Refuses to run unless the four checks passed.

    py ces_prediction/experiments/b86_mast_radius/readout86.py
Output: docs/paper/b86_mast_radius.json / .md.  The progress log carries row counts only.
"""
from __future__ import annotations

import importlib.util
import json
import time

import common86 as C

import numpy as np  # noqa: E402

N = C.N_BOOT
CHUNK = 1000
FORMS = ("session", "shot")          # order of the draws in every stream
EST_L5 = ("H", "G", "T0", "TX", "O")
EST_F = ("H", "G", "T0", "TX", "PCHIP", "OGP")


def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


# ------------------------------------------------------------------------------------------ bootstrap cells
class Cell:
    """Per-unit sums of named per-row columns, over one or more parts (bands) placed on the union of their units, and one
    set of draws per form from the stream of the cell's name (session draws first, then shot draws)."""

    def __init__(self, name, parts):
        self.name = name
        self.tot = {}
        self.draws = {}
        rng = C.rng_for(name)
        self.units = {}
        for form in FORMS:
            labs = np.concatenate([p[form] for p in parts])
            u = np.unique(labs)
            nu = len(u)
            self.units[form] = int(nu)
            cols, V = [], []
            for p in parts:
                inv = np.searchsorted(u, p[form])
                for k, v in p["cols"].items():
                    cols.append(k)
                    V.append(np.bincount(inv, v, nu))
            V = np.column_stack(V)
            D = np.empty((N, V.shape[1]))
            pr = np.full(nu, 1.0 / nu)
            for a in range(0, N, CHUNK):
                m = min(CHUNK, N - a)
                D[a:a + m] = rng.multinomial(nu, pr, size=m).astype(np.float64) @ V
            self.draws[form] = {k: D[:, i] for i, k in enumerate(cols)}
            if form == "session":
                self.tot = {k: float(V[:, i].sum()) for i, k in enumerate(cols)}

    def val(self, fn, *keys):
        out = {"point": float(fn(*[np.array(self.tot[k]) for k in keys]))}
        for form in FORMS:
            out[form] = q(fn(*[self.draws[form][k] for k in keys]))
        out["stream"] = self.name
        return out


SK = lambda A, B: 1 - A / B          # noqa: E731
RATIO = lambda A, B: A / B           # noqa: E731
REC = lambda T, X, F: (T - X) / (T - F)   # noqa: E731


def part(rows, shot_lab, sess_lab, cols):
    return {"shot": shot_lab[rows], "session": sess_lab[rows], "cols": cols}


# ------------------------------------------------------------------------------------------ band data
class Band:
    def __init__(self, b):
        self.b = b
        keys = ["shot", "y", "L5", "rho", "common_F", "common_L5"] + [f"{r}_{k}" for r in C.RULES for k in ("H", "G")]
        self.R = C.load_band(b, keys)
        self.n = len(self.R["shot"])
        self.us, tab = C.units(b)
        so = dict(zip(tab["shots"], tab["session"]))
        self.shot_lab = self.R["shot"].astype(np.int64)
        self.sess_lab = np.array([so.get(int(s), "none") for s in self.R["shot"]])
        self.refs = C.load_npz(C.CACHE / f"refs_{b}.npz")

    def units_of(self, d):
        return [u for u in self.us if u["design"] == d]

    def scored(self, d, X):
        uu = self.units_of(d)
        s = np.isin(self.R["shot"], np.array(sorted(set().union(*[set(u["test"]) for u in uu]))))
        return np.flatnonzero(s & self.R[f"common_{X}"])

    def fitted(self, d, L, r, X):
        a = np.full(self.n, np.nan)
        for u in self.units_of(d):
            z = np.load(C.fit_path(self.b, u["name"], L, r, X))
            a[z["idx"]] = z["est"]
        return a

    def estimates(self, d, r, X):
        R = self.R
        e = {"H": R[f"{r}_H"], "G": R[f"{r}_G"], "T0": self.fitted(d, "T0", r, X), "TX": self.fitted(d, "TX", r, X)}
        if X == "L5":
            e["O"] = R["y"]
        else:
            e["PCHIP"], e["OGP"] = self.refs["P"], self.refs["O"]
        return e


def row_cols(est, rows, tgt, prefix=""):
    """Per-row columns of SE, |e|, within 10 % and 20 % of the target, and the row count."""
    cols = {f"{prefix}n": np.ones(len(rows))}
    for k, v in est.items():
        e = v[rows] - tgt
        if not np.all(np.isfinite(e)):
            raise SystemExit(f"FATAL: {k} not finite on scored rows")
        cols[f"{prefix}se_{k}"] = e ** 2
        cols[f"{prefix}ae_{k}"] = np.abs(e)
        cols[f"{prefix}in10_{k}"] = (np.abs(e) <= 0.10 * tgt).astype(np.float64)
        cols[f"{prefix}in20_{k}"] = (np.abs(e) <= 0.20 * tgt).astype(np.float64)
    return cols


def measures(cell, est, rows, tgt, prefix=""):
    """Points of every measure, with intervals for the shares (ratios of per-unit sums)."""
    out = {}
    n = cell.tot[f"{prefix}n"]
    for k, v in est.items():
        e = v[rows] - tgt
        out[k] = {"rmse_eV": float(np.sqrt(cell.tot[f"{prefix}se_{k}"] / n)),
                  "MAE_eV": float(cell.tot[f"{prefix}ae_{k}"] / n),
                  "median_abs_error_eV": float(np.median(np.abs(e))),
                  "median_rel_error": float(np.median(np.abs(e) / tgt)),
                  "share_within_10pct": cell.val(RATIO, f"{prefix}in10_{k}", f"{prefix}n"),
                  "share_within_20pct": cell.val(RATIO, f"{prefix}in20_{k}", f"{prefix}n")}
    return out


# ------------------------------------------------------------------------------------------ tables
def band_tables(B):
    out = {}
    for d in ("M1", "M3"):
        rows = B.scored(d, "L5")
        tgt = B.R["L5"][rows]
        cols, ests = {}, {}
        for r in C.RULES:
            ests[r] = B.estimates(d, r, "L5")
            cols.update(row_cols(ests[r], rows, tgt, prefix=f"{r}|"))
        cell = Cell(f"{B.b}|{d}|L5|D5+D100|table", [part(rows, B.shot_lab, B.sess_lab, cols)])
        o = {"counts": {"rows": int(len(rows)), "shots": int(len(np.unique(B.shot_lab[rows]))),
                        "sessions": int(len(np.unique(B.sess_lab[rows])))}, "stream": cell.name}
        for r in C.RULES:
            p = f"{r}|"
            o[r] = {"estimators": measures(cell, ests[r], rows, tgt, prefix=p),
                    "A over R": {f"{a} over {b}": cell.val(SK, f"{p}se_{a}", f"{p}se_{b}")
                                 for a, b in (("TX", "T0"), ("T0", "G"), ("T0", "H"), ("TX", "G"), ("TX", "H"),
                                              ("G", "H"), ("TX", "O"), ("T0", "O"))}}
        o["cost of staleness"] = {L: cell.val(RATIO, f"D100|se_{L}", f"D5|se_{L}") for L in ("H", "G", "T0", "TX")}
        o["recovered share D100"] = cell.val(REC, "D100|se_T0", "D100|se_TX", "D5|se_T0")
        o["TX over T0, D100 minus D5"] = cell.val(lambda a, b, c, e: (1 - a / b) - (1 - c / e),
                                                  "D100|se_TX", "D100|se_T0", "D5|se_TX", "D5|se_T0")
        out[d] = o
        C.log(f"{B.b} {d}: {len(rows)} rows done")
    return out


def value_of_frame(B):
    rows = B.scored("M1", "F")
    rows = rows[B.refs["formed"][rows]]
    tgt = B.R["y"][rows]
    est = B.estimates("M1", "D5", "F")
    cols = row_cols(est, rows, tgt)
    cell = Cell(f"{B.b}|M1|F|D5|value_of_frame", [part(rows, B.shot_lab, B.sess_lab, cols)])
    o = {"counts": {"rows": int(len(rows)), "shots": int(len(np.unique(B.shot_lab[rows]))),
                    "sessions": int(len(np.unique(B.sess_lab[rows]))),
                    "rows_left_out_no_reference": int(len(B.scored("M1", "F")) - len(rows)),
                    "offline_gp_fit_failed_fell_back_to_hold": int(B.refs["gp_fail"][rows].sum())},
         "stream": cell.name, "estimators": measures(cell, est, rows, tgt)}
    o["A over R"] = {f"{a} over {b}": cell.val(SK, f"se_{a}", f"se_{b}")
                     for a, b in (("TX", "PCHIP"), ("TX", "OGP"), ("T0", "PCHIP"), ("T0", "OGP"), ("G", "PCHIP"),
                                  ("G", "OGP"), ("H", "PCHIP"), ("OGP", "PCHIP"))}
    C.log(f"{B.b} value of the frame: {len(rows)} rows done")
    return o


# ------------------------------------------------------------------------------------------ statements
TEXT = {"R1": "M1, D100: TX over T0 lies above zero in band {b}",
        "R2": "M1, D5: TX over T0 lies above zero in band {b}",
        "R3": "M3, D100: TX over T0 lies above zero in band {b}",
        "R4": "M1, D5: T0 over G lies above zero in band {b} (the learned history-only estimate against the causal GP)",
        "R5": "M1: the cost of staleness of T0 at D100 (MSE at D100 over MSE at D5, same rows) is larger in band E than in "
              "the innermost band with statements",
        "R6": "M1, D100: the Thomson increment (TX over T0) in band E minus that in the innermost band with statements lies above zero",
        "R7": "M1, D5: the share of estimates of TX within 10 % of the level is larger in the innermost band with statements than in band E"}


def expect(sid, b):
    if sid == "R1":
        return "holds in every band"
    if sid == "R2":
        return "holds in E (B.76); uncertain inside" if b == "E" else "uncertain inside (holds in E)"
    if sid == "R3":
        return "holds in E (B.76)" if b == "E" else "uncertain, leaning to holds, inside"
    if sid == "R4":
        return "holds in every band"
    return {"R5": "holds (the edge changes faster)", "R6": "uncertain", "R7": "holds (the series is smoother inside)"}[sid]


def verdict_vs_expectation(holds, e):
    if e.startswith("holds"):
        return "met" if holds else "not met"
    return "n/a (uncertain)"


def statements(bands, inner):
    S = []

    def put(sid, b, text, v, note=""):
        holds = bool(v["session"][0] > 0)
        e = expect(sid, b)
        S.append({"id": sid if b is None else f"{sid}.{b}", "text": text, "band": b, "value": v["point"],
                  "session_interval": v["session"], "shot_interval": v["shot"], "stream": v["stream"],
                  "verdict": "holds" if holds else "does not hold", "holds": holds, "expectation": e,
                  "expectation_met": verdict_vs_expectation(holds, e), "judged": "lower end of the session interval above zero",
                  "note": note})

    for b, B in bands.items():
        if B is None:
            continue
        for sid, d, r, a, ref in (("R1", "M1", "D100", "TX", "T0"), ("R2", "M1", "D5", "TX", "T0"),
                                  ("R3", "M3", "D100", "TX", "T0"), ("R4", "M1", "D5", "T0", "G")):
            rows = B.scored(d, "L5")
            tgt = B.R["L5"][rows]
            est = B.estimates(d, r, "L5")
            cols = {"se_A": (est[a][rows] - tgt) ** 2, "se_R": (est[ref][rows] - tgt) ** 2}
            cell = Cell(f"{sid}.{b}", [part(rows, B.shot_lab, B.sess_lab, cols)])
            put(sid, b, TEXT[sid].format(b=b), cell.val(SK, "se_A", "se_R"),
                note=f"{a} over {ref}, {d}, {r}, target L5, {len(rows):,} rows")
    if inner is None or bands.get("E") is None:
        return S, None
    E, I = bands["E"], bands[inner]

    def m1_parts(r_list, est_keys, fn_cols):
        ps = []
        for tag, B in (("E", E), ("I", I)):
            rows = B.scored("M1", "L5")
            tgt = B.R["L5"][rows]
            cols = {}
            for r in r_list:
                est = B.estimates("M1", r, "L5")
                for k in est_keys:
                    cols.update(fn_cols(tag, r, k, est[k][rows] - tgt, tgt))
            ps.append(part(rows, B.shot_lab, B.sess_lab, cols))
        return ps

    se = lambda tag, r, k, e, t: {f"{tag}|{r}|se_{k}": e ** 2}   # noqa: E731
    cell = Cell("R5", m1_parts(("D5", "D100"), ("T0",), se))
    v = cell.val(lambda a, b, c, d: a / b - c / d, "E|D100|se_T0", "E|D5|se_T0", "I|D100|se_T0", "I|D5|se_T0")
    put("R5", None, TEXT["R5"], v, note=f"E minus {inner}; E {cell.val(RATIO, 'E|D100|se_T0', 'E|D5|se_T0')['point']:.3f}, "
                                        f"{inner} {cell.val(RATIO, 'I|D100|se_T0', 'I|D5|se_T0')['point']:.3f}")
    cell = Cell("R6", m1_parts(("D100",), ("T0", "TX"), se))
    v = cell.val(lambda a, b, c, d: (1 - a / b) - (1 - c / d), "E|D100|se_TX", "E|D100|se_T0", "I|D100|se_TX", "I|D100|se_T0")
    put("R6", None, TEXT["R6"], v, note=f"E minus {inner}")
    inn = lambda tag, r, k, e, t: {f"{tag}|in10": (np.abs(e) <= 0.10 * t).astype(np.float64), f"{tag}|n": np.ones(len(t))}  # noqa: E731
    cell = Cell("R7", m1_parts(("D5",), ("TX",), inn))
    v = cell.val(lambda a, b, c, d: a / b - c / d, "I|in10", "I|n", "E|in10", "E|n")
    put("R7", None, TEXT["R7"], v, note=f"{inner} minus E; {inner} {cell.val(RATIO, 'I|in10', 'I|n')['point']:.3f}, "
                                        f"E {cell.val(RATIO, 'E|in10', 'E|n')['point']:.3f}")
    return S, inner


# ------------------------------------------------------------------------------------------ main
def main():
    ok, which = C.checks_passed()
    if not ok:
        raise SystemExit(f"FATAL: check {which} has not passed; no readout")
    t0 = time.time()
    si = C.sets_info()
    with_st = C.bands_with_statements()
    inner = next((b for b in ("B5", "B4", "B3", "B2") if b in with_st), None)
    res = {"registration": C.REG, "addendum": C.ADD, "n_boot": N, "seed": C.SEED, "draw_order": list(FORMS),
           "bands_with_statements": with_st, "innermost_band_with_statements": inner,
           "counts": {}, "shots": json.loads((C.CACHE / "shots.json").read_text(encoding="utf-8"))["info"],
           "refs_info": json.loads((C.CACHE / "refs_info.json").read_text(encoding="utf-8")),
           "bands": {}, "value_of_frame": {}}
    for b in C.BAND_NAMES:
        x = si["bands"][b]
        res["counts"][b] = {"distance_cm": [100 * C.BANDS[b][0], 100 * C.BANDS[b][1]],
                            "candidate_rows": res["shots"]["bands"][b]["rows_candidate"],
                            "complete_D5_and_D100": x["complete_both"],
                            "L5_common": x["common_L5"], "F_common": x["common_F"],
                            "design_shots": x["design_shots"], "design_sessions": x["design_sessions"],
                            "statements": x["statements"]}
    bands = {}
    for b in C.BAND_NAMES:
        B = Band(b)
        res["bands"][b] = band_tables(B)
        res["value_of_frame"][b] = value_of_frame(B)
        bands[b] = B if b in with_st else None
    st, _ = statements(bands, inner)
    res["statements"] = st
    res["checks"] = {n: json.loads((C.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("band_e", "visibility", "references", "rows")}
    res["runtime_readout_s"] = round(time.time() - t0, 1)
    C.save_json(C.OUT_JSON, res)
    spec =importlib.util.spec_from_file_location("render86", __import__("pathlib").Path(__file__).with_name("render86.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    C.save_text(C.OUT_MD, mod.render(res))
    C.log(f"written {C.OUT_JSON.name}, {C.OUT_MD.name} ({res['runtime_readout_s']} s)")


if __name__ == "__main__":
    main()
