"""B.25 verdict (PREREGISTRATION_B25.md) over data/.b25_rows.npz -- no model, no training.

Part A (TEST): low-k bins = the lowest 3 deciles of the fusion checkpoint's gain k per split and
target; R = median|k r|_fusion / median|r|_additive (H25-a: R > 1), median |yhat - y| fusion >
additive (H25-b), and the same for |a - y| (the named follow-up when H25-a fails).
Part B (VAL): inclusive strata F (raw T_i > 3000 eV) / N; H26-a share of the MSE-selected
checkpoint's T_i squared error in F > 0.5; H26-b N median |err| medae < mse and F MSE mse < medae;
cut strata E (top 10 pct |y - prev|) / Q, H26-c E MSE mse < medae. Each '>= 3/4 splits'.
Output: data/.b25_verdict.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/verdict_b25.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b16 import DATA, POPS, SEEDS  # noqa: E402

TN = ("CES_TI", "CES_VT")
F_EV = 3000.0


def load():
    z = np.load(DATA / ".b25_rows.npz")
    rows = {}
    for k in z.files:
        part, pop, s, arm, t, f = k.split("|")
        rows.setdefault((part, pop, int(s), arm, int(t)), {})[f] = z[k]
    return rows


def same_rows(r1, r2):
    return all(np.array_equal(r1[f], r2[f]) for f in ("y", "prev", "file"))


def part_a(rows):
    out = {}
    for pop in POPS:
        for t, tn in enumerate(TN):
            per = []
            for s in SEEDS:
                fu, ad = rows[("A", pop, s, "fusion", t)], rows[("A", pop, s, "additive", t)]
                if not same_rows(fu, ad):
                    raise SystemExit(f"FATAL: part A rows differ between arms ({pop} s{s} {tn})")
                k = fu["g"]
                low = k <= np.quantile(k, 0.3)
                c_f, c_a = np.abs(fu["yhat"] - fu["a"]), np.abs(ad["yhat"] - ad["a"])
                e_f, e_a = np.abs(fu["yhat"] - fu["y"]), np.abs(ad["yhat"] - ad["y"])
                an_f, an_a = np.abs(fu["a"] - fu["y"]), np.abs(ad["a"] - ad["y"])
                dec = np.minimum((np.argsort(np.argsort(k)) * 10) // len(k), 9)
                per.append({
                    "n_rows": int(len(k)), "n_low": int(low.sum()),
                    "k_low_max": float(np.quantile(k, 0.3)), "k_median": float(np.median(k)),
                    "R_low": float(np.median(c_f[low]) / np.median(c_a[low])),
                    "contrib_low_fusion": float(np.median(c_f[low])), "contrib_low_additive": float(np.median(c_a[low])),
                    "err_low_fusion": float(np.median(e_f[low])), "err_low_additive": float(np.median(e_a[low])),
                    "anchor_err_low_fusion": float(np.median(an_f[low])), "anchor_err_low_additive": float(np.median(an_a[low])),
                    "R_by_decile": [float(np.median(c_f[dec == d]) / np.median(c_a[dec == d])) for d in range(10)],
                    "err_ratio_by_decile": [float(np.median(e_f[dec == d]) / np.median(e_a[dec == d])) for d in range(10)],
                })
            a = sum(p["R_low"] > 1 for p in per)
            b = sum(p["err_low_fusion"] > p["err_low_additive"] for p in per)
            out[f"{pop}|{tn}"] = {"per_seed": per, "H25-a": {"splits": a, "pass": a >= 3},
                                  "H25-b": {"splits": b, "pass": b >= 3}}
    return out


def part_b(rows):
    out = {}
    for pop in POPS:
        per = []
        for s in SEEDS:
            ms, md = rows[("B", pop, s, "mse_sel", 0)], rows[("B", pop, s, "medae_sel", 0)]
            if not same_rows(ms, md):
                raise SystemExit(f"FATAL: part B rows differ between arms ({pop} s{s})")
            se_m, se_d = (ms["yhat"] - ms["y"]) ** 2, (md["yhat"] - md["y"]) ** 2
            ae_m, ae_d = np.sqrt(se_m), np.sqrt(se_d)
            rec = {"n_rows": int(len(se_m))}
            if pop == "incl":
                F = ms["y"] > F_EV
                N = ~F
                rec.update({"n_F": int(F.sum()), "share_F_mse_sel": float(se_m[F].sum() / se_m.sum()),
                            "share_F_medae_sel": float(se_d[F].sum() / se_d.sum()),
                            "N_medae_err_mse_sel": float(np.median(ae_m[N])), "N_medae_err_medae_sel": float(np.median(ae_d[N])),
                            "F_mse_mse_sel": float(se_m[F].mean()) if F.any() else None,
                            "F_mse_medae_sel": float(se_d[F].mean()) if F.any() else None,
                            "N_mse_mse_sel": float(se_m[N].mean()), "N_mse_medae_sel": float(se_d[N].mean())})
            else:
                jump = np.abs(ms["y"] - ms["prev"])
                E = jump >= np.quantile(jump, 0.9)
                rec.update({"n_E": int(E.sum()), "E_mse_mse_sel": float(se_m[E].mean()), "E_mse_medae_sel": float(se_d[E].mean()),
                            "Q_medae_err_mse_sel": float(np.median(ae_m[~E])), "Q_medae_err_medae_sel": float(np.median(ae_d[~E])),
                            "share_E_mse_sel": float(se_m[E].sum() / se_m.sum())})
            per.append(rec)
        if pop == "incl":
            a = sum(p["share_F_mse_sel"] > 0.5 for p in per)
            b = sum(p["N_medae_err_medae_sel"] < p["N_medae_err_mse_sel"]
                    and p["F_mse_mse_sel"] is not None and p["F_mse_mse_sel"] < p["F_mse_medae_sel"] for p in per)
            out[pop] = {"per_seed": per, "H26-a": {"splits": a, "pass": a >= 3}, "H26-b": {"splits": b, "pass": b >= 3}}
        else:
            c = sum(p["E_mse_mse_sel"] < p["E_mse_medae_sel"] for p in per)
            out[pop] = {"per_seed": per, "H26-c": {"splits": c, "pass": c >= 3}}
    return out


def main():
    rows = load()
    A, Bv = part_a(rows), part_b(rows)
    rules = {}
    for key, v in A.items():
        rules[f"A|{key}"] = ("reading confirmed" if v["H25-a"]["pass"] and v["H25-b"]["pass"]
                             else "reading rejected -> compare |a - y| between arms" if not v["H25-a"]["pass"]
                             else "residual larger on low-k rows, median loss not there")
    rules["B|incl"] = ("MSE key picks checkpoints by fit-failure rows: confirmed"
                       if Bv["incl"]["H26-a"]["pass"] and Bv["incl"]["H26-b"]["pass"]
                       else "F does not dominate val MSE -> split N by |delta| deciles next" if not Bv["incl"]["H26-a"]["pass"]
                       else "F dominates, but the selection trade is not where the reading puts it")
    rules["B|cut"] = ("median key stops before the event rows: confirmed" if Bv["cut"]["H26-c"]["pass"]
                      else "medae selection does not lose the event rows")
    out = {"A": A, "B": Bv, "rules": rules}
    (DATA / ".b25_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, v in A.items():
        p = v["per_seed"]
        print(f"A {key:12s} R_low " + " ".join(f"{q['R_low']:.2f}" for q in p) + f" | H25-a {v['H25-a']['splits']}/4"
              + " | err_low f/a " + " ".join(f"{q['err_low_fusion']:.1f}/{q['err_low_additive']:.1f}" for q in p)
              + f" | H25-b {v['H25-b']['splits']}/4 | anchor err f/a "
              + " ".join(f"{q['anchor_err_low_fusion']:.1f}/{q['anchor_err_low_additive']:.1f}" for q in p))
    p = Bv["incl"]["per_seed"]
    print("B incl share_F(mse_sel) " + " ".join(f"{q['share_F_mse_sel']:.2f}" for q in p) + f" | H26-a {Bv['incl']['H26-a']['splits']}/4"
          + " | N medAE mse/medae " + " ".join(f"{q['N_medae_err_mse_sel']:.1f}/{q['N_medae_err_medae_sel']:.1f}" for q in p)
          + " | F MSE mse/medae " + " ".join(f"{q['F_mse_mse_sel']:.3g}/{q['F_mse_medae_sel']:.3g}" for q in p)
          + f" | H26-b {Bv['incl']['H26-b']['splits']}/4")
    p = Bv["cut"]["per_seed"]
    print("B cut  E MSE mse/medae " + " ".join(f"{q['E_mse_mse_sel']:.3g}/{q['E_mse_medae_sel']:.3g}" for q in p)
          + f" | H26-c {Bv['cut']['H26-c']['splits']}/4")
    print(json.dumps(rules, indent=1))
    print(f"wrote {DATA / '.b25_verdict.json'}")


if __name__ == "__main__":
    main()
