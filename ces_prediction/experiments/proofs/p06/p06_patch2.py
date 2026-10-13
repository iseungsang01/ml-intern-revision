p='p06_mi_bound.py'; s=open(p,encoding='utf-8').read()
# collect deeper history levels (lags 2,3,5) and the persistence squared error per row
s=s.replace("            rows_prev.append(yb[idx - 1][good] - m)\n",
"""            rows_prev.append(yb[idx - 1][good] - m)
            HL = (2, 3, 5)
            idx_h = np.clip(idx[:, None] - np.array(HL)[None, :], 0, None)
            hl = np.stack([yb[idx_h[:, k]] for k in range(len(HL))], axis=1) - m
            hl[(idx[:, None] - np.array(HL)[None, :]) < 0] = np.nan
            rows_hist.append(hl[good])
            rows_pers.append((yb[idx] - yb[idx - 1])[good])
""")
s=s.replace("    rows_lag = []\n","    rows_lag = []\n    rows_hist = []\n    rows_pers = []\n")
s=s.replace("    ZL = np.vstack(rows_lag)\n","    ZL = np.vstack(rows_lag)\n    HIST = np.vstack(rows_hist)\n    PERS = np.concatenate(rows_pers)\n")
s=s.replace("""    # ---- conversion to skill-vs-persistence units ----""",
"""    # ---- (b'') partial R^2 given a DEEPER history (prev levels at lags 1,2,3,5 = 50 ms), and an
    #      EMPIRICAL conversion: skill vs persistence measured on the same pooled rows ----
    print("")
    print("(b'') history = previous levels at lags 1,2,3,5 (rows with all four observed); empirical persistence MSE on the same rows")
    hmask = np.isfinite(HIST).all(axis=1)
    H5 = np.column_stack([prev, HIST])[hmask]
    e5, odd5 = e[hmask], odd[hmask]
    mse_pers5 = float(PERS[hmask] @ PERS[hmask] / hmask.sum())
    deep = {"n": int(hmask.sum()), "mse_pers_same_rows": mse_pers5}
    print("  rows %d  persistence MSE on these rows = %.0f eV^2 (RMSE %.1f)" % (hmask.sum(), mse_pers5, np.sqrt(mse_pers5)))
    for nm, X in (("none", None), ("mod3", X3), ("ch15", X15), ("ch15+inc15", X30), ("ch15 lags0-5(75)", X75)):
        Xf = H5 if X is None else np.hstack([H5, X[hmask]])
        def sse(tr, te):
            _, _, b = r2_of_fit(Xf[tr], e5[tr], Xte=Xf[te], yte=e5[te])
            r = e5[te] - np.column_stack([np.ones(te.sum()), Xf[te]]) @ b
            return float(r @ r), int(te.sum())
        s_oe, n_e = sse(odd5, ~odd5); s_eo, n_o = sse(~odd5, odd5)
        mse_oos = (s_oe + s_eo) / (n_e + n_o)
        allm = np.ones(e5.size, bool); s_in, _ = sse(allm, allm)
        mse_in = s_in / e5.size
        deep[nm] = {"mse_in": mse_in, "mse_oos": mse_oos, "skill_vs_pers_in": 1 - mse_in / mse_pers5,
                    "skill_vs_pers_oos": 1 - mse_oos / mse_pers5}
        print("  h5 + %-18s MSE_in=%.0f MSE_oos=%.0f  skill vs pers (in/oos) = %.4f / %.4f"
              % (nm, mse_in, mse_oos, deep[nm]["skill_vs_pers_in"], deep[nm]["skill_vs_pers_oos"]))
    for nm in ("mod3", "ch15", "ch15+inc15", "ch15 lags0-5(75)"):
        d_in = deep[nm]["skill_vs_pers_in"] - deep["none"]["skill_vs_pers_in"]
        d_oos = deep[nm]["skill_vs_pers_oos"] - deep["none"]["skill_vs_pers_oos"]
        pr_in = 1 - deep[nm]["mse_in"] / deep["none"]["mse_in"]; pr_oos = 1 - deep[nm]["mse_oos"] / deep["none"]["mse_oos"]
        deep[nm].update({"dskill_vs_pers_in": d_in, "dskill_vs_pers_oos": d_oos, "r2_partial_in": pr_in, "r2_partial_oos": pr_oos})
        print("  fast adds over h5: %-18s partial R2 in/oos = %.4f / %.4f  -> dskill vs pers (empirical) = %+.4f / %+.4f"
              % (nm, pr_in, pr_oos, d_in, d_oos))
    out["deep_history_h5"] = deep

    # ---- conversion to skill-vs-persistence units ----""")
open(p,'w',encoding='utf-8').write(s)
import ast; ast.parse(s); print("patched ok")
