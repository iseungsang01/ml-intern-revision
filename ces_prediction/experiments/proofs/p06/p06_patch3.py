p='p06_mi_bound.py'; s=open(p,encoding='utf-8').read()
s=s.replace("""    out["deep_history_h5"] = deep
""","""    # tail decomposition of the h5-conditioned gain (joint fit, explained SSE share inside top-10% |e| rows)
    tail5 = tail[hmask]
    for nm, X in (("ch15+inc15", X30), ("ch15 lags0-5(75)", X75)):
        X0 = np.column_stack([np.ones(e5.size), H5]); X1 = np.column_stack([X0, X[hmask]])
        b0 = np.linalg.solve(X0.T @ X0 + 1e-8 * np.eye(X0.shape[1]), X0.T @ e5)
        b1 = np.linalg.solve(X1.T @ X1 + 1e-8 * np.eye(X1.shape[1]), X1.T @ e5)
        r0, r1 = e5 - X0 @ b0, e5 - X1 @ b1
        tot = r0 @ r0
        gain_tail = float((r0[tail5] @ r0[tail5] - r1[tail5] @ r1[tail5]) / (e5.size * mse_pers5))
        gain_rest = float((r0[~tail5] @ r0[~tail5] - r1[~tail5] @ r1[~tail5]) / (e5.size * mse_pers5))
        share_tail_null = float(r0[tail5] @ r0[tail5] / tot)
        deep[nm]["tail_decomp"] = {"dskill_tail": gain_tail, "dskill_rest": gain_rest, "null_sse_share_tail": share_tail_null,
                                   "r2_within_tail": float(1 - r1[tail5] @ r1[tail5] / (r0[tail5] @ r0[tail5])),
                                   "r2_within_rest": float(1 - r1[~tail5] @ r1[~tail5] / (r0[~tail5] @ r0[~tail5]))}
        print("  tail decomposition (h5 + %s): null SSE share in top-10%% |e| = %.3f ; dskill tail %+.4f + rest %+.4f ; R2 within tail %.4f / rest %.4f"
              % (nm, share_tail_null, gain_tail, gain_rest, deep[nm]["tail_decomp"]["r2_within_tail"], deep[nm]["tail_decomp"]["r2_within_rest"]))
    out["deep_history_h5"] = deep
""")
open(p,'w',encoding='utf-8').write(s)
import ast; ast.parse(s); print("patched ok")
