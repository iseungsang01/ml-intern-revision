# Check 8: the fast-channel gain against the level of the next five measured frames, by row class and by temperature third,
# and with the correction shrunk by a factor fitted on half of the discharges. Stored D1 estimates of B.70; read-only.
import sys, os, re, numpy as np, warnings
warnings.simplefilter("ignore")
allnpz, repo, rep = sys.argv[1], sys.argv[2], sys.argv[3]
P = lambda *a: print(*a, flush=True)
d = np.load(allnpz, allow_pickle=True); X, shot = d["X"], d["shot"]
t_all = X[:, 0]; ti_all = X[:, 1].copy(); ti_all[ti_all > 3000] = np.nan
us, first = np.unique(shot, return_index=True); cnt = np.bincount(np.searchsorted(us, shot))
off = {int(s): (a, n) for s, a, n in zip(us, first, cnt)}
names = np.array(sorted(f for f in os.listdir(os.path.join(repo, "data")) if re.fullmatch(r"s\d{5}\.csv", f)))
W = 5; rows = []
for k in range(5):
    z = np.load(os.path.join(repo, "data", ".b70", f"stage2_{rep}_f{k}.npz"), allow_pickle=True)
    fidx, grow, y = z["fidx"], z["grow"], z["y"]
    c, c2, tw, tw2 = z["est_c"], z["est_c2"], z["est_t"], z["est_t2"]
    for j in np.unique(fidx):
        s = int(names[j][1:6]); a, n = off[s]; q = np.flatnonzero(fidx == j); g = grow[q] - 1
        T = ti_all[a:a + n]; tt = t_all[a:a + n]
        assert np.all(np.abs(T[g] - y[q]) < 1e-2)
        dtt = np.r_[np.nan, np.diff(tt)]; bl = (dtt > 0.015) & (dtt < 0.045)
        since = np.full(n, 99); kk = 99
        for i in range(n):
            if bl[i]: kk = 0
            since[i] = kk; kk = min(kk + 1, 99)
        st = []
        for kk in range(1, W + 1):
            gi = g + kk; v = gi < n; val = np.full(g.size, np.nan); good = v.copy()
            good[v] &= np.abs((tt[gi[v]] - tt[g[v]]) - 0.01 * kk) < 1e-4
            val[good] = T[gi[good]]; st.append(val)
        st = np.vstack(st); mf = np.where(np.isfinite(st).sum(0) >= 3, np.nanmedian(st, 0), np.nan)
        medT = np.nanmedian(T)
        rows.append(np.c_[np.full(g.size, s), y[q], mf, c[q], tw[q], c2[q], tw2[q], since[g], np.full(g.size, medT)])
R = np.vstack(rows); sh, y, mf, c, tw, c2, tw2, since, medT = R.T
ok = np.isfinite(mf) & np.isfinite(c) & np.isfinite(tw) & np.isfinite(c2) & np.isfinite(tw2)
P("re-partition", rep, ": rows with a level of the next five frames:", ok.sum(), "of", len(y), "; discharges", np.unique(sh[ok]).size)
rng = np.random.default_rng(13)
def skill(a, b, z, sel, n=1000):
    ush = np.unique(sh[sel]); inv = np.searchsorted(ush, sh[sel]); K = ush.size
    num = np.bincount(inv, weights=((b - z) ** 2 - (a - z) ** 2)[sel], minlength=K); den = np.bincount(inv, weights=((b - z) ** 2)[sel], minlength=K)
    bs = [num[p].sum() / den[p].sum() for p in (rng.integers(0, K, K) for _ in range(n))]
    return num.sum() / den.sum(), np.percentile(bs, [2.5, 97.5]), K
umed = np.array([np.nanmedian(medT[sh == s]) for s in np.unique(sh[ok])]); q1, q2 = np.percentile(umed, [100 / 3, 200 / 3])
classes = (("all rows", ok), ("rows 1 to 10 after an absent frame", ok & (since >= 0) & (since <= 9)), ("other rows", ok & (since > 9)),
           ("coldest third of the discharges", ok & (medT <= q1)), ("middle third", ok & (medT > q1) & (medT <= q2)), ("hottest third", ok & (medT > q2)))
for lab, sel in classes:
    out = []
    for nm, a, b in (("network over twin", c, tw), ("two-stage over two-stage twin", c2, tw2)):
        s1, ci1, K = skill(a, b, y, sel); s2, ci2, _ = skill(a, b, mf, sel)
        out.append("%s: frame value %+.3f [%+.3f, %+.3f], level of the next five %+.3f [%+.3f, %+.3f]" % (nm, s1, ci1[0], ci1[1], s2, ci2[0], ci2[1]))
    P("%-36s n=%6d shots=%3d | %s | %s" % (lab, sel.sum(), K, out[0], out[1]))
# shrink factor fitted on half of the discharges, applied to the other half (two-fold, by discharge parity of a random permutation)
ush = np.unique(sh[ok]); perm = rng.permutation(ush.size); half = {s: perm[i] % 2 for i, s in enumerate(ush)}; hv = np.array([half.get(s, 0) for s in sh])
for nm, a, b in (("network over twin", c, tw), ("two-stage over two-stage twin", c2, tw2)):
    est = np.full(len(y), np.nan); lam = []
    for h in (0, 1):
        fit = ok & (hv != h); app = ok & (hv == h); dd = (a - b)
        l = np.sum(dd[fit] * (mf[fit] - b[fit])) / np.sum(dd[fit] ** 2); lam.append(l); est[app] = b[app] + l * dd[app]
    s2, ci2, _ = skill(est, b, mf, ok); s1, ci1, _ = skill(est, b, y, ok)
    P("%s with the correction multiplied by a factor fitted for the level on the other half of the discharges (%.3f, %.3f): level of the next five %+.3f [%+.3f, %+.3f]; frame value %+.3f [%+.3f, %+.3f]" % (nm, lam[0], lam[1], s2, ci2[0], ci2[1], s1, ci1[0], ci1[1]))
