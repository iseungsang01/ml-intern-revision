# Check 7: is the fast-channel gain a gain on the sustained level of T_i or on the departure of the scored frame from its neighbours?
# Stored D1 estimates of B.70 (re-partition given on the command line), read-only. Nothing is trained or re-scored by a model.
import sys, os, re, numpy as np, warnings
warnings.simplefilter("ignore")
allnpz, repo, rep = sys.argv[1], sys.argv[2], sys.argv[3]
P = lambda *a: print(*a, flush=True)
d = np.load(allnpz, allow_pickle=True); X, shot = d["X"], d["shot"]
t_all = X[:, 0]; ti_all = X[:, 1].copy(); ti_all[ti_all > 3000] = np.nan      # cut population
bes_all = X[:, 3:12].mean(1)
off = {};
us, first = np.unique(shot, return_index=True)
cnt = np.bincount(np.searchsorted(us, shot))
for s, a, n in zip(us, first, cnt): off[int(s)] = (a, n)
W = 5
rows = []
for k in range(5):
    z = np.load(os.path.join(repo, "data", ".b70", f"stage2_{rep}_f{k}.npz"), allow_pickle=True)
    tf = np.array(sorted(f for f in os.listdir(os.path.join(repo, "data")) if re.fullmatch(r"s\d{5}\.csv", f)))
    fidx, grow, y = z["fidx"], z["grow"], z["y"]
    c, c2, tw, tw2 = z["est_c"], z["est_c2"], z["est_t"], z["est_t2"]
    # find the offset convention of grow by matching y to the file values
    best = None
    for o in (-1, 0, 1):
        ok = 0; tot = 0
        for j in np.unique(fidx)[:10]:
            s = int(tf[j][1:6]); a, n = off[s]; q = fidx == j; g = grow[q] + o
            v = (g >= 0) & (g < n); tot += q.sum(); ok += np.sum(np.abs(X[a + g[v], 1] - y[q][v]) < 1e-2)
        if best is None or ok > best[1]: best = (o, ok, tot)
    o = best[0]
    if k == 0: P("grid-row offset %d matches %d of %d checked rows" % best)
    for j in np.unique(fidx):
        s = int(tf[j][1:6]); a, n = off[s]; q = np.flatnonzero(fidx == j); g = grow[q] + o
        T = ti_all[a:a + n]; tt = t_all[a:a + n]
        # two-sided reference level: median of the measured values of the five rows before and five after (10 ms steps), frame itself excluded
        st = []
        for kk in list(range(-W, 0)) + list(range(1, W + 1)):
            gi = g + kk; v = (gi >= 0) & (gi < n); val = np.full(g.size, np.nan)
            good = v & (g >= 0) & (g < n); good[good] &= np.abs((tt[gi[good]] - tt[g[good]]) - 0.01 * kk) < 1e-4
            val[good] = T[gi[good]]; st.append(val)
        st = np.vstack(st); nb = np.isfinite(st[:W]).sum(0); na = np.isfinite(st[W:]).sum(0)
        m = np.where((nb >= 3) & (na >= 3), np.nanmedian(st, 0), np.nan)
        mf = np.where(na >= 3, np.nanmedian(st[W:], 0), np.nan); mb = np.where(nb >= 3, np.nanmedian(st[:W], 0), np.nan); ynext = st[W]
        # causal comparators from the file: hold (last measured value before the row) and the median of the last five measured values
        hold = np.full(g.size, np.nan); med5 = np.full(g.size, np.nan)
        fin = np.isfinite(T); idxf = np.flatnonzero(fin)
        pos = np.searchsorted(idxf, g) - 1
        okp = pos >= 0; hold[okp] = T[idxf[pos[okp]]]
        ok5 = pos >= 4
        if ok5.any():
            med5[ok5] = np.median(np.vstack([T[idxf[pos[ok5] - r]] for r in range(5)]), 0)
        rows.append(np.c_[np.full(g.size, s), y[q], m, c[q], tw[q], c2[q], tw2[q], hold, med5, np.abs(T[np.clip(g, 0, n - 1)] - y[q]), mf, mb, ynext])
R = np.vstack(rows)
sh, y, m, c, tw, c2, tw2, hold, med5, chk, mf, mb, ynext = R.T
P("D1 rows read:", len(y), "; rows whose stored y equals the file value:", int(np.sum(chk < 1e-3)))
def sk(a, b, target, sel): return 1 - np.sum((a[sel] - target[sel]) ** 2) / np.sum((b[sel] - target[sel]) ** 2)
allr = np.isfinite(y) & np.isfinite(c) & np.isfinite(tw) & np.isfinite(c2) & np.isfinite(tw2) & np.isfinite(hold)
P("check against the record (all D1 rows, target = reported value): network over twin %+.3f ; two-stage over two-stage twin %+.3f ; two-stage over network %+.3f ; network over hold %+.3f" % (sk(c, tw, y, allr), sk(c2, tw2, y, allr), sk(c2, c, y, allr), sk(c, hold, y, allr)))
sel = allr & np.isfinite(m) & np.isfinite(med5)
P("rows with a two-sided reference level (at least three measured neighbours on each side): %d (%.3f of the D1 rows), discharges %d" % (sel.sum(), sel.sum() / allr.sum(), np.unique(sh[sel]).size))
e = y - m
P("departure of the scored frame from its neighbours, e = y - m: robust sd %.1f eV, rms %.1f eV ; rms of y - network %.1f eV" % (1.4826 * np.median(np.abs(e[sel] - np.median(e[sel]))), np.sqrt(np.mean(e[sel] ** 2)), np.sqrt(np.mean((y[sel] - c[sel]) ** 2))))
P("share of the squared error of each estimator on these rows that is the departure itself, sum e^2 / sum (y - est)^2: network %.3f ; twin %.3f ; two-stage %.3f ; hold %.3f" % tuple(np.sum(e[sel] ** 2) / np.sum((y[sel] - a[sel]) ** 2) for a in (c, tw, c2, hold)))
rng = np.random.default_rng(11); ush = np.unique(sh[sel]); inv = np.searchsorted(ush, sh[sel]); K = ush.size
def boot(f, n=1000):
    # f takes per-discharge sums; returns statistic
    out = []
    for _ in range(n):
        pick = rng.integers(0, K, K); out.append(f(pick))
    return np.percentile(out, [2.5, 97.5])
def per(v): return np.bincount(inv, weights=v[sel], minlength=K)
for name, a, b in (("network over twin", c, tw), ("two-stage over two-stage twin", c2, tw2), ("two-stage over network", c2, c), ("network over hold", c, hold), ("network over median of last five", c, med5), ("two-stage over median of last five", c2, med5), ("twin over median of last five", tw, med5)):
    ny = per((b - y) ** 2 - (a - y) ** 2); dy = per((b - y) ** 2)
    nm = per((b - m) ** 2 - (a - m) ** 2); dm = per((b - m) ** 2)
    cov = per(2 * e * (a - b))
    s_y = ny.sum() / dy.sum(); s_m = nm.sum() / dm.sum(); share = cov.sum() / ny.sum()
    ci_y = boot(lambda p: ny[p].sum() / dy[p].sum()); ci_m = boot(lambda p: nm[p].sum() / dm[p].sum()); ci_s = boot(lambda p: cov[p].sum() / ny[p].sum())
    P("%-36s skill against the reported value %+.3f [%+.3f, %+.3f] ; against the level of the neighbours %+.3f [%+.3f, %+.3f] ; share of the reduction that is covariance with the departure e: %.3f [%.3f, %.3f]" % (name, s_y, ci_y[0], ci_y[1], s_m, ci_m[0], ci_m[1], share, ci_s[0], ci_s[1]))
# regression of the fast-channel correction on the two parts of the twin's error
dcor = c - tw; lev = m - tw
A = np.c_[e[sel], lev[sel], np.ones(sel.sum())]; co = np.linalg.lstsq(A, dcor[sel], rcond=None)[0]
P("fast-channel correction (network - twin) regressed on [departure e, level error of the twin m - twin]: coefficients %+.3f, %+.3f ; rms of the correction %.1f eV" % (co[0], co[1], np.sqrt(np.mean(dcor[sel] ** 2))))
dcor2 = c2 - tw2; lev2 = m - tw2
A = np.c_[e[sel], lev2[sel], np.ones(sel.sum())]; co = np.linalg.lstsq(A, dcor2[sel], rcond=None)[0]
P("the same for the two-stage pair: coefficients %+.3f, %+.3f ; rms of the correction %.1f eV" % (co[0], co[1], np.sqrt(np.mean(dcor2[sel] ** 2))))
# rmse and median error against the level of the neighbours
for name, a in (("hold", hold), ("median of last five", med5), ("twin", tw), ("network", c), ("two-stage twin", tw2), ("two-stage", c2)):
    P("   %-22s against reported: rmse %.1f median |e| %.1f ; against the level of the neighbours: rmse %.1f median |e| %.1f" % (name, np.sqrt(np.mean((a[sel] - y[sel]) ** 2)), np.median(np.abs(a[sel] - y[sel])), np.sqrt(np.mean((a[sel] - m[sel]) ** 2)), np.median(np.abs(a[sel] - m[sel]))))

P("=== forward level (median of the next five measured frames) and regression coefficients of the correction ===")
sel = allr & np.isfinite(m) & np.isfinite(med5) & np.isfinite(mf) & np.isfinite(mb)
ush = np.unique(sh[sel]); inv = np.searchsorted(ush, sh[sel]); K = ush.size
P("rows %d discharges %d" % (sel.sum(), K))
for tname, z in (("reported value of the frame", y), ("two-sided level", m), ("level of the next five frames", mf), ("level of the five frames before", mb)):
    for name, a, b in (("network over twin", c, tw), ("two-stage over two-stage twin", c2, tw2), ("network over median of last five", c, med5), ("two-stage over median of last five", c2, med5), ("twin over median of last five", tw, med5)):
        nn = per((b - z) ** 2 - (a - z) ** 2); dd = per((b - z) ** 2); ci = boot(lambda pk: nn[pk].sum() / dd[pk].sum())
        P("   target = %-34s %-36s skill %+.3f [%+.3f, %+.3f]" % (tname, name, nn.sum() / dd.sum(), ci[0], ci[1]))
for name, a, b in (("network - twin", c, tw), ("two-stage - two-stage twin", c2, tw2)):
    dd_ = (a - b)
    sdd = per(dd_ ** 2)
    out = []
    for tname, z in (("y(t)", y), ("two-sided level", m), ("next five", mf), ("before five", mb)):
        num = per(dd_ * (z - b)); ci = boot(lambda pk: num[pk].sum() / sdd[pk].sum()); out.append("%s %.3f [%.3f, %.3f]" % (tname, num.sum() / sdd.sum(), ci[0], ci[1]))
    sely = sel & np.isfinite(ynext); 
    num1 = np.sum(dd_[sely] * (ynext[sely] - b[sely])) / np.sum(dd_[sely] ** 2)
    P("regression coefficient of (target - reference) on the correction %s: %s ; y(t+1) %.3f (a coefficient below 0.5 means the correction raises the squared error against that target)" % (name, " ; ".join(out), num1))
for name, a in (("hold", hold), ("median of last five", med5), ("twin", tw), ("network", c), ("two-stage twin", tw2), ("two-stage", c2)):
    P("   %-22s against the level of the next five frames: rmse %.1f median |e| %.1f" % (name, np.sqrt(np.mean((a[sel] - mf[sel]) ** 2)), np.median(np.abs(a[sel] - mf[sel]))))
