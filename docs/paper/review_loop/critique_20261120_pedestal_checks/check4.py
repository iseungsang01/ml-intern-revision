# Checks A, B, C, F on the 10 ms files (numpy only; read-only).
import sys, numpy as np, warnings
warnings.simplefilter("ignore")
d = np.load(sys.argv[1], allow_pickle=True); X, shot = d["X"], d["shot"]
t = X[:, 0]; ti_raw = X[:, 1]; vt = X[:, 2]; bes = X[:, 3:12]; ecei = X[:, 12:16]
ti = ti_raw.copy(); ti[ti > 3000] = np.nan
N = t.size
new_shot = np.r_[True, shot[1:] != shot[:-1]]
dt = np.r_[np.nan, np.diff(t)]; dt[new_shot] = np.nan
lvl = bes.mean(1)
blip1 = np.abs(dt - 0.020) < 1e-4
since = np.full(N, 99); k = 99
anyblip = (dt > 0.015) & (dt < 0.045)
for i in range(N):
    if new_shot[i]: k = 99
    if anyblip[i]: k = 0
    since[i] = k; k = min(k + 1, 99)
pvt = np.r_[np.nan, vt[:-1]]; pvt[new_shot] = np.nan
vf = vt.copy(); vf[np.isfinite(vt) & (vt == pvt)] = np.nan
P = lambda *a: print(*a, flush=True)

P("=== A. The beam restart as a natural experiment (single absent frame; six 10 ms rows before) ===")
ev = []
for i in np.flatnonzero(blip1):
    if i < 7 or i + 3 >= N or shot[i - 7] != shot[i] or shot[i + 3] != shot[i]: continue
    if not np.allclose(np.diff(t[i - 7:i]), 0.01, atol=1e-4): continue
    Tp = ti[i - 6:i]
    if np.isfinite(Tp).sum() < 3: continue
    vpre = vf[i - 6:i]
    V0 = np.nanmedian(vpre) if np.isfinite(vpre).sum() >= 2 else np.nan
    ev.append([shot[i], np.nanmedian(Tp), np.median(lvl[i - 6:i]), V0, np.median(ecei[i - 6:i].mean(1)),
               ti[i], lvl[i], vf[i], ecei[i].mean(), ti[i + 1], lvl[i + 1], ti[i + 2], lvl[i + 2]])
E = np.array(ev); P("events", len(E), "discharges", np.unique(E[:, 0]).size)
rT = E[:, 5] / E[:, 1]; rB = E[:, 6] / E[:, 2]
rV = np.where(np.abs(E[:, 3]) > 5, E[:, 7] / E[:, 3], np.nan)
rE = np.where(E[:, 4] > 0.02, E[:, 8] / E[:, 4], np.nan)
ok = np.isfinite(rT)
P("first row: T_i reported in", ok.sum(), "of", len(E), "events; share missing or above 3 keV %.3f" % (1 - ok.mean()))
P("median ratios, first row: T_i %.3f  BES %.3f  V_rot(fresh) %.3f (n=%d)  ECEI %.3f" % (np.nanmedian(rT), np.median(rB), np.nanmedian(rV), np.isfinite(rV).sum(), np.nanmedian(rE)))
P("quantiles of the first-row T_i ratio (5,10,25,50,75,90,95):", np.round(np.nanpercentile(rT, [5, 10, 25, 50, 75, 90, 95]), 3))
P("share of events with ratio < 0.5: %.3f ; < 0.3: %.3f ; > 1.0: %.3f" % (np.mean(rT[ok] < 0.5), np.mean(rT[ok] < 0.3), np.mean(rT[ok] > 1.0)))
sl = np.log(rT[ok]) / np.log(rB[ok])
P("implied log-slope ln(T ratio)/ln(BES ratio): median %.3f, quartiles [%.3f, %.3f]; from the two medians %.3f" % (np.median(sl), np.percentile(sl, 25), np.percentile(sl, 75), np.log(np.nanmedian(rT)) / np.log(np.median(rB))))
okv = np.isfinite(rV) & (rV > 0)
P("same for V_rot, from the two medians: %.3f (n=%d)" % (np.log(np.nanmedian(rV[okv])) / np.log(np.median(rB[okv])), okv.sum()))
r2 = E[:, 9] / E[:, 1]; b2 = E[:, 10] / E[:, 2]; r3 = E[:, 11] / E[:, 1]; b3 = E[:, 12] / E[:, 2]
P("rows 2 and 3: T_i ratio %.3f, %.3f ; BES ratio %.3f, %.3f ; T_i predicted by BES^0.5: %.3f, %.3f" % (np.nanmedian(r2), np.nanmedian(r3), np.median(b2), np.median(b3), np.median(b2) ** 0.5, np.median(b3) ** 0.5))
hot = ok & (E[:, 1] >= 400)
P("events with level before >= 400 eV:", hot.sum(), "; first-row T_i (eV) quantiles 5,10,25,50,75:", np.round(np.percentile(E[hot, 5], [5, 10, 25, 50, 75]), 0),
  "; share below 150 eV: %.3f ; below 200 eV: %.3f" % (np.mean(E[hot, 5] < 150), np.mean(E[hot, 5] < 200)))
idx = np.flatnonzero((since > 30) & np.isfinite(ti)); rng = np.random.default_rng(5); pick = rng.choice(idx, 20000, replace=False); c = []
for i in pick:
    if i < 7 or shot[i - 7] != shot[i] or not np.allclose(np.diff(t[i - 7:i + 1]), 0.01, atol=1e-4): continue
    Tp = ti[i - 6:i]
    if np.isfinite(Tp).sum() < 3: continue
    c.append([np.nanmedian(Tp), ti[i]])
c = np.array(c); hc = c[:, 0] >= 400
P("control rows far from blips (level before >= 400 eV, n=%d): share below 150 eV %.4f ; below 200 eV %.4f ; ratio < 0.5: %.4f ; median ratio %.3f" % (hc.sum(), np.mean(c[hc, 1] < 150), np.mean(c[hc, 1] < 200), np.mean(c[hc, 1] / c[hc, 0] < 0.5), np.median(c[hc, 1] / c[hc, 0])))

P("=== B. Departures of a frame from its ten neighbours, rows far from blips ===")
W = 5
def centred_median(a, need):
    st = []
    for kk in list(range(-W, 0)) + list(range(1, W + 1)):
        sh = np.roll(a, -kk); bad = (np.roll(shot, -kk) != shot) | (np.abs((np.roll(t, -kk) - t) - 0.01 * kk) > 1e-4); sh = sh.copy(); sh[bad] = np.nan; st.append(sh)
    st = np.vstack(st); return np.where(np.isfinite(st).sum(0) >= need, np.nanmedian(st, 0), np.nan)
lb = np.log(np.clip(lvl, 1e-6, None))
r = np.log(ti) - centred_median(np.log(ti), 8); rb = lb - centred_median(lb, 8)
far = np.ones(N, bool)
for kk in range(-6, 12): far &= ~np.roll(anyblip, kk)
m = far & np.isfinite(r) & np.isfinite(rb)
P("rows", m.sum(), "; robust sd of the ln T_i residual %.3f ; of the ln BES residual %.3f" % (1.4826 * np.median(np.abs(r[m])), 1.4826 * np.median(np.abs(rb[m]))))
for lab, sel in (("T_i more than 30 % below its neighbours", r < np.log(0.7)), ("T_i more than 30 % above", r > np.log(1.3)), ("T_i within 10 %", np.abs(r) < np.log(1.1))):
    s = m & sel
    P("  %-42s n=%6d (%.3f of rows)  median BES residual %+.3f  share with BES more than 10 %% below: %.3f  above: %.3f" % (lab, s.sum(), s.sum() / m.sum(), np.median(rb[s]), np.mean(rb[s] < np.log(0.9)), np.mean(rb[s] > np.log(1.1))))
for lab, sel in (("BES more than 15 % below its neighbours", rb < np.log(0.85)), ("BES more than 15 % above", rb > np.log(1.15)), ("BES within 5 %", np.abs(rb) < np.log(1.05))):
    s = m & sel
    P("  %-42s n=%6d (%.3f of rows)  median T_i residual %+.3f  share T_i more than 30 %% below: %.3f  above: %.3f" % (lab, s.sum(), s.sum() / m.sum(), np.median(r[s]), np.mean(r[s] < np.log(0.7)), np.mean(r[s] > np.log(1.3))))
mm = far & np.isfinite(rb)
for lab, sel in (("BES more than 15 % below", rb < np.log(0.85)), ("BES within 5 %", np.abs(rb) < np.log(1.05)), ("BES more than 15 % above", rb > np.log(1.15))):
    s = mm & sel
    P("  T_i missing or above 3 keV where %-26s: %.3f (n=%d)" % (lab, np.mean(~np.isfinite(ti[s])), s.sum()))

P("=== C. What the CES history alone gives at the 100 ms scale of the cited requirement ===")
brk = new_shot | ~(np.abs(dt - 0.010) < 1e-4); run = np.cumsum(brk)
starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]]); ends = np.r_[starts[1:], N]
prev_b = []; cur_b = []; shb = []; hold_last = []
for a, b in zip(starts, ends):
    n = (b - a) // 10
    if n < 2: continue
    y = ti[a:a + 10 * n].reshape(n, 10); cnt = np.isfinite(y).sum(1); mu = np.where(cnt >= 7, np.nanmean(y, 1), np.nan)
    last = np.array([row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.nan for row in y])
    prev_b += list(mu[:-1]); cur_b += list(mu[1:]); shb += [shot[a]] * (n - 1); hold_last += list(last[:-1])
prev_b, cur_b, hold_last = map(np.array, (prev_b, cur_b, hold_last)); okb = np.isfinite(prev_b) & np.isfinite(cur_b) & np.isfinite(hold_last)
rel = np.abs(prev_b[okb] - cur_b[okb]) / cur_b[okb]; rel2 = np.abs(hold_last[okb] - cur_b[okb]) / cur_b[okb]
P("pairs of adjacent 100 ms blocks:", okb.sum(), "discharges", np.unique(np.array(shb)[okb]).size)
P("previous 100 ms mean as the estimate of the next 100 ms mean: within 10 %%: %.3f ; within 20 %%: %.3f ; median relative error %.3f" % (np.mean(rel < 0.1), np.mean(rel < 0.2), np.median(rel)))
P("last single frame as the estimate of the next 100 ms mean:     within 10 %%: %.3f ; within 20 %%: %.3f ; median relative error %.3f" % (np.mean(rel2 < 0.1), np.mean(rel2 < 0.2), np.median(rel2)))
pti = np.r_[np.nan, ti[:-1]]; pti[brk] = np.nan; o = np.isfinite(ti) & np.isfinite(pti); rr = np.abs(pti[o] - ti[o]) / ti[o]
P("frame scale, previous frame as the estimate of this frame:      within 10 %%: %.3f ; within 20 %%: %.3f ; median relative error %.3f (n=%d)" % (np.mean(rr < 0.1), np.mean(rr < 0.2), np.median(rr), o.sum()))

P("=== F. Rows with the BES-viewed beam essentially off that still carry a T_i ===")
medl = np.zeros(N); medt = np.zeros(N)
for s in np.unique(shot):
    q = shot == s; medl[q] = np.median(lvl[q]); medt[q] = np.nanmedian(ti[q]) if np.isfinite(ti[q]).any() else np.nan
low = lvl < 0.25 * medl
P("rows with BES level below a quarter of the discharge median: %d (%.4f of rows); with a T_i value: %d (%.3f of them); with T_i <= 3 keV: %d" % (low.sum(), low.mean(), np.isfinite(ti_raw[low]).sum(), np.mean(np.isfinite(ti_raw[low])), np.isfinite(ti[low]).sum()))
q = low & np.isfinite(ti)
if q.sum():
    P("   T_i there, quantiles 10,25,50,75,90 (eV):", np.round(np.percentile(ti[q], [10, 25, 50, 75, 90]), 0), "; discharges:", np.unique(shot[q]).size)
    P("   ratio to the discharge median T_i, quantiles 10,50,90:", np.round(np.nanpercentile(ti[q] / medt[q], [10, 50, 90]), 3))
fin = np.isfinite(ti)
P("all measured T_i (cut): share below 75 eV %.4f ; below 150 eV %.4f" % (np.mean(ti[fin] < 75), np.mean(ti[fin] < 150)))
hotd = fin & (medt > 400)
P("discharges with median above 400 eV: share of their measured rows below 150 eV %.4f ; below a third of the discharge median %.4f" % (np.mean(ti[hotd] < 150), np.mean(ti[hotd] < medt[hotd] / 3)))
