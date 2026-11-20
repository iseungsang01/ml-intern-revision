"""B.82: the cluster bootstrap (registration section 2, Addendum 1 section 7).

Draw mechanics of b75_availability/readout.draws: under V1 (V2) a re-partition (an assignment) is drawn first for each
draw, then the units (shot clusters, or discharges) with at least one row in the cell are resampled with replacement as
multinomial counts.  A sum over rows becomes counts @ per-unit sums.  A quantile over rows becomes a weighted quantile
with the counts of the row's unit as integer weights (WQ below), identical to the quantile of the resampled rows.
"""
from __future__ import annotations

import numpy as np

import common82 as M

N = M.C.N_BOOT
FORMS = ("cluster", "discharge")          # order of the draws in every stream


def q(d):
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def draws(rng, nsub, nu):
    pick = rng.integers(0, nsub, N) if nsub > 1 else np.zeros(N, np.int64)
    out = []
    for si in range(nsub):
        sel = np.flatnonzero(pick == si)
        out.append((sel, rng.multinomial(nu, np.full(nu, 1.0 / nu), size=len(sel)).astype(np.float64)))
    return out


class Units:
    """Unit indices of the rows of a cell, per form."""

    def __init__(self, rows, unit_d, unit_c):
        self.n = len(rows)
        self.idx = {}
        for form in FORMS:
            uu = (unit_d if form == "discharge" else unit_c)[rows]
            u, inv = np.unique(uu, return_inverse=True)
            self.idx[form] = (inv, len(u))

    def sums(self, form, v):
        inv, nu = self.idx[form]
        return np.bincount(inv, v, nu)

    def count(self, form, mask=None):
        inv, nu = self.idx[form]
        if mask is None:
            return nu
        return int(len(np.unique(inv[mask])))


def boot(stream, U, nsub, vecs_of, stat):
    """Interval of a statistic of per-unit sums.  vecs_of(form, si) -> dict name -> per-unit vector; stat(dict) -> value
    (vectorised over draws).  Point: stat on the per-sub totals, mean over subs."""
    rng = M.rng_for(stream)
    out = {}
    nan_draws = {}
    for form in FORMS:
        _, nu = U.idx[form]
        W = draws(rng, nsub, nu)
        vals = np.empty(N)
        for si, (sel, Wm) in enumerate(W):
            v = vecs_of(form, si)
            vals[sel] = stat({k: Wm @ x for k, x in v.items()})
        out[form] = q(vals)
        nan_draws[form] = int((~np.isfinite(vals)).sum())
        del W
    pts = []
    for si in range(nsub):
        v = vecs_of("discharge", si)
        pts.append(float(stat({k: np.array([x.sum()]) for k, x in v.items()})[0]))
    res = {"point": float(np.mean(pts)), "discharge": out["discharge"], "cluster": out["cluster"], "stream": stream}
    if nsub > 1:
        res["per_split"] = pts
    if any(nan_draws.values()):
        res["draws_without_a_value"] = nan_draws
    return res


# ------------------------------------------------------------------------------------------ weighted quantiles
def lerp(a, b, t):
    """numpy's _lerp (linear method of np.quantile)."""
    diff = b - a
    out = a + diff * t
    return np.where(t >= 0.5, b - diff * (1 - t), out)


class WQ:
    """Quantiles of `vals` (rows of a cell) under integer unit weights, exact (linear method of np.quantile on the
    resampled rows).  Rows are sorted once; the cumulative weight at the end of K contiguous chunks of the sorted rows is
    counts @ (per-unit cumulative counts), and the crossing row is found inside its chunk."""

    def __init__(self, vals, inv, nu, K=400):
        o = np.argsort(vals, kind="stable")
        self.x = vals[o]
        self.u = inv[o]
        n = len(o)
        self.n, self.nu = n, nu
        K = max(1, min(K, n))
        self.bs = int(np.ceil(n / K))
        self.K = int(np.ceil(n / self.bs))
        chunk = np.arange(n) // self.bs
        H = np.zeros((nu, self.K))
        np.add.at(H, (self.u, chunk), 1.0)
        self.Hcum = np.cumsum(H, axis=1)              # nu x K: rows of unit up to the end of chunk j
        self.cnt = self.Hcum[:, -1].copy()

    def point(self, qs):
        return np.quantile(self.x, qs)

    def _at(self, Wm, CB, k):
        """Value of the repeated sorted array at index k (per draw)."""
        D = Wm.shape[0]
        j = (CB <= k[:, None]).sum(1)                  # first chunk whose cumulative weight exceeds k
        j = np.minimum(j, self.K - 1)
        base = np.where(j > 0, CB[np.arange(D), np.maximum(j - 1, 0)], 0.0)
        start = j * self.bs
        idx = start[:, None] + np.arange(self.bs)[None, :]
        valid = idx < self.n
        idxc = np.minimum(idx, self.n - 1)
        w = np.take_along_axis(Wm, self.u[idxc], axis=1) * valid
        cw = base[:, None] + np.cumsum(w, axis=1)
        pos = np.argmax(cw > k[:, None], axis=1)
        return self.x[np.minimum(start + pos, self.n - 1)]

    def draws(self, Wm, qs, chunk=4000):
        """D x len(qs) quantiles for the count matrix Wm (D x nu)."""
        D = Wm.shape[0]
        out = np.empty((D, len(qs)))
        for a in range(0, D, chunk):
            Wc = Wm[a:a + chunk]
            CB = Wc @ self.Hcum
            T = CB[:, -1]
            for jq, qq in enumerate(qs):
                h = (T - 1) * qq                                       # numpy 'linear': get_virtual_index = (n - 1) q
                lo = np.floor(h)
                g = h - lo
                hi = np.minimum(lo + 1, T - 1)
                top = h >= T - 1                                       # numpy: the largest value
                lo = np.where(top, T - 1, lo)
                hi = np.where(top, T - 1, hi)
                out[a:a + chunk, jq] = lerp(self._at(Wc, CB, lo), self._at(Wc, CB, hi), g)
        return out

    def direct(self, Wrow, qs):
        """The same quantiles from the explicitly resampled rows of one draw (verification)."""
        rep = np.repeat(self.x, Wrow[self.u].astype(np.int64))
        return np.quantile(rep, qs)
