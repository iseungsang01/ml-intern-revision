import zlib
import numpy as np

SEED = 20261104
CHUNK = 500
FORMS = ("cluster", "discharge")
QUANTS = ("short_fast", "long_fast", "fast_budget", "history_budget", "delta")


def _ratios(t):
    # t[..., 4] sums in order SF, SH, LF, LH
    sf, sh, lf, lh = (t[..., k] for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 - sf / sh
        l = 1.0 - lf / lh
        return {"short_fast": s, "long_fast": l, "fast_budget": 1.0 - lf / sf,
                "history_budget": 1.0 - lh / sh, "delta": l - s}


def _interval(vals):
    v = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        raise ValueError("no finite bootstrap draws")
    lo, hi = np.quantile(v[ok], [0.025, 0.975])
    return [float(lo), float(hi)], int((~ok).sum())


def compute(cells, universe, draws=20000):
    draws = int(draws)
    assert draws > 0
    uni = {}
    for f in FORMS:
        u = np.asarray(universe[f])
        assert u.ndim == 1 and u.size > 0
        assert np.all(u[1:] > u[:-1]), "universe not sorted unique"
        uni[f] = u
    prep = {}
    for ep, c in cells.items():
        e = np.asarray(c["errors"], dtype=np.float64)
        assert e.ndim == 3 and e.shape[:2] == (3, 4) and e.shape[2] > 0
        assert np.all(np.isfinite(e)), "nonfinite errors"
        sq = e * e
        n = e.shape[2]
        agg = {}
        for f in FORMS:
            lab = np.asarray(c[f])
            assert lab.shape == (n,)
            u = uni[f]
            idx = np.searchsorted(u, lab)
            assert np.all(idx < u.size) and np.all(u[np.minimum(idx, u.size - 1)] == lab), "label missing"
            S = np.zeros((3, 4, u.size))
            for i in range(3):
                for v in range(4):
                    S[i, v] = np.bincount(idx, weights=sq[i, v], minlength=u.size)
            agg[f] = S
        prep[ep] = (sq.sum(axis=2), agg)
    fixed = {f: {ep: {q: np.empty(draws) for q in QUANTS} for ep in cells} for f in FORMS}
    mix = {f: {ep: {q: np.empty(draws) for q in QUANTS} for ep in cells} for f in FORMS}
    for f in FORMS:
        crc = zlib.crc32(f.encode())
        rng = np.random.default_rng([SEED, crc])
        picks = np.random.default_rng([SEED, 42, crc]).integers(0, 3, draws)
        L = uni[f].size
        p = np.full(L, 1.0 / L)
        for a in range(0, draws, CHUNK):
            b = min(a + CHUNK, draws)
            W = rng.multinomial(L, p, size=b - a).astype(np.float64)
            pk = picks[a:b]
            rows = np.arange(b - a)
            for ep in cells:
                S = prep[ep][1][f]
                T = np.einsum("cl,ivl->civ", W, S)
                R = _ratios(T)
                for q in QUANTS:
                    r = R[q]  # (c,3)
                    bad = ~np.all(np.isfinite(r), axis=1)
                    fx = r.mean(axis=1)
                    fx[bad] = np.nan
                    fixed[f][ep][q][a:b] = fx
                    mix[f][ep][q][a:b] = r[rows, pk]
    out = {}
    for ep in cells:
        tot = prep[ep][0]
        R = _ratios(tot)
        out[ep] = {}
        for q in QUANTS:
            per = [float(x) for x in R[q]]
            d = {"point": float(np.mean(R[q])), "per_init": per}
            for f in FORMS:
                fi, fu = _interval(fixed[f][ep][q])
                mi, mu = _interval(mix[f][ep][q])
                d[f] = {"fixed": fi, "mixture": mi,
                        "fixed_undefined": fu, "mixture_undefined": mu,
                        "fixed_excludes_zero": bool(fi[0] > 0 or fi[1] < 0),
                        "mixture_excludes_zero": bool(mi[0] > 0 or mi[1] < 0)}
            out[ep][q] = d
    return out
